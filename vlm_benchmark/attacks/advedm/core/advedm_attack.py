"""Paper-exact ADVEDM-A/-R attacks with attention reallocation and attention-weighted losses."""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
from PIL import Image
import torchvision.transforms as transforms

from .attention_utils import (
    extract_cls_to_patch_attention,
    reallocate_attention_vector,
    compute_attention_weighted_features_vector,
)
from .mask_utils import (
    compute_text_patch_similarity,
    construct_top_k_mask,
    construct_threshold_mask,
    create_masked_image,
)

_COS_EPS = 1e-6


def _safe_normalize(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable normalization used by cosine-style losses."""
    return F.normalize(x, dim=-1, eps=_COS_EPS)


def _cosine_mean(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Mean cosine similarity with stable epsilon."""
    return F.cosine_similarity(a, b, dim=-1, eps=_COS_EPS).mean()


def _sanitize_grad_inplace(grad: Optional[torch.Tensor]) -> bool:
    """Replace non-finite gradient values in-place; returns True if patched."""
    if grad is None or torch.isfinite(grad).all():
        return False
    grad.copy_(torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0))
    return True


def _sanitize_delta_inplace(delta: torch.Tensor, epsilon: float) -> bool:
    """Replace non-finite delta values in-place; returns True if patched."""
    if torch.isfinite(delta).all():
        return False
    delta.copy_(torch.nan_to_num(delta, nan=0.0, posinf=epsilon, neginf=-epsilon))
    return True


class ADVEDMSemanticAdditionPaperExact:
    """ADVEDM-A semantic addition attack (paper Equations 8-12)."""

    def __init__(
        self,
        vision_encoder: torch.nn.Module,
        reference_image_path: str,
        device: str = "cuda",
        lambda_cls: float = 0.8,
        lambda_preserve: float = 2.0,
        lambda_attention: float = 0.3,
        alpha: float = 0.5,
        beta: float = 0.4,
        norm_mean: Tuple[float, ...] = (0.48145466, 0.4578275, 0.40821073),
        norm_std: Tuple[float, ...] = (0.26862954, 0.26130258, 0.27577711),
        image_size: int = 224,
    ):
        """Initialize the paper-exact semantic addition attack."""
        self.device = device
        self.vision_encoder = vision_encoder
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.image_size = image_size

        self.lambda_cls = lambda_cls
        self.lambda_preserve = lambda_preserve
        self.lambda_attention = lambda_attention
        self.alpha = alpha
        self.beta = beta

        print(f"Loading reference image: {reference_image_path}")
        self.reference_patch_embeds, self.reference_cls_embed, self.reference_attention = \
            self._load_reference_image(reference_image_path)
        print(f"✓ Reference loaded:")
        print(f"  Patches: {self.reference_patch_embeds.shape}")
        print(f"  CLS: {self.reference_cls_embed.shape}")
        print(f"  Attention: {self.reference_attention.shape}")

    def _load_reference_image(
        self,
        path: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load the reference image and extract patch embeddings, CLS token, and CLS->patch attention."""
        ref_image = Image.open(path).convert("RGB")

        image_size = self.image_size

        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        ref_tensor = transform(ref_image).unsqueeze(0).to(self.device)

        mean = torch.tensor(self.norm_mean, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(self.norm_std, device=self.device).view(1, 3, 1, 1)
        ref_normalized = (ref_tensor - mean) / std

        vis = self.vision_encoder
        trunk = getattr(vis, "trunk", vis)
        is_openai_style = hasattr(trunk, "conv1")

        with torch.no_grad():
            if is_openai_style:
                attention_vec = extract_cls_to_patch_attention(vis, ref_normalized)

                conv1_dtype = trunk.conv1.weight.dtype
                x = trunk.conv1(ref_normalized.to(conv1_dtype))
                x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
                x = torch.cat([
                    trunk.class_embedding.to(x.dtype) +
                    torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                    x
                ], dim=1)
                x = x + trunk.positional_embedding.to(x.dtype)
                x = trunk.ln_pre(x)
                x = x.permute(1, 0, 2)
                x = trunk.transformer(x)
                x = x.permute(1, 0, 2)
                x = trunk.ln_post(x)
                x = x.float()

                cls_embed = x[:, 0, :]
                patch_embeds = x[:, 1:, :]
            else:
                # timm-style (SigLIP, etc.): patch_embed -> blocks -> norm.
                x = trunk.patch_embed(ref_normalized)
                has_cls = hasattr(trunk, "cls_token") and trunk.cls_token is not None
                if has_cls:
                    cls_token = trunk.cls_token.expand(x.shape[0], -1, -1)
                    x = torch.cat([cls_token, x], dim=1)
                if hasattr(trunk, "pos_embed") and trunk.pos_embed is not None:
                    x = x + trunk.pos_embed
                if hasattr(trunk, "patch_drop"):
                    x = trunk.patch_drop(x)
                if hasattr(trunk, "norm_pre"):
                    x = trunk.norm_pre(x)
                for blk in trunk.blocks:
                    x = blk(x)
                if hasattr(trunk, "norm"):
                    x = trunk.norm(x)
                x = x.float()

                if has_cls:
                    cls_embed = x[:, 0, :]
                    patch_embeds = x[:, 1:, :]
                else:
                    cls_embed = x.mean(dim=1)
                    patch_embeds = x

                # No CLS: use uniform attention.
                N = patch_embeds.shape[1]
                attention_vec = torch.ones(1, N, device=x.device, dtype=x.dtype) / N

            cls_embed = _safe_normalize(cls_embed)
            patch_embeds = _safe_normalize(patch_embeds)

        return patch_embeds.detach(), cls_embed.detach(), attention_vec.detach()

    def compute_cls_loss(
        self,
        cls_embed_adv: torch.Tensor,
        cls_embed_orig: torch.Tensor
    ) -> torch.Tensor:
        """L_cls (Eq.9): CLS fusion loss between clean and reference images."""
        cls_adv_norm = _safe_normalize(cls_embed_adv)
        cls_orig_norm = _safe_normalize(cls_embed_orig)
        cls_ref_norm = _safe_normalize(self.reference_cls_embed.to(cls_embed_adv.dtype))

        cls_fused = (1 - self.alpha) * cls_orig_norm + self.alpha * cls_ref_norm
        cls_fused_norm = _safe_normalize(cls_fused)

        similarity = F.cosine_similarity(
            cls_adv_norm, cls_fused_norm, dim=-1, eps=_COS_EPS
        )

        return -similarity.mean()

    def compute_local_injection_loss(
        self,
        patch_embeds_adv: torch.Tensor,
        A_adv: torch.Tensor,
        A_reallocated: torch.Tensor,
        target_indices: torch.Tensor
    ) -> torch.Tensor:
        """L_p (Eq.11): local injection loss over injection patches using attention-weighted features."""
        B = patch_embeds_adv.shape[0]
        if target_indices.dim() == 1:
            target_indices = target_indices.unsqueeze(0)
        if target_indices.shape[0] == 1 and B > 1:
            target_indices = target_indices.expand(B, -1)
        if target_indices.shape[0] != B:
            raise ValueError(
                f"target_indices batch size mismatch: got {target_indices.shape[0]}, expected {B}"
            )
        if target_indices.shape[1] == 0:
            raise ValueError("target_indices is empty; cannot compute injection loss.")
        injection_losses = []

        for b in range(B):
            weighted_adv = compute_attention_weighted_features_vector(
                A_adv[b:b+1],
                patch_embeds_adv[b:b+1]
            )

            ref_patches = self.reference_patch_embeds.to(patch_embeds_adv.dtype)
            weighted_ref = compute_attention_weighted_features_vector(
                A_reallocated[b:b+1],
                ref_patches
            )

            target_weighted_adv = weighted_adv[0, target_indices[b]]
            target_weighted_ref = weighted_ref[0, target_indices[b]]
            if target_weighted_adv.shape[0] == 0:
                raise ValueError("No injection patches selected for local injection loss.")

            patch_sim = _cosine_mean(target_weighted_adv, target_weighted_ref)

            injection_losses.append(-patch_sim)

        return torch.stack(injection_losses).mean()

    def compute_attention_fixation_loss(
        self,
        patch_embeds_adv: torch.Tensor,
        patch_embeds_orig: torch.Tensor,
        A_adv: torch.Tensor,
        A_reallocated: torch.Tensor,
        target_indices: torch.Tensor
    ) -> torch.Tensor:
        """L_fix (Eq.12): preserve non-target regions using attention-weighted features."""
        B, num_patches, D = patch_embeds_adv.shape
        if target_indices.dim() == 1:
            target_indices = target_indices.unsqueeze(0)
        if target_indices.shape[0] == 1 and B > 1:
            target_indices = target_indices.expand(B, -1)
        if target_indices.shape[0] != B:
            raise ValueError(
                f"target_indices batch size mismatch: got {target_indices.shape[0]}, expected {B}"
            )
        if target_indices.shape[1] == 0:
            raise ValueError("target_indices is empty; cannot compute fixation loss.")
        fixation_losses = []

        for b in range(B):
            weighted_adv = compute_attention_weighted_features_vector(
                A_adv[b:b+1],
                patch_embeds_adv[b:b+1]
            )[0]

            weighted_orig = compute_attention_weighted_features_vector(
                A_reallocated[b:b+1],
                patch_embeds_orig[b:b+1]
            )[0]

            mask = torch.ones(num_patches, dtype=torch.bool, device=patch_embeds_adv.device)
            mask[target_indices[b]] = False

            non_target_weighted_adv = weighted_adv[mask]
            non_target_weighted_orig = weighted_orig[mask]
            if non_target_weighted_adv.shape[0] == 0:
                raise ValueError("All patches are marked as target; no non-target patches to preserve.")

            similarity = _cosine_mean(non_target_weighted_adv, non_target_weighted_orig)

            fixation_losses.append(-similarity)

        return torch.stack(fixation_losses).mean()

    def compute_total_loss(
        self,
        patch_embeds_adv: torch.Tensor,
        cls_embed_adv: torch.Tensor,
        cls_embed_orig: torch.Tensor,
        patch_embeds_orig: torch.Tensor,
        A_adv: torch.Tensor,
        A_orig: torch.Tensor,
        target_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """Compute the Eq.8 total loss (w1*L_cls + w2*L_p + w3*L_fix) with attention reallocation."""
        B = patch_embeds_adv.shape[0]
        num_patches = patch_embeds_adv.shape[1]

        # Binary mask: 0 for target patches, 1 for others.
        if target_indices.dim() == 1:
            target_indices_batch = target_indices.unsqueeze(0).expand(B, -1)
        else:
            target_indices_batch = target_indices

        if target_indices_batch.shape[0] != B:
            raise ValueError(
                f"target_indices batch size mismatch: got {target_indices_batch.shape[0]}, expected {B}"
            )

        mask = torch.ones(B, num_patches, device=patch_embeds_adv.device, dtype=A_orig.dtype)
        for b in range(B):
            if target_indices_batch[b].numel() == 0:
                raise ValueError("target_indices is empty; cannot build Equation 10 mask.")
            if torch.any((target_indices_batch[b] < 0) | (target_indices_batch[b] >= num_patches)):
                raise ValueError(
                    f"target_indices out of valid patch range [0, {num_patches - 1}]"
                )
            mask[b, target_indices_batch[b]] = 0  # 0 for inject, 1 for preserve

        # Reallocate attention (Eq.10) using clean-image attention A_orig.
        A_ref = self.reference_attention.to(A_orig.dtype)
        if A_ref.shape[0] == 1 and B > 1:
            A_ref = A_ref.expand(B, -1)
        if A_orig.shape != A_ref.shape or A_orig.shape != mask.shape:
            raise ValueError(
                f"Attention/mask shape mismatch: A_orig={tuple(A_orig.shape)}, "
                f"A_ref={tuple(A_ref.shape)}, mask={tuple(mask.shape)}"
            )

        A_reallocated = reallocate_attention_vector(
            A_orig,
            A_ref,
            mask,
            beta=self.beta
        )

        L_cls = self.compute_cls_loss(cls_embed_adv, cls_embed_orig)

        L_p = self.compute_local_injection_loss(
            patch_embeds_adv,
            A_adv,
            A_reallocated,
            target_indices
        )

        L_fix = self.compute_attention_fixation_loss(
            patch_embeds_adv,
            patch_embeds_orig,
            A_adv,
            A_reallocated,
            target_indices
        )

        total = (
            self.lambda_cls * L_cls +
            self.lambda_preserve * L_p +
            self.lambda_attention * L_fix
        )

        return total, {
            "total": total.item(),
            "cls": L_cls.item(),
            "reference_sim": L_p.item(),
            "attention_fix": L_fix.item(),
        }


def adam_attack_advedm_a_paper_exact(
    image: torch.Tensor,
    vision_encoder: torch.nn.Module,
    attack: ADVEDMSemanticAdditionPaperExact,
    target_indices: torch.Tensor,
    epsilon: float = 8/255,
    num_iters: int = 500,
    learning_rate: float = 0.005,
    constraint: str = "l2",  # "l2" (paper) or "linf"
    verbose: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, list]:
    """Paper-exact ADVEDM-A optimization with per-iteration attention extraction."""
    device = image.device
    B, C, H, W = image.shape

    delta = torch.zeros_like(image, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=learning_rate)

    mean = torch.tensor([0.48145466, 0.45782750, 0.40821073]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)
    image_normalized = (image - mean) / std

    with torch.no_grad():
        # CLS->patch attention from the clean image, reused for reallocation (Eq.10).
        A_orig = extract_cls_to_patch_attention(vision_encoder, image_normalized)

        conv1_dtype = vision_encoder.conv1.weight.dtype
        x = vision_encoder.conv1(image_normalized.to(conv1_dtype))
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        x = torch.cat([
            vision_encoder.class_embedding.to(x.dtype) +
            torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x
        ], dim=1)
        x = x + vision_encoder.positional_embedding.to(x.dtype)
        x = vision_encoder.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = vision_encoder.transformer(x)
        x = x.permute(1, 0, 2)
        x = vision_encoder.ln_post(x)
        x = x.float()  # cast FP16->FP32 before normalization to avoid NaN
        cls_embed_orig = _safe_normalize(x[:, 0, :])
        patch_embeds_orig = _safe_normalize(x[:, 1:, :])

    loss_history = []

    if verbose:
        if target_indices.dim() == 1:
            target_patch_count = target_indices.shape[0]
        else:
            target_patch_count = target_indices.shape[1]
        print(f"\nStarting paper-exact optimization...")
        print(f"  Iterations: {num_iters}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Epsilon: {epsilon:.4f} ({int(epsilon*255)}/255)")
        print(f"  Target patches: {target_patch_count}")
        print(f"  CLS→patch attention shape: {A_orig.shape}")

    for iteration in range(num_iters):
        optimizer.zero_grad()

        adv_image = image + delta
        adv_image_normalized = (adv_image - mean) / std

        # Adversarial attention as fixed per-iteration weights (detached for stability).
        A_adv = extract_cls_to_patch_attention(
            vision_encoder, adv_image_normalized
        ).detach()

        x = vision_encoder.conv1(adv_image_normalized.to(conv1_dtype))
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        x = torch.cat([
            vision_encoder.class_embedding.to(x.dtype) +
            torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x
        ], dim=1)
        x = x + vision_encoder.positional_embedding.to(x.dtype)
        x = vision_encoder.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = vision_encoder.transformer(x)
        x = x.permute(1, 0, 2)
        x = vision_encoder.ln_post(x)
        x = x.float()  # cast FP16->FP32 before normalization to avoid NaN under autograd
        cls_embed_adv = _safe_normalize(x[:, 0, :])
        patch_embeds_adv = _safe_normalize(x[:, 1:, :])

        loss, loss_dict = attack.compute_total_loss(
            patch_embeds_adv,
            cls_embed_adv,
            cls_embed_orig,
            patch_embeds_orig,
            A_adv,
            A_orig,
            target_indices
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite ADVEDM-A loss at iteration {iteration}: {loss_dict}"
            )

        loss.backward()
        if _sanitize_grad_inplace(delta.grad) and verbose:
            print(f"  [warn] Sanitized non-finite ADVEDM-A gradient at iter {iteration}.")
        optimizer.step()

        with torch.no_grad():
            if _sanitize_delta_inplace(delta, epsilon) and verbose:
                print(f"  [warn] Sanitized non-finite ADVEDM-A delta at iter {iteration}.")
            if constraint == "linf":
                delta_min = torch.max(-epsilon * torch.ones_like(image), -image)
                delta_max = torch.min(epsilon * torch.ones_like(image), 1 - image)
                delta.copy_(torch.clamp(delta, delta_min, delta_max))
            elif constraint == "l2":
                delta_norm = torch.norm(delta.reshape(B, -1), p=2, dim=1, keepdim=True)
                scale = torch.clamp(epsilon / (delta_norm + 1e-8), max=1.0)
                delta.copy_(delta * scale.view(B, 1, 1, 1))

                adv_clamped = torch.clamp(image + delta, 0.0, 1.0)
                delta.copy_(adv_clamped - image)
            else:
                raise ValueError(f"Unsupported constraint '{constraint}'. Use 'l2' or 'linf'.")

        loss_history.append(loss_dict)

        if verbose and (iteration % 50 == 0 or iteration == num_iters - 1):
            print(f"  Iter {iteration:3d}: Loss={loss_dict['total']:.4f} | "
                  f"L_cls={loss_dict['cls']:.4f} | "
                  f"L_p={loss_dict['reference_sim']:.4f} | "
                  f"L_fix={loss_dict['attention_fix']:.4f}")

    if verbose:
        final_perturbation = delta.detach().abs().max().item()
        print(f"\n✓ Optimization complete")
        print(f"  Final perturbation: {final_perturbation:.6f} ({final_perturbation*255:.2f}/255)")

    return image + delta.detach(), delta.detach(), loss_history


# ============================================================================
# ADVEDM-R: Semantic Removal Attack (Paper-Faithful Implementation)
# ============================================================================

class ADVEDMSemanticRemovalPaperExact:
    """ADVEDM-R semantic removal attack via text-guided masking (paper Equations 3-8)."""

    def __init__(
        self,
        vision_encoder,
        text_encoder,
        target_text: str,
        lambda_cls: float = 0.5,     # w1 (Eq.8)
        lambda_local: float = 2.0,    # w2 (Eq.8)
        lambda_fix: float = 0.2,      # w3 (Eq.8)
        k_ratio: float = 0.2,         # Top 20% for removal
        mask_threshold: Optional[float] = None,  # Eq.4 threshold xi (optional)
        device: str = "cuda"
    ):
        """Initialize the ADVEDM-R attack."""
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.target_text = target_text
        self.lambda_cls = lambda_cls
        self.lambda_local = lambda_local
        self.lambda_fix = lambda_fix
        self.k_ratio = k_ratio
        self.mask_threshold = mask_threshold
        self.device = device

        self.target_text_embed = self._encode_target_text()

    def _encode_target_text(self) -> torch.Tensor:
        """Encode the target text to an embedding [1, 768]."""
        text_features = self.text_encoder.encode_text([self.target_text])
        return text_features

    def compute_cls_loss(
        self,
        cls_embed_adv: torch.Tensor  # [B, D]
    ) -> torch.Tensor:
        """L_cls (Eq.5): positive similarity to the target text, minimized to remove semantics."""
        cls_norm = _safe_normalize(cls_embed_adv)
        text_norm = _safe_normalize(self.target_text_embed.to(cls_embed_adv.dtype))

        similarity = F.cosine_similarity(cls_norm, text_norm, dim=-1, eps=_COS_EPS)

        return similarity.mean()

    def compute_local_removal_loss(
        self,
        patch_embeds_adv: torch.Tensor,  # [B, 576, D]
        patch_embeds_masked: torch.Tensor,  # [B, 576, D]
        mask: torch.Tensor  # [B, 576]
    ) -> torch.Tensor:
        """L_p (Eq.6): maximize similarity to masked patches in the removal region."""
        removal_losses = []
        B = patch_embeds_adv.shape[0]

        for b in range(B):
            removal_indices = (mask[b] == 0).nonzero(as_tuple=True)[0]
            if removal_indices.numel() == 0:
                raise ValueError("No removal patches found; adjust k_ratio/mask_threshold.")

            adv_removal = patch_embeds_adv[b, removal_indices]
            masked_removal = patch_embeds_masked[b, removal_indices]

            similarity = _cosine_mean(adv_removal, masked_removal)

            removal_losses.append(-similarity)

        return torch.stack(removal_losses).mean()

    def compute_attention_fixation_loss(
        self,
        patch_embeds_adv: torch.Tensor,   # [B, 576, D]
        patch_embeds_orig: torch.Tensor,  # [B, 576, D]
        A_adv: torch.Tensor,              # [B, 576]
        A_orig: torch.Tensor,             # [B, 576]
        mask: torch.Tensor                # [B, 576]
    ) -> torch.Tensor:
        """L_fix (Eq.7): preserve non-removal regions using clean attention weighting (no reallocation)."""
        fixation_losses = []
        B = patch_embeds_adv.shape[0]

        for b in range(B):
            weighted_adv = compute_attention_weighted_features_vector(
                A_adv[b:b+1], patch_embeds_adv[b:b+1]
            )[0]

            weighted_orig = compute_attention_weighted_features_vector(
                A_orig[b:b+1], patch_embeds_orig[b:b+1]
            )[0]

            preserve_indices = (mask[b] == 1).nonzero(as_tuple=True)[0]
            if preserve_indices.numel() == 0:
                raise ValueError("No preserve patches found; adjust k_ratio/mask_threshold.")

            preserve_adv = weighted_adv[preserve_indices]
            preserve_orig = weighted_orig[preserve_indices]

            similarity = _cosine_mean(preserve_adv, preserve_orig)

            fixation_losses.append(-similarity)

        return torch.stack(fixation_losses).mean()

    def compute_total_loss(
        self,
        cls_embed_adv: torch.Tensor,
        patch_embeds_adv: torch.Tensor,
        patch_embeds_orig: torch.Tensor,
        patch_embeds_masked: torch.Tensor,
        A_adv: torch.Tensor,
        A_orig: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """Compute the Eq.8 total loss (w1*L_cls + w2*L_p + w3*L_fix)."""
        L_cls = self.compute_cls_loss(cls_embed_adv)

        L_p = self.compute_local_removal_loss(
            patch_embeds_adv, patch_embeds_masked, mask
        )

        L_fix = self.compute_attention_fixation_loss(
            patch_embeds_adv, patch_embeds_orig, A_adv, A_orig, mask
        )

        total_loss = (
            self.lambda_cls * L_cls +
            self.lambda_local * L_p +
            self.lambda_fix * L_fix
        )

        return total_loss, {
            'total': total_loss.item(),
            'cls': L_cls.item(),
            'local': L_p.item(),
            'fix': L_fix.item()
        }


def extract_patch_and_cls_embeddings(vision_encoder, image_normalized):
    """Extract normalized patch and CLS embeddings from an OpenAI-CLIP vision encoder."""
    conv1_dtype = vision_encoder.conv1.weight.dtype
    x = vision_encoder.conv1(image_normalized.to(conv1_dtype))
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    x = torch.cat([
        vision_encoder.class_embedding.to(x.dtype) +
        torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
        x
    ], dim=1)
    x = x + vision_encoder.positional_embedding.to(x.dtype)
    x = vision_encoder.ln_pre(x)
    x = x.permute(1, 0, 2)
    x = vision_encoder.transformer(x)
    x = x.permute(1, 0, 2)
    x = vision_encoder.ln_post(x)
    x = x.float()  # cast FP16->FP32 before normalization to avoid NaN

    cls_embed = _safe_normalize(x[:, 0, :])
    patch_embeds = _safe_normalize(x[:, 1:, :])

    return patch_embeds, cls_embed


def adam_attack_advedm_r_paper_exact(
    image: torch.Tensor,
    vision_encoder: torch.nn.Module,
    attack: ADVEDMSemanticRemovalPaperExact,
    vision_backend=None,
    epsilon: float = 8/255,
    num_iters: int = 500,
    learning_rate: float = 0.005,
    constraint: str = "l2",  # "l2" (paper) or "linf"
    verbose: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, list]:
    """ADVEDM-R optimization with text-guided masking (mask and masked image computed once from the clean image)."""
    device = image.device
    B, C, H, W = image.shape

    delta = torch.zeros_like(image, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=learning_rate)

    mean = torch.tensor([0.48145466, 0.45782750, 0.40821073]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)
    image_normalized = (image - mean) / std

    def _extract_attention(img: torch.Tensor, img_normalized: torch.Tensor) -> torch.Tensor:
        if vision_backend is not None:
            return vision_backend.extract_cls_to_patch_attention(img, normalized=False)
        return extract_cls_to_patch_attention(vision_encoder, img_normalized)

    def _extract_patch_cls(
        img: torch.Tensor,
        img_normalized: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract projected 768-dim features used only for masking and L_cls."""
        if vision_backend is not None:
            return vision_backend.extract_patch_cls_embeddings(img, normalized=False)
        return extract_patch_and_cls_embeddings(vision_encoder, img_normalized)

    def _extract_spatial_feats(
        img: torch.Tensor,
        img_normalized: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract pre-projection 1024-dim features (the space LLaVA's projector consumes) for L_p and L_fix."""
        if vision_backend is not None:
            return vision_backend.extract_patch_cls_embeddings_noproj(img, normalized=False)
        return extract_patch_and_cls_embeddings(vision_encoder, img_normalized)

    # CLIP visual projection (1024-dim pre-proj CLS -> 768-dim contrastive space) for L_cls.
    _clip_visual_proj: Optional[torch.Tensor] = None
    if vision_backend is not None:
        _clip_visual_proj = getattr(vision_backend, "_alignment_proj", None)
        if _clip_visual_proj is None:
            _clip_visual_proj = getattr(
                getattr(vision_backend, "vision_encoder", None), "proj", None
            )
    else:
        _clip_visual_proj = getattr(vision_encoder, "proj", None)

    # Extract clean-image features once.
    with torch.no_grad():
        A_orig = _extract_attention(image, image_normalized)

        # Masking uses projected 768-dim features to match the text contrastive space.
        patch_embeds_for_mask, _ = _extract_patch_cls(image, image_normalized)

        S = compute_text_patch_similarity(patch_embeds_for_mask, attack.target_text_embed)
        if attack.mask_threshold is not None:
            mask = construct_threshold_mask(S, threshold=attack.mask_threshold)
        else:
            mask = construct_top_k_mask(S, k_ratio=attack.k_ratio)

        # Spatial losses use pre-proj 1024-dim features (the space LLaVA consumes).
        patch_embeds_orig, _ = _extract_spatial_feats(image, image_normalized)

        # Masked image M is built from the clean image and fixed throughout (Eq.4, 6).
        if vision_backend is not None:
            patch_size = int(vision_backend.patch_size)
        else:
            patch_size = int(vision_encoder.conv1.kernel_size[0])
        image_size = H

        masked_image = create_masked_image(
            image, mask,
            patch_size=patch_size,
            image_size=image_size
        )

        patch_embeds_masked, _ = _extract_spatial_feats(
            masked_image,
            (masked_image - mean) / std,
        )

    loss_history = []

    if verbose:
        print(f"\nStarting ADVEDM-R optimization...")
        print(f"  Target text: '{attack.target_text}'")
        removal_count = int((mask == 0).sum().item())
        total_count = int(mask.numel())
        removal_ratio = 100.0 * removal_count / max(total_count, 1)
        if attack.mask_threshold is None:
            print(f"  Removal patches: {removal_count}/{total_count} ({removal_ratio:.1f}%, top-k mode)")
        else:
            print(
                f"  Removal patches: {removal_count}/{total_count} ({removal_ratio:.1f}%, "
                f"threshold mode xi={attack.mask_threshold})"
            )
        print(f"  Constraint: {constraint.upper()} (ε={epsilon:.4f})")

    for iteration in range(num_iters):
        optimizer.zero_grad()

        adv_image = image + delta
        adv_image_normalized = (adv_image - mean) / std

        # Adversarial attention as fixed per-iteration weights (detached for stability).
        A_adv = _extract_attention(adv_image, adv_image_normalized).detach()

        patch_embeds_adv, cls_embed_adv_spatial = _extract_spatial_feats(
            adv_image, adv_image_normalized
        )

        # Project pre-proj CLS to the 768-dim contrastive space for L_cls text comparison.
        if _clip_visual_proj is not None:
            cls_proj = cls_embed_adv_spatial.float() @ _clip_visual_proj.to(
                device=cls_embed_adv_spatial.device, dtype=cls_embed_adv_spatial.dtype
            )
            cls_embed_adv = _safe_normalize(cls_proj.float())
        else:
            # Fallback: 1024-dim CLS; caller should ensure a projection is available.
            cls_embed_adv = cls_embed_adv_spatial

        loss, loss_dict = attack.compute_total_loss(
            cls_embed_adv,
            patch_embeds_adv,
            patch_embeds_orig,
            patch_embeds_masked,
            A_adv,
            A_orig,
            mask
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite ADVEDM-R loss at iteration {iteration}: {loss_dict}"
            )

        loss.backward()
        if _sanitize_grad_inplace(delta.grad) and verbose:
            print(f"  [warn] Sanitized non-finite ADVEDM-R gradient at iter {iteration}.")
        optimizer.step()

        with torch.no_grad():
            if _sanitize_delta_inplace(delta, epsilon) and verbose:
                print(f"  [warn] Sanitized non-finite ADVEDM-R delta at iter {iteration}.")
            if constraint == "linf":
                delta_min = torch.max(-epsilon * torch.ones_like(image), -image)
                delta_max = torch.min(epsilon * torch.ones_like(image), 1 - image)
                delta.copy_(torch.clamp(delta, delta_min, delta_max))
            elif constraint == "l2":
                delta_norm = torch.norm(delta.reshape(B, -1), p=2, dim=1, keepdim=True)
                scale = torch.clamp(epsilon / (delta_norm + 1e-8), max=1.0)
                delta.copy_(delta * scale.view(B, 1, 1, 1))

                adv_clamped = torch.clamp(image + delta, 0.0, 1.0)
                delta.copy_(adv_clamped - image)
            else:
                raise ValueError(f"Unsupported constraint '{constraint}'. Use 'l2' or 'linf'.")

        loss_history.append(loss_dict)

        if verbose and (iteration % 50 == 0 or iteration == num_iters - 1):
            print(f"  Iter {iteration:3d}: Loss={loss_dict['total']:.4f} | "
                  f"L_cls={loss_dict['cls']:.4f} | "
                  f"L_p={loss_dict['local']:.4f} | "
                  f"L_fix={loss_dict['fix']:.4f}")

    adv_image = (image + delta).detach()
    delta_final = delta.detach()
    return adv_image, delta_final, loss_history
