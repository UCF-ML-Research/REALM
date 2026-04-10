"""
ADVEDM-A: Paper-Exact Implementation with Attention Reallocation

This implements the full ADVEDM-A method from the paper including:
- Equation 10: Attention reallocation
- Equation 11: L_p with attention-weighted features
- Equation 12: L_fix with attention-weighted features

Key difference from previous version:
- Extracts ACTUAL attention weights from vision encoder
- Implements proper attention reallocation (β * A_R for targets, (1-β) * A_I for others)
- Uses attention-weighted embeddings in loss functions
"""

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
    """
    ADVEDM-A: Paper-exact implementation matching Equations 8-12

    This implementation EXACTLY follows the paper equations:

    Equation 9 (L_cls): Global semantic guidance via CLS fusion
        L_cls = -CS([cls]_I', (1-α)*[cls]_I + α*[cls]_R)
        - Uses CLS tokens from adversarial, clean, and reference images
        - α=0.5: fusion weight
        - NO text encoder used!

    Equation 10: Attention reallocation
        A'_i = β*A_{R,i}        if mask_i=0 (injection patches)
               (1-β)*A_{I,i}    if mask_i=1 (other patches)
        - β=0.4: reallocation strength

    Equation 11 (L_p): Local injection loss
        L_p = -(1-mask) * CS(A_I' * [patch]_I', A' * [patch]_R)
        - Attention-weighted patch features
        - ONLY injection patches (1-mask)
        - NO CLS term, NO alpha

    Equation 12 (L_fix): Attention fixation for preservation
        L_fix = -mask * CS(A_I' * [patch]_I', A' * [patch]_I)
        - Preserves non-injection regions
        - Uses reallocated attention

    Equation 8 (Total loss): ONLY 3 terms
        min w1 * L_cls + w2 * L_p + w3 * L_fix
        - w1=0.8, w2=2.0, w3=0.3 for ADVEDM-A
        - NO extra preservation loss!
    """

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
        """
        Initialize paper-exact semantic addition attack

        Args:
            vision_encoder: CLIP vision encoder (must have transformer structure)
            reference_image_path: Path to reference image containing target object
            device: Device
            lambda_cls: w1 in paper (0.8 for ADVEDM-A)
            lambda_preserve: w2 in paper (2.0)
            lambda_attention: w3 in paper (0.3 for ADVEDM-A)
            alpha: CLS fusion weight (0.5) - used in Eq.9 for CLS fusion
            beta: Attention reallocation factor (0.4) - used in Eq.10

        Note:
            Text encoder is NOT used in ADVEDM-A losses (Equations 9-12).
            The reference image R should be generated offline using text-to-image
            and contain only the target object.
        """
        self.device = device
        self.vision_encoder = vision_encoder
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.image_size = image_size

        # Hyperparameters
        self.lambda_cls = lambda_cls
        self.lambda_preserve = lambda_preserve
        self.lambda_attention = lambda_attention
        self.alpha = alpha
        self.beta = beta

        # Load reference image and extract embeddings + attention
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
        """
        Load reference image and extract (Paper-Faithful):
        1. Patch embeddings [1, num_patches, D]
        2. CLS token [1, D]
        3. CLS→patch attention [1, 576] (for Equation 10)

        Args:
            path: Path to reference image

        Returns:
            - Patch embeddings [1, 576, 1024]  (pre-proj, matches LLaVA tower dim)
            - CLS token [1, 1024]
            - CLS→patch attention [1, 576]
        """
        ref_image = Image.open(path).convert("RGB")

        image_size = self.image_size

        # Preprocess
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        ref_tensor = transform(ref_image).unsqueeze(0).to(self.device)

        # Normalize with model-specific mean/std
        mean = torch.tensor(self.norm_mean, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(self.norm_std, device=self.device).view(1, 3, 1, 1)
        ref_normalized = (ref_tensor - mean) / std

        # Extract attention and embeddings
        vis = self.vision_encoder
        trunk = getattr(vis, "trunk", vis)
        is_openai_style = hasattr(trunk, "conv1")

        with torch.no_grad():
            if is_openai_style:
                # OpenAI CLIP / open_clip ViT with conv1 structure
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
                # timm-style (SigLIP, etc.): patch_embed → blocks → norm
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

                # No CLS → uniform attention
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
        """
        L_cls (Equation 9): CLS fusion between clean and reference images

        Paper Equation 9:
        L_cls = -CS([cls]_I', (1-α)*[cls]_I + α*[cls]_R)

        Where:
        - [cls]_I': CLS token from adversarial image
        - [cls]_I: CLS token from original (clean) image
        - [cls]_R: CLS token from reference image
        - α: fusion weight (0.5 in paper)

        NOTE: Text encoder is NOT used in this loss!

        Args:
            cls_embed_adv: Adversarial CLS embedding [B, D]
            cls_embed_orig: Original (clean) CLS embedding [B, D]

        Returns:
            CLS loss (negative, to maximize similarity)
        """
        # Normalize CLS embeddings
        cls_adv_norm = _safe_normalize(cls_embed_adv)
        cls_orig_norm = _safe_normalize(cls_embed_orig)
        cls_ref_norm = _safe_normalize(self.reference_cls_embed.to(cls_embed_adv.dtype))

        # CLS fusion: (1-α)*[cls]_I + α*[cls]_R
        cls_fused = (1 - self.alpha) * cls_orig_norm + self.alpha * cls_ref_norm
        cls_fused_norm = _safe_normalize(cls_fused)

        # Cosine similarity: CS([cls]_I', fused)
        similarity = F.cosine_similarity(
            cls_adv_norm, cls_fused_norm, dim=-1, eps=_COS_EPS
        )

        # Negative (to maximize)
        return -similarity.mean()

    def compute_local_injection_loss(
        self,
        patch_embeds_adv: torch.Tensor,
        A_adv: torch.Tensor,
        A_reallocated: torch.Tensor,
        target_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        L_p (Equation 11): Local injection loss with attention-weighted features (Paper-Faithful)

        Paper Equation 11:
        L_p = -(1-mask) * CS(A_I' * [patch]_I', A' * [patch]_R)

        Where:
        - A_I': CLS→patch attention from adversarial image [B, 576]
        - [patch]_I': Patch embeddings from adversarial image
        - A': Reallocated attention [B, 576]
        - [patch]_R: Patch embeddings from reference image
        - (1-mask): Selects ONLY injection (target) patches

        NOTE: NO CLS term, NO alpha fusion in this loss!
        Only attention-weighted patch features.

        Args:
            patch_embeds_adv: Adversarial patches [B, 576, D]
            A_adv: CLS→patch attention from adversarial [B, 576]
            A_reallocated: Reallocated attention [B, 576]
            target_indices: Target patch indices [k] or [B, k]

        Returns:
            Local injection loss (negative, to maximize similarity)
        """
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
            # A_I' * [patch]_I' (adversarial, attention-weighted)
            weighted_adv = compute_attention_weighted_features_vector(
                A_adv[b:b+1],
                patch_embeds_adv[b:b+1]
            )  # [1, 576, D]

            # A' * [patch]_R (reference, using reallocated attention)
            ref_patches = self.reference_patch_embeds.to(patch_embeds_adv.dtype)
            weighted_ref = compute_attention_weighted_features_vector(
                A_reallocated[b:b+1],
                ref_patches
            )  # [1, 576, D]

            # Select ONLY injection patches ((1-mask) in paper)
            target_weighted_adv = weighted_adv[0, target_indices[b]]  # [k, D]
            target_weighted_ref = weighted_ref[0, target_indices[b]]  # [k, D]
            if target_weighted_adv.shape[0] == 0:
                raise ValueError("No injection patches selected for local injection loss.")

            # Cosine similarity (ONLY patch-level, no CLS)
            patch_sim = _cosine_mean(target_weighted_adv, target_weighted_ref)

            injection_losses.append(-patch_sim)  # Negative to maximize

        return torch.stack(injection_losses).mean()

    def compute_attention_fixation_loss(
        self,
        patch_embeds_adv: torch.Tensor,
        patch_embeds_orig: torch.Tensor,
        A_adv: torch.Tensor,
        A_reallocated: torch.Tensor,
        target_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        L_fix (Equation 12): Preserve non-target regions with attention-weighted features (Paper-Faithful)

        Paper: L_fix = -mask * CS(A_I' * [patch]_I', A' * [patch]_I)

        Where:
        - A_I': CLS→patch attention from adversarial image [B, 576]
        - A': Reallocated attention [B, 576]
        - mask: Non-target patches (to preserve)

        Args:
            patch_embeds_adv: Adversarial patches [B, 576, D]
            patch_embeds_orig: Original patches [B, 576, D]
            A_adv: CLS→patch attention from adversarial [B, 576]
            A_reallocated: Reallocated attention [B, 576]
            target_indices: Target patch indices [k] or [B, k]

        Returns:
            Fixation loss (negative, to maximize preservation)
        """
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
            # Attention-weighted features (Paper-Faithful: per Equation 12)
            # A_I' * [patch]_I' (adversarial, attention-weighted)
            weighted_adv = compute_attention_weighted_features_vector(
                A_adv[b:b+1],
                patch_embeds_adv[b:b+1]
            )[0]  # [576, D]

            # A' * [patch]_I (original, using reallocated attention)
            weighted_orig = compute_attention_weighted_features_vector(
                A_reallocated[b:b+1],
                patch_embeds_orig[b:b+1]
            )[0]  # [576, D]

            # Select NON-TARGET patches (mask=1 in paper)
            mask = torch.ones(num_patches, dtype=torch.bool, device=patch_embeds_adv.device)
            mask[target_indices[b]] = False  # False for targets, True for non-targets

            non_target_weighted_adv = weighted_adv[mask]  # [num_non_target, D]
            non_target_weighted_orig = weighted_orig[mask]  # [num_non_target, D]
            if non_target_weighted_adv.shape[0] == 0:
                raise ValueError("All patches are marked as target; no non-target patches to preserve.")

            # Cosine similarity (maximize to preserve)
            similarity = _cosine_mean(non_target_weighted_adv, non_target_weighted_orig)

            fixation_losses.append(-similarity)  # Negative to maximize

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
        """
        Compute total loss with attention reallocation (Paper-Faithful)

        Paper Equation 8:
        L_total = w1 * L_cls + w2 * L_p + w3 * L_fix

        ONLY 3 terms (no extra preservation loss)!

        CRITICAL FIX: Uses A_orig (clean image attention) for reallocation, NOT A_adv!

        Args:
            patch_embeds_adv: Adversarial patches [B, 576, D]
            cls_embed_adv: Adversarial CLS [B, D]
            cls_embed_orig: Original (clean) CLS [B, D]
            patch_embeds_orig: Original patches [B, 576, D]
            A_adv: CLS→patch attention from adversarial image [B, 576]
            A_orig: CLS→patch attention from CLEAN image [B, 576] (CRITICAL!)
            target_indices: Target patch indices [k] or [B, k]

        Returns:
            - Total loss
            - Loss components dict
        """
        B = patch_embeds_adv.shape[0]
        num_patches = patch_embeds_adv.shape[1]

        # Create binary mask: 0 for target patches, 1 for others
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

        # Reallocate attention (Equation 10) - CRITICAL: Uses A_orig, NOT A_adv!
        A_ref = self.reference_attention.to(A_orig.dtype)
        if A_ref.shape[0] == 1 and B > 1:
            A_ref = A_ref.expand(B, -1)
        if A_orig.shape != A_ref.shape or A_orig.shape != mask.shape:
            raise ValueError(
                f"Attention/mask shape mismatch: A_orig={tuple(A_orig.shape)}, "
                f"A_ref={tuple(A_ref.shape)}, mask={tuple(mask.shape)}"
            )

        A_reallocated = reallocate_attention_vector(
            A_orig,  # Clean image attention (CORRECT!)
            A_ref,
            mask,
            beta=self.beta
        )

        # L_cls: CLS fusion (Equation 9)
        # Uses (1-α)*[cls]_I + α*[cls]_R, NOT text embedding
        L_cls = self.compute_cls_loss(cls_embed_adv, cls_embed_orig)

        # L_p: Local injection with attention-weighted features (Equation 11)
        # ONLY patch-level, no CLS or alpha
        L_p = self.compute_local_injection_loss(
            patch_embeds_adv,
            A_adv,  # Adversarial attention for weighting
            A_reallocated,
            target_indices
        )

        # L_fix: Attention fixation with reallocated attention (Equation 12)
        L_fix = self.compute_attention_fixation_loss(
            patch_embeds_adv,
            patch_embeds_orig,
            A_adv,  # Adversarial attention for weighting
            A_reallocated,
            target_indices
        )

        # Total loss (Equation 8): ONLY 3 terms
        total = (
            self.lambda_cls * L_cls +
            self.lambda_preserve * L_p +  # w2 in paper
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
    """
    Paper-exact ADVEDM-A optimization with attention extraction

    Key differences from previous version:
    - Extracts attention weights from vision encoder each iteration
    - Uses attention reallocation per Equation 10
    - Computes attention-weighted features in losses

    Args:
        image: Original image [B, C, H, W] in [0, 1]
        vision_encoder: CLIP vision encoder
        attack: Paper-exact attack instance
        target_indices: Patch indices to inject [k]
        epsilon: Perturbation budget (8/255)
        num_iters: Number of iterations (500)
        learning_rate: Adam learning rate (0.005)
        constraint: "l2" (paper default) or "linf" (ablation)
        verbose: Print progress

    Returns:
        - Adversarial image [B, C, H, W]
        - Perturbation [B, C, H, W]
        - Loss history
    """
    device = image.device
    B, C, H, W = image.shape

    # Initialize perturbation
    delta = torch.zeros_like(image, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=learning_rate)

    # Preprocess original image and extract features + attention
    mean = torch.tensor([0.48145466, 0.45782750, 0.40821073]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)
    image_normalized = (image - mean) / std

    with torch.no_grad():
        # Paper-Faithful: Extract CLS→patch attention from CLEAN image
        # This is extracted ONCE and used for reallocation throughout (Equation 10)
        A_orig = extract_cls_to_patch_attention(vision_encoder, image_normalized)  # [B, 576]

        # Extract CLS token and patch embeddings from original image (full forward pass)
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
        x = x.float()  # cast FP16→FP32 before normalization to avoid NaN
        cls_embed_orig = _safe_normalize(x[:, 0, :])  # CLS token
        patch_embeds_orig = _safe_normalize(x[:, 1:, :])  # Patch tokens

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

        # Forward pass: adversarial image
        adv_image = image + delta
        adv_image_normalized = (adv_image - mean) / std

        # Paper-Faithful: Extract CLS→patch attention from ADVERSARIAL image
        # This is extracted EACH iteration and used for weighting in losses
        # Treat attention as per-iteration weighting (Eq.10-12) and stop unstable
        # higher-order gradients through manual attention extraction.
        A_adv = extract_cls_to_patch_attention(
            vision_encoder, adv_image_normalized
        ).detach()  # [B, 576]

        # Extract CLS token and patch embeddings (full forward)
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
        x = x.float()  # cast FP16→FP32 before normalization to avoid NaN under autograd
        cls_embed_adv = _safe_normalize(x[:, 0, :])
        patch_embeds_adv = _safe_normalize(x[:, 1:, :])

        # Compute loss with attention reallocation
        # CRITICAL: Pass A_orig (clean) for reallocation, A_adv for weighting
        loss, loss_dict = attack.compute_total_loss(
            patch_embeds_adv,
            cls_embed_adv,
            cls_embed_orig,
            patch_embeds_orig,
            A_adv,      # Adversarial attention for weighting
            A_orig,     # Clean attention for reallocation (CRITICAL!)
            target_indices
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite ADVEDM-A loss at iteration {iteration}: {loss_dict}"
            )

        # Backward
        loss.backward()
        if _sanitize_grad_inplace(delta.grad) and verbose:
            print(f"  [warn] Sanitized non-finite ADVEDM-A gradient at iter {iteration}.")
        optimizer.step()

        # Project to epsilon ball
        with torch.no_grad():
            if _sanitize_delta_inplace(delta, epsilon) and verbose:
                print(f"  [warn] Sanitized non-finite ADVEDM-A delta at iter {iteration}.")
            if constraint == "linf":
                # L∞ projection with image bounds
                delta_min = torch.max(-epsilon * torch.ones_like(image), -image)
                delta_max = torch.min(epsilon * torch.ones_like(image), 1 - image)
                delta.copy_(torch.clamp(delta, delta_min, delta_max))
            elif constraint == "l2":
                # L2 projection with image bounds
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
    """
    Paper-faithful ADVEDM-R implementation (Equations 3-8)

    Removes target semantics using text-guided masking.

    Key differences from ADVEDM-A:
    - Target: Text description (not reference image)
    - Mask: Top 20% text-patch similarity (not GPT annotation)
    - L_cls: Direct text similarity (not CLS fusion)
    - L_p: Masked image similarity (not reference patches)
    - L_fix: No attention reallocation (uses A_I directly)
    - Weights: (0.5, 2.0, 0.2) vs ADVEDM-A's (0.8, 2.0, 0.3)
    """

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
        """
        Initialize ADVEDM-R attack

        Args:
            vision_encoder: CLIP vision encoder
            text_encoder: CLIP text encoder (for target text)
            target_text: Text description of semantic to remove
            lambda_cls: w1 in Equation 8 (0.5 for ADVEDM-R)
            lambda_local: w2 in Equation 8 (2.0)
            lambda_fix: w3 in Equation 8 (0.2 for ADVEDM-R)
            k_ratio: Top-k ratio for removal mask (0.2 = 20%)
            mask_threshold: Optional fixed threshold xi in Eq.4. If provided,
                mask uses mask_i=0 if s_i>xi else 1. If None, xi is derived
                from top-k ratio per Appendix A.
            device: Device
        """
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.target_text = target_text
        self.lambda_cls = lambda_cls
        self.lambda_local = lambda_local
        self.lambda_fix = lambda_fix
        self.k_ratio = k_ratio
        self.mask_threshold = mask_threshold
        self.device = device

        # Encode target text once
        self.target_text_embed = self._encode_target_text()

    def _encode_target_text(self) -> torch.Tensor:
        """Encode target text to embedding [1, 768]"""
        text_features = self.text_encoder.encode_text([self.target_text])
        return text_features  # [1, 768]

    def compute_cls_loss(
        self,
        cls_embed_adv: torch.Tensor  # [B, D]
    ) -> torch.Tensor:
        """
        L_cls (Equation 5): Minimize similarity to target text

        Paper: L_cls = CS([cls]_I', t)

        CRITICAL: Positive cosine similarity (minimize to remove semantics)

        Args:
            cls_embed_adv: Adversarial CLS embedding [B, D]

        Returns:
            CLS loss (positive, to minimize for removal)
        """
        cls_norm = _safe_normalize(cls_embed_adv)
        text_norm = _safe_normalize(self.target_text_embed.to(cls_embed_adv.dtype))

        similarity = F.cosine_similarity(cls_norm, text_norm, dim=-1, eps=_COS_EPS)

        # Positive (minimize to remove)
        return similarity.mean()

    def compute_local_removal_loss(
        self,
        patch_embeds_adv: torch.Tensor,  # [B, 576, D]
        patch_embeds_masked: torch.Tensor,  # [B, 576, D]
        mask: torch.Tensor  # [B, 576]
    ) -> torch.Tensor:
        """
        L_p (Equation 6): Maximize similarity to masked patches in removal region

        Paper: L_p = -(1-mask) * CS([patch]_I', [patch]_M)

        Where (1-mask) selects removal patches (mask=0)

        Args:
            patch_embeds_adv: Adversarial patch embeddings [B, 576, D]
            patch_embeds_masked: Masked image patch embeddings [B, 576, D]
            mask: Binary mask [B, 576] (0=removal, 1=preserve)

        Returns:
            Local removal loss (negative, to maximize similarity)
        """
        removal_losses = []
        B = patch_embeds_adv.shape[0]

        for b in range(B):
            # Select removal patches (mask=0)
            removal_indices = (mask[b] == 0).nonzero(as_tuple=True)[0]
            if removal_indices.numel() == 0:
                raise ValueError("No removal patches found; adjust k_ratio/mask_threshold.")

            adv_removal = patch_embeds_adv[b, removal_indices]  # [k, D]
            masked_removal = patch_embeds_masked[b, removal_indices]  # [k, D]

            # Cosine similarity
            similarity = _cosine_mean(adv_removal, masked_removal)

            removal_losses.append(-similarity)  # Negative to maximize

        return torch.stack(removal_losses).mean()

    def compute_attention_fixation_loss(
        self,
        patch_embeds_adv: torch.Tensor,   # [B, 576, D]
        patch_embeds_orig: torch.Tensor,  # [B, 576, D]
        A_adv: torch.Tensor,              # [B, 576]
        A_orig: torch.Tensor,             # [B, 576]
        mask: torch.Tensor                # [B, 576]
    ) -> torch.Tensor:
        """
        L_fix (Equation 7): Preserve non-removal regions with attention weighting

        Paper: L_fix = -mask * CS(A_I' * [patch]_I', A_I * [patch]_I)

        CRITICAL: Uses A_I (clean attention), NOT A_reallocated (no reallocation in ADVEDM-R)

        Args:
            patch_embeds_adv: Adversarial patch embeddings [B, 576, D]
            patch_embeds_orig: Original patch embeddings [B, 576, D]
            A_adv: Adversarial CLS→patch attention [B, 576]
            A_orig: Original CLS→patch attention [B, 576]
            mask: Binary mask [B, 576] (0=removal, 1=preserve)

        Returns:
            Fixation loss (negative, to maximize preservation)
        """
        fixation_losses = []
        B = patch_embeds_adv.shape[0]

        for b in range(B):
            # Attention-weighted features
            weighted_adv = compute_attention_weighted_features_vector(
                A_adv[b:b+1], patch_embeds_adv[b:b+1]
            )[0]  # [576, D]

            weighted_orig = compute_attention_weighted_features_vector(
                A_orig[b:b+1], patch_embeds_orig[b:b+1]
            )[0]  # [576, D]

            # Select preserve patches (mask=1)
            preserve_indices = (mask[b] == 1).nonzero(as_tuple=True)[0]
            if preserve_indices.numel() == 0:
                raise ValueError("No preserve patches found; adjust k_ratio/mask_threshold.")

            preserve_adv = weighted_adv[preserve_indices]  # [N-k, D]
            preserve_orig = weighted_orig[preserve_indices]  # [N-k, D]

            # Cosine similarity
            similarity = _cosine_mean(preserve_adv, preserve_orig)

            fixation_losses.append(-similarity)  # Negative to maximize

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
        """
        Total loss (Equation 8): L = w1*L_cls + w2*L_p + w3*L_fix

        Args:
            cls_embed_adv: Adversarial CLS embedding [B, D]
            patch_embeds_adv: Adversarial patch embeddings [B, 576, D]
            patch_embeds_orig: Original patch embeddings [B, 576, D]
            patch_embeds_masked: Masked image patch embeddings [B, 576, D]
            A_adv: Adversarial CLS→patch attention [B, 576]
            A_orig: Original CLS→patch attention [B, 576]
            mask: Binary mask [B, 576] (0=removal, 1=preserve)

        Returns:
            - Total loss
            - Loss components dict: {total, cls, local, fix}
        """
        # Equation 5: CLS loss
        L_cls = self.compute_cls_loss(cls_embed_adv)

        # Equation 6: Local removal loss
        L_p = self.compute_local_removal_loss(
            patch_embeds_adv, patch_embeds_masked, mask
        )

        # Equation 7: Attention fixation loss
        L_fix = self.compute_attention_fixation_loss(
            patch_embeds_adv, patch_embeds_orig, A_adv, A_orig, mask
        )

        # Equation 8: Total loss
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
    """
    Extract patch and CLS embeddings (reusable pattern)

    Args:
        vision_encoder: CLIP vision encoder
        image_normalized: Preprocessed images [B, C, H, W]

    Returns:
        - patch_embeds: Normalized patch embeddings [B, 576, D]
        - cls_embed: Normalized CLS embedding [B, D]
    """
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
    x = x.float()  # cast FP16→FP32 before normalization to avoid NaN

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
    """
    ADVEDM-R optimization with text-guided masking (Paper-Faithful)

    Key differences from ADVEDM-A:
    - No target_indices parameter (mask computed from clean image)
    - No attention reallocation (uses A_orig directly in L_fix)
    - Masked image M created ONCE from clean image (cached with no_grad)

    Args:
        image: Original image [B, C, H, W] in [0, 1]
        vision_encoder: CLIP vision encoder (used when vision_backend is None)
        attack: ADVEDM-R attack instance
        vision_backend: Optional backend adapter with:
            - extract_patch_cls_embeddings(image, normalized=False)
            - extract_cls_to_patch_attention(image, normalized=False)
            - patch_size attribute
        epsilon: Perturbation budget (default: 8/255)
        num_iters: Number of iterations (default: 500)
        learning_rate: Adam learning rate (default: 0.005)
        constraint: "l2" (paper default, ε=8/255) or "linf" (ablation)
        verbose: Print progress

    Returns:
        - Adversarial image [B, C, H, W]
        - Perturbation [B, C, H, W]
        - Loss history
    """
    device = image.device
    B, C, H, W = image.shape

    # Initialize perturbation
    delta = torch.zeros_like(image, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=learning_rate)

    # CLIP normalization
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
        """Extract PROJECTED 768-dim features — used ONLY for masking and L_cls."""
        if vision_backend is not None:
            return vision_backend.extract_patch_cls_embeddings(img, normalized=False)
        return extract_patch_and_cls_embeddings(vision_encoder, img_normalized)

    def _extract_spatial_feats(
        img: torch.Tensor,
        img_normalized: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract PRE-PROJECTION 1024-dim features for L_p and L_fix.

        LLaVA's MLP projector receives these pre-projection patch tokens, so optimizing
        in this space directly targets the features LLaVA uses for language generation.
        """
        if vision_backend is not None:
            return vision_backend.extract_patch_cls_embeddings_noproj(img, normalized=False)
        return extract_patch_and_cls_embeddings(vision_encoder, img_normalized)

    # Resolve CLIP visual projection matrix for L_cls (projects 1024-dim pre-proj CLS →
    # 768-dim contrastive space for text similarity comparison).
    _clip_visual_proj: Optional[torch.Tensor] = None
    if vision_backend is not None:
        _clip_visual_proj = getattr(vision_backend, "_alignment_proj", None)
        if _clip_visual_proj is None:
            _clip_visual_proj = getattr(
                getattr(vision_backend, "vision_encoder", None), "proj", None
            )
    else:
        _clip_visual_proj = getattr(vision_encoder, "proj", None)

    # Extract CLEAN image features (once)
    with torch.no_grad():
        # CLS→patch attention from clean image
        A_orig = _extract_attention(image, image_normalized)

        # Masking: use PROJECTED features (768-dim) for text-patch similarity.
        # Text embeddings live in the 768-dim CLIP contrastive space, so patch
        # features must also be projected to find semantically relevant patches.
        patch_embeds_for_mask, _ = _extract_patch_cls(image, image_normalized)

        # Compute mask based on clean image
        S = compute_text_patch_similarity(patch_embeds_for_mask, attack.target_text_embed)
        if attack.mask_threshold is not None:
            # Eq.4 direct threshold mode: mask_i = 0 if s_i > xi else 1
            mask = construct_threshold_mask(S, threshold=attack.mask_threshold)
        else:
            # Appendix A mode: top-20% (implemented via Eq.4-derived threshold)
            mask = construct_top_k_mask(S, k_ratio=attack.k_ratio)

        # Spatial losses: use PRE-PROJ features (1024-dim) for L_p and L_fix.
        # These features live in the same space that LLaVA's MLP projector receives,
        # so the gradient directly acts on what LLaVA sees.
        patch_embeds_orig, _ = _extract_spatial_feats(image, image_normalized)

        # CRITICAL FIX: Create masked image M from CLEAN image (not adversarial)
        # M is FIXED throughout optimization (Equation 4, 6)
        # Derive patch_size from vision encoder (not hardcoded)
        if vision_backend is not None:
            patch_size = int(vision_backend.patch_size)
        else:
            patch_size = int(vision_encoder.conv1.kernel_size[0])
        image_size = H  # Use actual image dimensions

        masked_image = create_masked_image(
            image, mask,
            patch_size=patch_size,
            image_size=image_size
        )

        # Extract masked image embeddings ONCE (cached) — also pre-proj 1024-dim
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

        # Forward pass: adversarial image
        adv_image = image + delta
        adv_image_normalized = (adv_image - mean) / std

        # Extract adversarial attention
        # Use attention as fixed weights for this iteration (Eq.7), avoiding
        # unstable second-order gradients through attention computation.
        A_adv = _extract_attention(adv_image, adv_image_normalized).detach()

        # Extract adversarial features: PRE-PROJ 1024-dim for spatial losses (L_p, L_fix)
        patch_embeds_adv, cls_embed_adv_spatial = _extract_spatial_feats(
            adv_image, adv_image_normalized
        )

        # Project CLS to 768-dim contrastive space for L_cls text comparison.
        # L_cls = CS([cls]_I', t) where t is a CLIP text embedding (768-dim).
        # We apply the CLIP visual projection (1024→768) to the pre-proj CLS so that
        # both sides of the cosine similarity live in the same contrastive space.
        if _clip_visual_proj is not None:
            cls_proj = cls_embed_adv_spatial.float() @ _clip_visual_proj.to(
                device=cls_embed_adv_spatial.device, dtype=cls_embed_adv_spatial.dtype
            )
            cls_embed_adv = _safe_normalize(cls_proj.float())
        else:
            # Fallback: use 1024-dim CLS (will dim-mismatch at compute_cls_loss if
            # text_embed is 768-dim; caller should ensure proj is available).
            cls_embed_adv = cls_embed_adv_spatial

        # patch_embeds_masked is CACHED (not recomputed)

        # Compute total loss.
        # cls_embed_adv  → 768-dim projected  (for L_cls text similarity)
        # patch_embeds_* → 1024-dim pre-proj  (for L_p / L_fix spatial losses)
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

        # Backward
        loss.backward()
        if _sanitize_grad_inplace(delta.grad) and verbose:
            print(f"  [warn] Sanitized non-finite ADVEDM-R gradient at iter {iteration}.")
        optimizer.step()

        # Project to constraint
        with torch.no_grad():
            if _sanitize_delta_inplace(delta, epsilon) and verbose:
                print(f"  [warn] Sanitized non-finite ADVEDM-R delta at iter {iteration}.")
            if constraint == "linf":
                # L∞ projection
                delta_min = torch.max(-epsilon * torch.ones_like(image), -image)
                delta_max = torch.min(epsilon * torch.ones_like(image), 1 - image)
                delta.copy_(torch.clamp(delta, delta_min, delta_max))
            elif constraint == "l2":
                # L2 projection
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

    # Detach final results (no longer need gradients)
    adv_image = (image + delta).detach()
    delta_final = delta.detach()
    return adv_image, delta_final, loss_history
