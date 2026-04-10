"""ADVEDM attack wrappers for VLM benchmark framework.

Integrates ADVEDM-A/R semantic addition/removal attacks and their black-box
transfer variants (SSA-CWA + multi-CLIP ensemble) into the attack registry.
Core logic lives in core/advedm_attack.py (unchanged from the original ADVEDM
codebase).
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

from ..base_attack import AttackConfig, AttackResult, BaseAttack
from ...data import Sample


# ---------------------------------------------------------------------------
# Utility functions (inlined from ADVEDM/main.py)
# ---------------------------------------------------------------------------

def _select_contiguous_patch_block(
    bbox_pixels: Tuple[int, int, int, int],
    patch_size: int,
    image_size: int,
) -> torch.Tensor:
    """
    Convert pixel bounding box to contiguous patch indices.

    Args:
        bbox_pixels: (x, y, w, h) bounding box in pixel coordinates
        patch_size: Size of each patch (e.g. 14 for ViT-L/14)
        image_size: Image size in pixels (e.g. 336)

    Returns:
        Tensor of patch indices [k]
    """
    x, y, w, h = bbox_pixels
    grid_size = image_size // patch_size

    # Clamp to image bounds
    x = max(0, min(x, image_size - 1))
    y = max(0, min(y, image_size - 1))
    w = max(1, min(w, image_size - x))
    h = max(1, min(h, image_size - y))

    # Convert pixel coords to patch coords
    patch_x_start = x // patch_size
    patch_y_start = y // patch_size
    patch_x_end = min((x + w + patch_size - 1) // patch_size, grid_size)
    patch_y_end = min((y + h + patch_size - 1) // patch_size, grid_size)

    indices = []
    for i in range(patch_y_start, patch_y_end):
        for j in range(patch_x_start, patch_x_end):
            indices.append(i * grid_size + j)

    return torch.tensor(indices, dtype=torch.long)


# ===========================================================================
# Black-box transfer variants (SSA-CWA + multi-CLIP ensemble)
# ===========================================================================

@dataclass
class ADVEDMConfig(AttackConfig):
    """Configuration for ADVEDM-A attack (SSA-CWA + 4 CLIP ensemble).

    Uses SSA-CWA optimizer with 4 CLIP surrogates for black-box transferability.
    ε=16/255, 30 iterations (paper Appendix D defaults).
    """
    # SSA-CWA optimizer params
    epsilon: float = 16.0 / 255.0
    num_iters: int = 30
    inner_step_size: float = 250.0
    ksi: float = 16.0 / 255.0 / 5.0
    mu: float = 1.0
    ssa_N: int = 20
    ssa_sigma: float = 16.0 / 255.0
    ssa_rho: float = 0.5

    # ADVEDM-A loss weights
    # lambda_attention=0: L_fix preserves non-injection regions, which fights
    # ASR in a benchmark where we want full-image semantic change.
    lambda_cls: float = 0.8
    lambda_preserve: float = 2.0
    lambda_attention: float = 0.0
    cls_fusion_alpha: float = 0.5
    attention_beta: float = 0.4

    # Ensemble
    ensemble_preset: str = "paper"  # "paper" for the 4-model ensemble

    # Target specification (single image or per-sample directory)
    reference_image_path: Optional[str] = None
    target_images_dir: Optional[str] = None  # dir with {stem}.jpg/.png per source
    annotations_file: Optional[str] = None   # optional pre-computed bboxes JSON
    region_size: int = 100
    image_size: int = 224   # working resolution for adversarial image (surrogates resize as needed)


@dataclass
class ADVEDMRConfig(AttackConfig):
    """Configuration for ADVEDM-R attack (SSA-CWA + 4 CLIP ensemble).

    Uses SSA-CWA optimizer with 4 CLIP surrogates for black-box transferability.
    """
    # SSA-CWA optimizer params
    epsilon: float = 16.0 / 255.0
    num_iters: int = 30
    inner_step_size: float = 250.0
    ksi: float = 16.0 / 255.0 / 5.0
    mu: float = 1.0
    ssa_N: int = 20
    ssa_sigma: float = 16.0 / 255.0
    ssa_rho: float = 0.5

    # ADVEDM-R loss weights
    # lambda_fix=0: L_fix preserves non-removal regions, which fights
    # ASR in a benchmark where we want full-image semantic change.
    lambda_cls: float = 0.5
    lambda_local: float = 2.0
    lambda_fix: float = 0.0
    k_ratio: float = 0.2

    # Ensemble
    ensemble_preset: str = "paper"

    # Required semantic target
    target_text: Optional[str] = None


class ADVEDMAttack(BaseAttack):
    """
    ADVEDM-A attack (default).

    Uses AdvEDM losses (L_cls, L_p, L_fix) + SSA-CWA optimizer + 4 CLIP
    surrogates for black-box transferability.
    """

    def __init__(self, config: ADVEDMConfig):
        super().__init__(config)
        self.config: ADVEDMConfig = config

        self._ensemble = None
        self._attack_objs = None   # one per surrogate
        self._current_ref: Optional[str] = None  # tracks which reference is loaded
        self._annotations: Optional[Dict[str, Any]] = None

    def _resolve_reference(self, sample_id: str) -> str:
        """Resolve reference image path for a sample.

        If target_images_dir is set, looks up {stem}.jpg/.png there.
        Otherwise falls back to reference_image_path (single image).
        """
        if self.config.target_images_dir:
            d = Path(self.config.target_images_dir)
            for ext in (".jpg", ".png", ".jpeg"):
                p = d / f"{sample_id}{ext}"
                if p.exists():
                    return str(p)
        if self.config.reference_image_path:
            return self.config.reference_image_path
        raise ValueError(
            f"No reference image found for sample '{sample_id}'. "
            "Set reference_image_path or target_images_dir."
        )

    def _initialize(self, reference_path: str) -> None:
        """Lazy-load ensemble and create per-surrogate attack objects.

        Re-creates attack objects if the reference image changes.
        """
        if self._ensemble is not None and self._current_ref == reference_path:
            return

        from .core.ensemble_encoder import EnsembleEncoder, PAPER_ENSEMBLE

        if self._ensemble is None:
            print("Loading CLIP ensemble for ADVEDM-A...")
            self._ensemble = EnsembleEncoder(
                specs=PAPER_ENSEMBLE, device=self.config.device,
            )
            self._ensemble.load_all()

        # Create (or recreate) one ADVEDMSemanticAdditionPaperExact per surrogate
        from .core.advedm_attack import ADVEDMSemanticAdditionPaperExact

        self._attack_objs = []
        for sm in self._ensemble.surrogates:
            # SigLIP has no CLS token → attention is uniform (1/N) →
            # attention-weighted patch losses (L_p, L_fix) carry no spatial
            # signal.  Only keep L_cls for these surrogates.
            lam_p = self.config.lambda_preserve if sm.has_cls else 0.0
            lam_a = self.config.lambda_attention if sm.has_cls else 0.0
            atk = ADVEDMSemanticAdditionPaperExact(
                vision_encoder=sm.vision_encoder,
                reference_image_path=reference_path,
                device=self.config.device,
                lambda_cls=self.config.lambda_cls,
                lambda_preserve=lam_p,
                lambda_attention=lam_a,
                alpha=self.config.cls_fusion_alpha,
                beta=self.config.attention_beta,
                norm_mean=sm.spec.norm_mean,
                norm_std=sm.spec.norm_std,
                image_size=sm.image_size,
            )
            self._attack_objs.append(atk)
        self._current_ref = reference_path

    def _load_annotations(self) -> Optional[Dict[str, Any]]:
        if self._annotations is not None:
            return self._annotations
        if self.config.annotations_file is None:
            return None
        with open(self.config.annotations_file, encoding="utf-8") as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            raise ValueError(
                f"Annotation file must contain a JSON object map, got {type(loaded).__name__}"
            )
        self._annotations = loaded
        return self._annotations

    def _get_bbox(
        self,
        sample: Sample,
        image_tensor: torch.Tensor,
    ) -> Tuple[int, int, int, int]:
        """Return injection bbox for ADVEDM-A.

        Priority:
          1. Pre-computed annotations_file entry (if provided).
          2. CLIP attention-based auto-detection: lowest-attention k×k block
             = background region the model looks at least.
        """
        image_file = str(sample.metadata.get("image_file", ""))
        annotations = self._load_annotations()
        if annotations is not None and image_file:
            entry = annotations.get(image_file)
            if entry is None:
                image_name = Path(image_file).name
                for key, value in annotations.items():
                    if Path(key).name == image_name:
                        entry = value
                        break
            if entry is not None:
                bbox_value = entry.get("bbox") if isinstance(entry, dict) else entry
                if bbox_value is not None:
                    return tuple(int(v) for v in bbox_value)

        bbox = self._ensemble.clip_bbox_from_attention(
            image_tensor,
            region_pixels=self.config.region_size,
            source_size=self.config.image_size,
        )
        print(f"  [ADVEDM-A] Auto bbox (CLIP attention): {bbox}")
        return bbox

    def _preprocess_image(self, pil_image: Image.Image) -> torch.Tensor:
        """Resize to config.image_size and convert to [1, 3, H, W] in [0,1]."""
        sz = self.config.image_size
        transform = transforms.Compose([
            transforms.Resize((sz, sz)),
            transforms.ToTensor(),
        ])
        return transform(pil_image).unsqueeze(0).to(self.config.device)

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        t = tensor.squeeze(0).clamp(0.0, 1.0).cpu()
        arr = (t.permute(1, 2, 0).numpy() * 255).astype("uint8")
        return Image.fromarray(arr)

    def generate(
        self,
        model,
        sample: Sample,
        **kwargs,
    ) -> AttackResult:
        """Generate ADVEDM-A blackbox adversarial image."""
        ref_path = self._resolve_reference(sample.id)
        self._initialize(ref_path)

        original_pil = sample.images[0] if sample.images else None
        if original_pil is None:
            raise ValueError(f"Sample '{sample.id}' has no images.")

        image_tensor = self._preprocess_image(original_pil)
        bbox = self._get_bbox(sample, image_tensor)
        source_image_size = self.config.image_size

        ensemble = self._ensemble
        attack_objs = self._attack_objs

        # Pre-compute clean image features per surrogate (no grad)
        clean_cache = []
        for sm, atk in zip(ensemble.surrogates, attack_objs):
            with torch.no_grad():
                img_norm = ensemble.resize_and_normalize(image_tensor, sm)
                patches_orig, cls_orig = ensemble.extract_features(sm, img_norm)
                A_orig = ensemble.extract_attention(sm, img_norm)
                target_idx = ensemble.bbox_to_target_indices(
                    bbox, sm, source_image_size
                ).to(self.config.device)
            clean_cache.append({
                "sm": sm, "atk": atk,
                "patches_orig": patches_orig,
                "cls_orig": cls_orig,
                "A_orig": A_orig,
                "target_indices": target_idx,
            })

        # Build per-surrogate loss closures for SSA-CWA
        def make_loss_fn(cache_entry, surrogate_idx):
            sm = cache_entry["sm"]
            atk = cache_entry["atk"]
            patches_orig = cache_entry["patches_orig"]
            cls_orig = cache_entry["cls_orig"]
            A_orig = cache_entry["A_orig"]
            target_idx = cache_entry["target_indices"]
            _n_calls = [0]

            def loss_fn(x_aug: torch.Tensor) -> torch.Tensor:
                """Compute ADVEDM-A total loss for one surrogate."""
                img_norm = ensemble.resize_and_normalize(x_aug, sm)
                patches_adv, cls_adv = ensemble.extract_features(sm, img_norm)
                A_adv = ensemble.extract_attention(sm, img_norm).detach()

                loss, components = atk.compute_total_loss(
                    patch_embeds_adv=patches_adv,
                    cls_embed_adv=cls_adv,
                    cls_embed_orig=cls_orig,
                    patch_embeds_orig=patches_orig,
                    A_adv=A_adv,
                    A_orig=A_orig,
                    target_indices=target_idx,
                )
                _n_calls[0] += 1
                if surrogate_idx == 0 and _n_calls[0] % self.config.ssa_N == 1:
                    outer_iter = (_n_calls[0] - 1) // self.config.ssa_N
                    print(f"  [ADVEDM-A sur0 iter={outer_iter:2d}] loss={loss.item():.4f}  "
                          f"cls={components['cls']:.4f}  ref={components['reference_sim']:.4f}  "
                          f"fix={components['attention_fix']:.4f}", flush=True)
                return -loss  # Negate: SSA ascends, loss is for minimization

            return loss_fn

        loss_fns = [make_loss_fn(c, i) for i, c in enumerate(clean_cache)]

        # Run SSA-CWA
        from .core.ssa_cwa import ssa_cwa_attack

        adv_tensor = ssa_cwa_attack(
            x_orig=image_tensor,
            loss_fns=loss_fns,
            num_iters=self.config.num_iters,
            epsilon=self.config.epsilon,
            inner_step_size=self.config.inner_step_size,
            ksi=self.config.ksi,
            mu=self.config.mu,
            N=self.config.ssa_N,
            sigma=self.config.ssa_sigma,
            rho=self.config.ssa_rho,
            verbose=True,
        )

        adversarial_pil = self._tensor_to_pil(adv_tensor)
        delta = adv_tensor - image_tensor
        perturbation_linf = float(delta.abs().max().item())

        return AttackResult(
            success=True,
            adversarial_sample=adversarial_pil,
            original_output="",
            adversarial_output="",
            perturbation_norm=perturbation_linf,
            metadata={
                "reference_image": ref_path,
                "constraint": "linf",
                "num_iterations": self.config.num_iters,
                "epsilon": self.config.epsilon,
                "num_surrogates": len(self._ensemble.surrogates),
                "attack_type": "advedm_a",
            },
        )

    def is_gradient_based(self) -> bool:
        return True


class ADVEDMRAttack(BaseAttack):
    """
    ADVEDM-R attack (default).

    Uses AdvEDM-R losses (L_cls, L_p, L_fix) + SSA-CWA optimizer + 4 CLIP
    surrogates for black-box transferability.
    """

    def __init__(self, config: ADVEDMRConfig):
        super().__init__(config)
        self.config: ADVEDMRConfig = config

        self._ensemble = None
        self._attack_objs = None   # one per surrogate

    def _initialize(self) -> None:
        """Lazy-load ensemble (once) and create per-surrogate attack objects.

        Ensemble loading is expensive (4 CLIP models); attack_obj creation is
        cheap (text embedding only).  Resetting self._attack_objs = None and
        updating self.config.target_text allows per-sample target refresh
        without reloading the ensemble.
        """
        if self._ensemble is None:
            from .core.ensemble_encoder import EnsembleEncoder, PAPER_ENSEMBLE

            print("Loading CLIP ensemble for ADVEDM-R...")
            self._ensemble = EnsembleEncoder(
                specs=PAPER_ENSEMBLE, device=self.config.device,
            )
            self._ensemble.load_all()

        if self._attack_objs is None:
            target_text = (self.config.target_text or "").strip()
            if not target_text:
                raise ValueError(
                    "ADVEDMRConfig.target_text is required for ADVEDM-R."
                )

            from .core.advedm_attack import ADVEDMSemanticRemovalPaperExact

            self._attack_objs = []
            for sm in self._ensemble.surrogates:
                text_enc = _SurrogateTextEncoder(self._ensemble, sm)
                atk = ADVEDMSemanticRemovalPaperExact(
                    vision_encoder=sm.vision_encoder,
                    text_encoder=text_enc,
                    target_text=target_text,
                    lambda_cls=self.config.lambda_cls,
                    lambda_local=self.config.lambda_local,
                    lambda_fix=self.config.lambda_fix,
                    k_ratio=self.config.k_ratio,
                    device=self.config.device,
                )
                self._attack_objs.append(atk)
            print(f"  Created {len(self._attack_objs)} per-surrogate ADVEDM-R attack objects.")

    def _preprocess_image(self, pil_image: Image.Image) -> torch.Tensor:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        return transform(pil_image).unsqueeze(0).to(self.config.device)

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        t = tensor.squeeze(0).clamp(0.0, 1.0).cpu()
        arr = (t.permute(1, 2, 0).numpy() * 255).astype("uint8")
        return Image.fromarray(arr)

    def generate(
        self,
        model,
        sample: Sample,
        **kwargs,
    ) -> AttackResult:
        """Generate ADVEDM-R blackbox adversarial image."""
        # ADVEDM-R removes the SOURCE object's semantics — always use attack_source_text.
        # attack_target_text is the object to inject (used by ADVEDM-A), not remove.
        per_sample_text = (sample.metadata or {}).get("attack_source_text")
        if per_sample_text and per_sample_text != self.config.target_text:
            self.config.target_text = per_sample_text
            self._attack_objs = None  # Force re-creation with new text

        self._initialize()

        original_pil = sample.images[0] if sample.images else None
        if original_pil is None:
            raise ValueError(f"Sample '{sample.id}' has no images.")

        image_tensor = self._preprocess_image(original_pil)

        ensemble = self._ensemble
        attack_objs = self._attack_objs

        from .core.mask_utils import (
            compute_text_patch_similarity,
            construct_top_k_mask,
            create_masked_image,
        )

        # Pre-compute clean image features + mask per surrogate (no grad)
        clean_cache = []
        for sm, atk in zip(ensemble.surrogates, attack_objs):
            with torch.no_grad():
                img_norm = ensemble.resize_and_normalize(image_tensor, sm)
                patches_orig, cls_orig = ensemble.extract_features(sm, img_norm)
                A_orig = ensemble.extract_attention(sm, img_norm)

                # Compute text-patch similarity mask for this surrogate
                # Project patches to output embedding space (same space as text features)
                patches_proj = ensemble.project_patches(sm, patches_orig)
                S = compute_text_patch_similarity(patches_proj, atk.target_text_embed)
                mask = construct_top_k_mask(S, k_ratio=self.config.k_ratio)

                # Create masked image and extract its features
                # Need to resize image_tensor to surrogate's size first
                sz = sm.image_size
                ps = sm.patch_size
                if image_tensor.shape[2] != sz or image_tensor.shape[3] != sz:
                    img_resized = F.interpolate(
                        image_tensor, size=(sz, sz),
                        mode="bilinear", align_corners=False,
                    )
                else:
                    img_resized = image_tensor

                masked_img = create_masked_image(
                    img_resized, mask,
                    patch_size=ps, image_size=sz,
                    grid_size=sm.grid_size,
                )
                masked_norm = ensemble.resize_and_normalize(masked_img, sm)
                patches_masked, _ = ensemble.extract_features(sm, masked_norm)

            clean_cache.append({
                "sm": sm, "atk": atk,
                "patches_orig": patches_orig,
                "patches_masked": patches_masked,
                "A_orig": A_orig,
                "mask": mask,
            })

        # Build per-surrogate loss closures
        def make_loss_fn(cache_entry, surrogate_idx):
            sm = cache_entry["sm"]
            atk = cache_entry["atk"]
            patches_orig = cache_entry["patches_orig"]
            patches_masked = cache_entry["patches_masked"]
            A_orig = cache_entry["A_orig"]
            mask = cache_entry["mask"]
            _n_calls = [0]

            def loss_fn(x_aug: torch.Tensor) -> torch.Tensor:
                """Compute ADVEDM-R total loss for one surrogate."""
                img_norm = ensemble.resize_and_normalize(x_aug, sm)
                patches_adv, cls_adv = ensemble.extract_features(sm, img_norm)
                A_adv = ensemble.extract_attention(sm, img_norm).detach()

                # Project CLS to output embedding space (same space as text features)
                cls_adv_proj = ensemble.project_patches(sm, cls_adv.unsqueeze(1)).squeeze(1)

                loss, components = atk.compute_total_loss(
                    cls_embed_adv=cls_adv_proj,
                    patch_embeds_adv=patches_adv,
                    patch_embeds_orig=patches_orig,
                    patch_embeds_masked=patches_masked,
                    A_adv=A_adv,
                    A_orig=A_orig,
                    mask=mask,
                )
                _n_calls[0] += 1
                if surrogate_idx == 0 and _n_calls[0] % self.config.ssa_N == 1:
                    outer_iter = (_n_calls[0] - 1) // self.config.ssa_N
                    print(f"  [ADVEDM-R sur0 iter={outer_iter:2d}] loss={loss.item():.4f}  "
                          f"cls={components['cls']:.4f}  local={components['local']:.4f}  "
                          f"fix={components['fix']:.4f}", flush=True)
                return -loss  # Negate: SSA ascends, loss is for minimization

            return loss_fn

        loss_fns = [make_loss_fn(c, i) for i, c in enumerate(clean_cache)]

        # Run SSA-CWA
        from .core.ssa_cwa import ssa_cwa_attack

        adv_tensor = ssa_cwa_attack(
            x_orig=image_tensor,
            loss_fns=loss_fns,
            num_iters=self.config.num_iters,
            epsilon=self.config.epsilon,
            inner_step_size=self.config.inner_step_size,
            ksi=self.config.ksi,
            mu=self.config.mu,
            N=self.config.ssa_N,
            sigma=self.config.ssa_sigma,
            rho=self.config.ssa_rho,
            verbose=True,
        )

        adversarial_pil = self._tensor_to_pil(adv_tensor)
        delta = adv_tensor - image_tensor
        perturbation_linf = float(delta.abs().max().item())

        return AttackResult(
            success=True,
            adversarial_sample=adversarial_pil,
            original_output="",
            adversarial_output="",
            perturbation_norm=perturbation_linf,
            metadata={
                "target_text": self.config.target_text,
                "constraint": "linf",
                "num_iterations": self.config.num_iters,
                "epsilon": self.config.epsilon,
                "num_surrogates": len(self._ensemble.surrogates),
                "attack_type": "advedm_r",
            },
        )

    def is_gradient_based(self) -> bool:
        return True


class _SurrogateTextEncoder:
    """Adapter: wraps EnsembleEncoder.encode_text for a specific surrogate.

    ADVEDMSemanticRemovalPaperExact calls self.text_encoder.encode_text([text])
    in __init__. This adapter forwards that call through the ensemble.
    """

    def __init__(self, ensemble, surrogate):
        self._ensemble = ensemble
        self._surrogate = surrogate

    def encode_text(self, texts: list) -> torch.Tensor:
        return self._ensemble.encode_text(self._surrogate, texts)
