"""
M-Attack implementation for VLM benchmark.

Implements adversarial perturbations using simple cosine similarity loss
with CLIP ensemble optimization (no OT, faster than FOA-Attack).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch.nn as nn

from ..base_attack import BaseAttack, AttackConfig, AttackResult
from ...data import Sample


@dataclass
class MAttackConfig(AttackConfig):
    """Configuration for M-Attack."""

    # Inherited from AttackConfig:
    # - epsilon: float = 8.0            # [0, 255] range (legacy)
    # - max_iterations: int = 300       # Optimization steps (legacy)
    # - alpha: float = 1.0              # Step size (legacy)
    # - device: str = "cuda"
    epsilon: float = 16.0
    max_iterations: int = 300
    alpha: float = 1.0

    # Attack algorithm selection
    attack_method: str = "pgd"          # "fgsm", "mifgsm", "pgd"

    # CLIP ensemble configuration
    backbone: List[str] = field(
        default_factory=lambda: ["B16", "B32", "Laion"]
    )

    # M-Attack: NO clustering (simpler than FOA)
    # No cluster_number, no adaptive cluster

    # Crop parameters for feature matching
    use_source_crop: bool = True        # Crop adversarial image
    use_target_crop: bool = True        # Crop target image
    crop_scale: Tuple[float, float] = (0.5, 0.9)

    # Source and target image directories (flat dirs, stem = sample.id)
    source_images_dir: Optional[str] = None
    target_strategy: str = "stop_sign"  # Unused (legacy pairs by dataset order)
    target_images_dir: Optional[str] = None

    # Image resolution (224×224 matching legacy)
    input_res: int = 224


class MAttack(BaseAttack):
    """
    M-Attack wrapper integrating with VLM benchmark.

    Implements adversarial perturbations using simple cosine similarity loss
    (no Optimal Transport) with CLIP ensemble optimization.

    Key differences from FOA-Attack:
    - Simple cosine similarity loss (55 lines vs 776 lines)
    - No k-means clustering
    - No adaptive cluster escalation
    - 5-10x faster execution
    """

    def __init__(self, config: MAttackConfig):
        super().__init__(config)
        self.config: MAttackConfig = config

        # Lazy initialization (models are heavy)
        self._models_initialized = False
        self._ensemble_extractor = None
        self._ensemble_loss = None

    def _initialize_models(self):
        """Lazy load CLIP surrogate models (NO cluster parameter - simpler than FOA)."""
        if self._models_initialized:
            return  # Already initialized

        print(f"Loading CLIP ensemble models...")

        # Import M-Attack modules
        from .surrogates import (
            ClipB16FeatureExtractor,
            ClipB32FeatureExtractor,
            ClipL336FeatureExtractor,
            ClipLaionFeatureExtractor,
            EnsembleFeatureExtractor,
            EnsembleFeatureLoss,
        )

        # Backbone mapping
        BACKBONE_MAP = {
            "B16": ClipB16FeatureExtractor,
            "B32": ClipB32FeatureExtractor,
            "L336": ClipL336FeatureExtractor,
            "Laion": ClipLaionFeatureExtractor,
        }

        # Load backbones
        models = []
        for backbone in self.config.backbone:
            if backbone not in BACKBONE_MAP:
                raise ValueError(f"Unknown backbone: {backbone}")

            model_class = BACKBONE_MAP[backbone]
            model = model_class().eval().to(self.config.device)
            model.requires_grad_(False)
            models.append(model)
            print(f"  ✓ Loaded {backbone}")

        # Create ensemble (NO cluster parameter)
        self._ensemble_extractor = EnsembleFeatureExtractor(models)
        self._ensemble_loss = EnsembleFeatureLoss(models)

        self._models_initialized = True
        print(f"✓ Ensemble ready\n")

    def _prepare_image(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL image to tensor in [0, 255] range (NOT normalized)."""
        # Resize to resolution
        image = transforms.Resize(
            self.config.input_res,
            interpolation=transforms.InterpolationMode.BICUBIC
        )(image)
        image = transforms.CenterCrop(self.config.input_res)(image)
        image = image.convert("RGB")

        # Convert to tensor WITHOUT normalization (keep [0, 255] range)
        mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
        img_array = np.array(image, mode_to_nptype.get(image.mode, np.uint8), copy=True)
        img_tensor = torch.from_numpy(img_array)
        img_tensor = img_tensor.view(image.size[1], image.size[0], len(image.getbands()))
        img_tensor = img_tensor.permute(2, 0, 1).contiguous().float()

        return img_tensor.unsqueeze(0).to(self.config.device)

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert tensor from [0, 1] range back to PIL image."""
        # Tensor comes back in [0, 1] range from attack
        # Handle both [B, C, H, W] and [C, H, W] formats
        if len(tensor.shape) == 4:
            tensor = tensor.squeeze(0)  # Remove batch dimension
        elif len(tensor.shape) != 3:
            raise ValueError(f"Expected tensor with 3 or 4 dimensions, got {len(tensor.shape)}")

        tensor = torch.clamp(tensor, 0, 1)

        # Convert to [0, 255] uint8
        tensor = (tensor * 255).cpu().byte()

        # Convert to numpy and then PIL
        img_array = tensor.permute(1, 2, 0).numpy()
        return Image.fromarray(img_array, mode='RGB')

    def _load_image_from_dir(self, images_dir: str, sample_id: str) -> torch.Tensor:
        """Load image from a flat directory by sample id stem (e.g. '00', '02')."""
        img_dir = Path(images_dir)
        for ext in (".jpg", ".jpeg", ".png"):
            p = img_dir / f"{sample_id}{ext}"
            if p.exists():
                return self._prepare_image(Image.open(p).convert("RGB"))
        raise FileNotFoundError(f"No image found for id '{sample_id}' in {img_dir}")

    def _create_crops(self):
        """Create crop functions for source and target (no coordinate constraints)."""
        # M-Attack uses simple RandomResizedCrop
        if self.config.use_source_crop:
            source_crop = transforms.RandomResizedCrop(
                size=self.config.input_res,
                scale=self.config.crop_scale
            )
        else:
            source_crop = nn.Identity()

        if self.config.use_target_crop:
            target_crop = transforms.RandomResizedCrop(
                size=self.config.input_res,
                scale=self.config.crop_scale
            )
        else:
            target_crop = nn.Identity()

        return source_crop, target_crop

    def generate(self, model, sample: Sample, **kwargs) -> AttackResult:
        """
        Generate M-Attack adversarial example.

        Args:
            model: VLM model (not used for attack, only for evaluation)
            sample: Driving sample to attack
            **kwargs: Additional arguments (not used)

        Returns:
            AttackResult with adversarial image
        """
        # 1. Initialize models
        self._initialize_models()

        # 2. Load source and target images by sample id (stem match, same as FOA)
        if not self.config.source_images_dir:
            raise ValueError("source_images_dir must be set in config")
        if not self.config.target_images_dir:
            raise ValueError("target_images_dir must be set in config")
        sample_idx = kwargs.get("sample_idx", 0)
        clean_image  = self._load_image_from_dir(self.config.source_images_dir, str(sample.id))
        target_image = self._load_image_from_dir(self.config.target_images_dir, str(sample.id))

        # 3. Create crops
        source_crop, target_crop = self._create_crops()

        # 4. Run attack (NO adaptive clustering - simpler than FOA!)
        from .core.attacks import fgsm_attack, mifgsm_attack, pgd_attack

        attack_fn_map = {
            "fgsm": fgsm_attack,
            "mifgsm": mifgsm_attack,
            "pgd": pgd_attack,
        }
        attack_fn = attack_fn_map.get(self.config.attack_method, pgd_attack)

        # Run single attack (no retry logic like FOA)
        adv_image = attack_fn(
            image_tensor=clean_image,
            tgt_tensor=target_image,
            ensemble_extractor=self._ensemble_extractor,
            ensemble_loss=self._ensemble_loss,
            source_crop=source_crop,
            target_crop=target_crop,
            img_index=sample_idx if sample_idx is not None else 0,
            num_iters=self.config.max_iterations,
            epsilon=self.config.epsilon,
            alpha=self.config.alpha,
            device=self.config.device,
            use_source_crop=self.config.use_source_crop,
            use_target_crop=self.config.use_target_crop,
        )

        # 5. Convert to PIL
        adv_pil = self._tensor_to_pil(adv_image)

        # 6. Return result
        return AttackResult(
            success=False,  # Evaluated later in pipeline
            adversarial_sample=adv_pil,
            original_output="",  # Filled by evaluation pipeline
            adversarial_output="",  # Filled by evaluation pipeline
            perturbation_norm=self.config.epsilon / 255.0,
            queries=1,
            metadata={
                "attack_method": self.config.attack_method,
                "backbone": self.config.backbone,
                "epsilon": self.config.epsilon,
                "max_iterations": self.config.max_iterations,
                "alpha": self.config.alpha,
            }
        )

    def is_gradient_based(self) -> bool:
        """M-Attack uses gradients from surrogate CLIP models."""
        return True
