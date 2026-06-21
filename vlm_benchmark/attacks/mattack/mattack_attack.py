"""M-Attack: adversarial perturbations via cosine similarity over a CLIP ensemble."""

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

    epsilon: float = 16.0
    max_iterations: int = 300
    alpha: float = 1.0

    attack_method: str = "pgd"          # "fgsm", "mifgsm", "pgd"

    backbone: List[str] = field(
        default_factory=lambda: ["B16", "B32", "Laion"]
    )

    use_source_crop: bool = True
    use_target_crop: bool = True
    crop_scale: Tuple[float, float] = (0.5, 0.9)

    source_images_dir: Optional[str] = None
    target_strategy: str = "stop_sign"  # Unused (legacy pairs by dataset order)
    target_images_dir: Optional[str] = None

    input_res: int = 224


class MAttack(BaseAttack):
    """M-Attack: adversarial perturbations via cosine similarity over a CLIP ensemble."""

    def __init__(self, config: MAttackConfig):
        super().__init__(config)
        self.config: MAttackConfig = config

        self._models_initialized = False
        self._ensemble_extractor = None
        self._ensemble_loss = None

    def _initialize_models(self):
        """Lazy-load the CLIP surrogate models."""
        if self._models_initialized:
            return

        print(f"Loading CLIP ensemble models...")

        from .surrogates import (
            ClipB16FeatureExtractor,
            ClipB32FeatureExtractor,
            ClipL336FeatureExtractor,
            ClipLaionFeatureExtractor,
            EnsembleFeatureExtractor,
            EnsembleFeatureLoss,
        )

        BACKBONE_MAP = {
            "B16": ClipB16FeatureExtractor,
            "B32": ClipB32FeatureExtractor,
            "L336": ClipL336FeatureExtractor,
            "Laion": ClipLaionFeatureExtractor,
        }

        models = []
        for backbone in self.config.backbone:
            if backbone not in BACKBONE_MAP:
                raise ValueError(f"Unknown backbone: {backbone}")

            model_class = BACKBONE_MAP[backbone]
            model = model_class().eval().to(self.config.device)
            model.requires_grad_(False)
            models.append(model)
            print(f"  ✓ Loaded {backbone}")

        self._ensemble_extractor = EnsembleFeatureExtractor(models)
        self._ensemble_loss = EnsembleFeatureLoss(models)

        self._models_initialized = True
        print(f"✓ Ensemble ready\n")

    def _prepare_image(self, image: Image.Image) -> torch.Tensor:
        """Convert a PIL image to an unnormalized [0, 255] tensor."""
        image = transforms.Resize(
            self.config.input_res,
            interpolation=transforms.InterpolationMode.BICUBIC
        )(image)
        image = transforms.CenterCrop(self.config.input_res)(image)
        image = image.convert("RGB")

        mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
        img_array = np.array(image, mode_to_nptype.get(image.mode, np.uint8), copy=True)
        img_tensor = torch.from_numpy(img_array)
        img_tensor = img_tensor.view(image.size[1], image.size[0], len(image.getbands()))
        img_tensor = img_tensor.permute(2, 0, 1).contiguous().float()

        return img_tensor.unsqueeze(0).to(self.config.device)

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert a [0, 1] tensor back to a PIL image."""
        if len(tensor.shape) == 4:
            tensor = tensor.squeeze(0)
        elif len(tensor.shape) != 3:
            raise ValueError(f"Expected tensor with 3 or 4 dimensions, got {len(tensor.shape)}")

        tensor = torch.clamp(tensor, 0, 1)

        tensor = (tensor * 255).cpu().byte()

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
        """Create RandomResizedCrop transforms for source and target."""
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
        """Generate an M-Attack adversarial example for the given sample."""
        self._initialize_models()

        if not self.config.source_images_dir:
            raise ValueError("source_images_dir must be set in config")
        if not self.config.target_images_dir:
            raise ValueError("target_images_dir must be set in config")
        sample_idx = kwargs.get("sample_idx", 0)
        clean_image  = self._load_image_from_dir(self.config.source_images_dir, str(sample.id))
        target_image = self._load_image_from_dir(self.config.target_images_dir, str(sample.id))

        source_crop, target_crop = self._create_crops()

        from .core.attacks import fgsm_attack, mifgsm_attack, pgd_attack

        attack_fn_map = {
            "fgsm": fgsm_attack,
            "mifgsm": mifgsm_attack,
            "pgd": pgd_attack,
        }
        attack_fn = attack_fn_map.get(self.config.attack_method, pgd_attack)

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

        adv_pil = self._tensor_to_pil(adv_image)

        return AttackResult(
            success=False,
            adversarial_sample=adv_pil,
            original_output="",
            adversarial_output="",
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
