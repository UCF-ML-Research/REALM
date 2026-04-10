"""V-Attack: text-guided adversarial attack on CLIP Value features."""

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from ..base_attack import AttackConfig, AttackResult, BaseAttack
from ...data.base_dataset import Sample


@dataclass
class VAttackConfig(AttackConfig):
    epsilon: float = 16.0
    max_iterations: int = 300
    alpha: float = 0.75
    backbone: List[str] = field(default_factory=lambda: ["B16", "B32", "Laion"])
    input_res: int = 224
    crop_scale: Tuple[float, float] = (0.75, 1.0)
    use_source_crop: bool = True
    vattack: bool = True
    enhance: bool = True
    both: bool = False
    source_text: str = ""
    target_text: str = ""


class VAttack(BaseAttack):
    """V-Attack wrapper using BaseAttack interface.

    Lazy-loads the CLIP ensemble on first generate() call (same pattern as MAttack/FOA).
    source_text and target_text are set per-record by the generate pipeline.
    """

    def __init__(self, config: VAttackConfig):
        super().__init__(config)
        self.config: VAttackConfig = config
        self._ensemble_extractor = None
        self._ensemble_loss = None
        self._models = None

    def _initialize_models(self) -> None:
        if self._ensemble_extractor is not None:
            return

        from .core.surrogates.FeatureExtractors import (
            ClipB16FeatureExtractor,
            ClipB32FeatureExtractor,
            ClipLaionFeatureExtractor,
            EnsembleFeatureExtractor,
            EnsembleFeatureLoss,
        )

        class_map = {
            "B16": ClipB16FeatureExtractor,
            "B32": ClipB32FeatureExtractor,
            "Laion": ClipLaionFeatureExtractor,
        }

        models = []
        for name in self.config.backbone:
            cls = class_map[name]
            model = cls().eval().to(self.config.device).requires_grad_(False)
            models.append(model)

        self._models = models
        self._ensemble_extractor = EnsembleFeatureExtractor(models)
        self._ensemble_loss = EnsembleFeatureLoss(models)

    def _prepare_image(self, pil_img: Image.Image) -> torch.Tensor:
        """PIL → tensor [0,255] with Resize+CenterCrop to input_res."""
        transform = transforms.Compose([
            transforms.Resize(
                self.config.input_res,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(self.config.input_res),
            transforms.Lambda(lambda img: img.convert("RGB")),
        ])
        img = transform(pil_img)
        arr = np.array(img, dtype=np.uint8)
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous().float()
        return tensor.unsqueeze(0)  # [1, C, H, W]

    @staticmethod
    def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
        """Tensor [0,1] → PIL Image."""
        arr = tensor.squeeze(0).detach().cpu().clamp(0, 1).mul(255).byte()
        arr = arr.permute(1, 2, 0).numpy()
        return Image.fromarray(arr)

    def _dict_to_list(self, features_dict):
        return [features_dict[i] for i in range(len(features_dict))]

    def _resolve_texts(self, sample: Sample) -> tuple[str, str]:
        """Get source/target text from config or per-sample metadata."""
        target = self.config.target_text
        source = self.config.source_text
        # Per-sample override from metadata (labels.json)
        if not target and sample.metadata.get("attack_target_text"):
            target = sample.metadata["attack_target_text"]
        if not source and sample.metadata.get("attack_source_text"):
            source = sample.metadata["attack_source_text"]
        if not target:
            raise ValueError(
                f"No target_text: set --target_text or provide "
                f"metadata['attack_target_text'] on sample '{sample.id}'."
            )
        return source or "", target

    def generate(self, model, sample: Sample, **kwargs) -> AttackResult:
        self._initialize_models()

        cfg = self.config
        device = cfg.device

        # Prepare image
        pil_img = sample.images[0]
        image_org = self._prepare_image(pil_img).to(device)

        # Source crop
        if cfg.use_source_crop:
            source_crop = transforms.RandomResizedCrop(cfg.input_res, scale=cfg.crop_scale)
        else:
            source_crop = torch.nn.Identity()

        # Resolve per-sample text
        source_text, target_text = self._resolve_texts(sample)

        # Encode texts
        source_text_enc = self._dict_to_list(
            self._ensemble_extractor.tforward([source_text])
        )
        target_text_enc = self._dict_to_list(
            self._ensemble_extractor.tforward([target_text])
        )

        # Run PGD attack
        from .core.attacks import pgd_attack
        adv_tensor = pgd_attack(
            image_org=image_org,
            ensemble_extractor=self._ensemble_extractor,
            ensemble_loss=self._ensemble_loss,
            source_crop=source_crop,
            source_text=source_text_enc,
            target_text=target_text_enc,
            steps=cfg.max_iterations,
            epsilon=cfg.epsilon,
            alpha=cfg.alpha,
            vattack=cfg.vattack,
            enhance=cfg.enhance,
            both=cfg.both,
            vision_attack=False,
            target_text_flag=True,
            device=device,
        )

        adv_pil = self._tensor_to_pil(adv_tensor)

        return AttackResult(
            success=True,
            adversarial_sample=adv_pil,
            original_output="",
            adversarial_output="",
            perturbation_norm=cfg.epsilon / 255.0,
            metadata={
                "source_text": source_text,
                "target_text": target_text,
                "steps": cfg.max_iterations,
                "epsilon": cfg.epsilon,
            },
        )

    def is_gradient_based(self) -> bool:
        return True
