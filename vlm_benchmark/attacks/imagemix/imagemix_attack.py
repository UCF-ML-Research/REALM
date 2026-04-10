"""ImageMix: simple alpha-blend / cutmix pixel perturbation.

No gradients, no CLIP, no GPU required. Serves as a trivial lower-bound
baseline for pixel-level attacks.

  mixup:   adv = alpha * target + (1-alpha) * clean
  cutmix:  paste a random rectangular region from target onto clean
"""

import random
from pathlib import Path

from PIL import Image

from ..base_attack import AttackConfig, AttackResult, BaseAttack
from ...data.base_dataset import Sample
from .config import ImageMixConfig


class ImageMixAttack(BaseAttack):

    def is_gradient_based(self) -> bool:
        return False

    def _load_target(self, sample_id: str) -> Image.Image:
        target_dir = Path(self.config.target_images_dir)
        for ext in (".jpg", ".png", ".jpeg"):
            p = target_dir / f"{sample_id}{ext}"
            if p.exists():
                return Image.open(p).convert("RGB")
        raise FileNotFoundError(
            f"No target image for '{sample_id}' in {target_dir}"
        )

    def _mixup(self, clean: Image.Image, target: Image.Image) -> Image.Image:
        target = target.resize(clean.size, Image.LANCZOS)
        return Image.blend(clean, target, self.config.alpha)

    def _cutmix(self, clean: Image.Image, target: Image.Image) -> Image.Image:
        target = target.resize(clean.size, Image.LANCZOS)
        w, h = clean.size
        # Random rectangle covering alpha fraction of the image
        area = self.config.alpha
        rw = int(w * area ** 0.5)
        rh = int(h * area ** 0.5)
        x = random.randint(0, max(0, w - rw))
        y = random.randint(0, max(0, h - rh))
        result = clean.copy()
        region = target.crop((x, y, x + rw, y + rh))
        result.paste(region, (x, y))
        return result

    def generate(self, model, sample: Sample, **kwargs) -> AttackResult:
        if not self.config.target_images_dir:
            raise ValueError(
                "ImageMix requires 'target_images_dir' to be set. "
                "Use the prepare pipeline or pass target_images_dir= at creation."
            )
        clean = sample.images[0]
        target = self._load_target(sample.id)

        if self.config.mix_type == "cutmix":
            adv = self._cutmix(clean, target)
        else:
            adv = self._mixup(clean, target)

        return AttackResult(
            success=False,
            adversarial_sample=adv,
            original_output="",
            adversarial_output="",
            perturbation_norm=0.0,
            queries=0,
            metadata={
                "mix_type": self.config.mix_type,
                "alpha": self.config.alpha,
                "attack": "imagemix",
            },
        )
