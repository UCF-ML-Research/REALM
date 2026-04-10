"""Corruption attack: 6 natural image corruptions as a benign baseline.

Supported modes:
    brightness, fog, lowlight, motionblur, watersplash, saturate

No gradients, no model queries. Applies a deterministic corruption to the
clean image, useful for comparing adversarial ASR against natural robustness
degradation.
"""

import numpy as np
from PIL import Image

from imagecorruptions import corrupt

from ..base_attack import AttackResult, BaseAttack
from ...data.base_dataset import Sample
from .config import CorruptionConfig, VALID_MODES


class CorruptionAttack(BaseAttack):

    def is_gradient_based(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Corruption functions
    # ------------------------------------------------------------------

    @staticmethod
    def _brightness(img: np.ndarray, severity: int) -> np.ndarray:
        return corrupt(img, corruption_name='brightness', severity=severity)

    @staticmethod
    def _fog(img: np.ndarray, severity: int) -> np.ndarray:
        return corrupt(img, corruption_name='fog', severity=severity)

    @staticmethod
    def _lowlight(img: np.ndarray, severity: int) -> np.ndarray:
        gamma_values = [1.5, 2.0, 2.5, 3.0, 3.5]
        noise_std = [5, 10, 15, 20, 25]
        gamma = gamma_values[severity - 1]
        std = noise_std[severity - 1]

        out = (img / 255.0) ** gamma * 255.0
        noise = np.random.normal(0, std, img.shape)
        return np.clip(out + noise, 0, 255).astype(np.uint8)

    @staticmethod
    def _motionblur(img: np.ndarray, severity: int) -> np.ndarray:
        return corrupt(img, corruption_name='motion_blur', severity=severity)

    @staticmethod
    def _watersplash(img: np.ndarray, severity: int) -> np.ndarray:
        n_drops = [5, 10, 20, 35, 50]
        n = n_drops[severity - 1]

        h, w = img.shape[:2]
        out = img.copy().astype(np.float32)
        for _ in range(n):
            cx = np.random.randint(0, w)
            cy = np.random.randint(0, h)
            r = np.random.randint(15, 60)
            Y, X = np.ogrid[:h, :w]
            drop = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) <= r
            angle = np.random.uniform(0, 2 * np.pi)
            shift = int(r * 0.3)
            sx = int(shift * np.cos(angle))
            sy = int(shift * np.sin(angle))
            ys, xs = np.where(drop)
            src_y = np.clip(ys + sy, 0, h - 1)
            src_x = np.clip(xs + sx, 0, w - 1)
            out[ys, xs] = img[src_y, src_x].astype(np.float32) * 0.85 + 40
        return np.clip(out, 0, 255).astype(np.uint8)

    @staticmethod
    def _saturate(img: np.ndarray, severity: int) -> np.ndarray:
        factors = [2.0, 3.0, 4.0, 5.0, 6.0]
        factor = factors[severity - 1]

        hsv = np.array(Image.fromarray(img).convert('HSV')).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * factor, 0, 255)
        return np.array(Image.fromarray(hsv.astype(np.uint8), 'HSV').convert('RGB'))

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    _DISPATCH = {
        "brightness": _brightness.__func__,
        "fog": _fog.__func__,
        "lowlight": _lowlight.__func__,
        "motionblur": _motionblur.__func__,
        "watersplash": _watersplash.__func__,
        "saturate": _saturate.__func__,
    }

    def generate(self, model, sample: Sample, **kwargs) -> AttackResult:
        mode = self.config.mode
        if mode not in VALID_MODES:
            raise ValueError(
                f"Unknown corruption mode: {mode}. "
                f"Available: {VALID_MODES}"
            )

        clean = sample.images[0]
        img_array = np.array(clean.convert("RGB"))

        corrupt_fn = self._DISPATCH[mode]
        corrupted = corrupt_fn(img_array, self.config.severity)
        adv = Image.fromarray(corrupted)

        return AttackResult(
            success=False,
            adversarial_sample=adv,
            original_output="",
            adversarial_output="",
            perturbation_norm=0.0,
            queries=0,
            metadata={
                "attack": "corruption",
                "mode": mode,
                "severity": self.config.severity,
            },
        )
