"""
Image corruption for robustness testing.

Implements various corruption types following DriveBench methodology
to test VLM sensitivity to visual degradation.
"""

from enum import Enum
from typing import Optional, Tuple
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import random


class CorruptionType(Enum):
    """Types of image corruption."""
    NONE = "none"                    # Clean image
    GAUSSIAN_NOISE = "gaussian"      # Additive Gaussian noise
    MOTION_BLUR = "motion_blur"      # Motion blur
    DEFOCUS_BLUR = "defocus_blur"    # Defocus blur
    BRIGHTNESS = "brightness"        # Brightness change
    CONTRAST = "contrast"            # Contrast change
    OCCLUSION = "occlusion"          # Random occlusion
    RAIN = "rain"                    # Simulated rain
    FOG = "fog"                      # Simulated fog
    JPEG_COMPRESSION = "jpeg"        # JPEG artifacts
    TEXT_ONLY = "text_only"          # Black/placeholder image


class ImageCorruptor:
    """Apply corruptions to images for robustness testing."""

    def __init__(self, severity: int = 3):
        """Initialize image corruptor.

        Args:
            severity: Corruption severity level (1-5)
        """
        self.severity = max(1, min(5, severity))

    def corrupt(
        self,
        image: Image.Image,
        corruption_type: CorruptionType
    ) -> Image.Image:
        """Apply corruption to image.

        Args:
            image: Input PIL Image
            corruption_type: Type of corruption to apply

        Returns:
            Corrupted image
        """
        if corruption_type == CorruptionType.NONE:
            return image

        if corruption_type == CorruptionType.TEXT_ONLY:
            return self._text_only(image)

        if corruption_type == CorruptionType.GAUSSIAN_NOISE:
            return self._gaussian_noise(image)

        if corruption_type == CorruptionType.MOTION_BLUR:
            return self._motion_blur(image)

        if corruption_type == CorruptionType.DEFOCUS_BLUR:
            return self._defocus_blur(image)

        if corruption_type == CorruptionType.BRIGHTNESS:
            return self._brightness(image)

        if corruption_type == CorruptionType.CONTRAST:
            return self._contrast(image)

        if corruption_type == CorruptionType.OCCLUSION:
            return self._occlusion(image)

        if corruption_type == CorruptionType.RAIN:
            return self._rain(image)

        if corruption_type == CorruptionType.FOG:
            return self._fog(image)

        if corruption_type == CorruptionType.JPEG_COMPRESSION:
            return self._jpeg_compression(image)

        return image

    def _text_only(self, image: Image.Image) -> Image.Image:
        """Return black placeholder image (text-only test)."""
        return Image.new('RGB', image.size, color='black')

    def _gaussian_noise(self, image: Image.Image) -> Image.Image:
        """Add Gaussian noise."""
        img_array = np.array(image).astype(float)
        noise_std = [10, 20, 35, 50, 70][self.severity - 1]
        noise = np.random.normal(0, noise_std, img_array.shape)
        noisy = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)

    def _motion_blur(self, image: Image.Image) -> Image.Image:
        """Apply motion blur."""
        kernel_size = [3, 5, 7, 9, 11][self.severity - 1]
        return image.filter(ImageFilter.BoxBlur(kernel_size // 2))

    def _defocus_blur(self, image: Image.Image) -> Image.Image:
        """Apply defocus/Gaussian blur."""
        radius = [1, 2, 3, 4, 5][self.severity - 1]
        return image.filter(ImageFilter.GaussianBlur(radius))

    def _brightness(self, image: Image.Image) -> Image.Image:
        """Change brightness (darker)."""
        factor = [0.8, 0.6, 0.4, 0.3, 0.2][self.severity - 1]
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)

    def _contrast(self, image: Image.Image) -> Image.Image:
        """Reduce contrast."""
        factor = [0.8, 0.6, 0.5, 0.4, 0.3][self.severity - 1]
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)

    def _occlusion(self, image: Image.Image) -> Image.Image:
        """Add random rectangular occlusions."""
        img_array = np.array(image)
        h, w = img_array.shape[:2]

        num_occlusions = self.severity
        for _ in range(num_occlusions):
            # Random box size (5-20% of image)
            box_h = int(h * random.uniform(0.05, 0.15))
            box_w = int(w * random.uniform(0.05, 0.15))
            # Random position
            y = random.randint(0, h - box_h)
            x = random.randint(0, w - box_w)
            # Black occlusion
            img_array[y:y+box_h, x:x+box_w] = 0

        return Image.fromarray(img_array)

    def _rain(self, image: Image.Image) -> Image.Image:
        """Simulate rain effect."""
        img_array = np.array(image).astype(float)
        h, w = img_array.shape[:2]

        # Create rain streaks
        num_drops = [100, 200, 400, 600, 800][self.severity - 1]
        for _ in range(num_drops):
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 5)
            length = random.randint(5, 15)
            # White-ish rain streak
            for dy in range(min(length, h - y)):
                img_array[y + dy, x] = img_array[y + dy, x] * 0.7 + 200 * 0.3

        return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

    def _fog(self, image: Image.Image) -> Image.Image:
        """Simulate fog effect."""
        img_array = np.array(image).astype(float)

        # Fog density
        density = [0.1, 0.2, 0.35, 0.5, 0.7][self.severity - 1]
        fog_color = np.array([200, 200, 200])

        # Blend with fog
        fogged = img_array * (1 - density) + fog_color * density
        return Image.fromarray(np.clip(fogged, 0, 255).astype(np.uint8))

    def _jpeg_compression(self, image: Image.Image) -> Image.Image:
        """Apply JPEG compression artifacts."""
        import io
        quality = [50, 30, 20, 10, 5][self.severity - 1]
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert('RGB')

    @classmethod
    def get_all_corruptions(cls) -> list:
        """Get list of all corruption types for testing."""
        return [c for c in CorruptionType if c != CorruptionType.NONE]

    @classmethod
    def get_driving_relevant_corruptions(cls) -> list:
        """Get corruptions relevant to autonomous driving scenarios."""
        return [
            CorruptionType.RAIN,
            CorruptionType.FOG,
            CorruptionType.BRIGHTNESS,  # Night/tunnel
            CorruptionType.MOTION_BLUR,  # Fast movement
            CorruptionType.OCCLUSION,    # Partial view blockage
        ]
