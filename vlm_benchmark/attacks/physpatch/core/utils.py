"""Utility functions for patch application and image operations."""

import torch
import torchvision.transforms.functional as TF
import random
import math
from typing import Tuple


def apply_patch(image, patch):
    """
    Apply patch to center of image.

    Args:
        image: Image tensor of shape (B, C, H, W)
        patch: Patch tensor of shape (B, C, ph, pw)

    Returns:
        Image with patch applied at center
    """
    img = image.clone()
    _, _, H, W = img.shape
    ph, pw = patch.shape[2:]
    top = (H - ph) // 2
    left = (W - pw) // 2
    img[:, :, top:top+ph, left:left+pw] = patch
    return img


def apply_patch_with_center(image, patch, center):
    """
    Apply a patch to an image at a specified normalized center location.

    Args:
        image: Input image tensor of shape (N, C, H, W)
        patch: Patch tensor of shape (N, C, ph, pw)
        center: Normalized coordinates tensor of shape (N, 2),
                each coordinate in range [-1, 1], format [x, y]

    Returns:
        Image with the patch applied
    """
    img = image.clone()
    N, C, H, W = img.shape
    ph, pw = patch.shape[2:]

    assert img.shape[0] == patch.shape[0] == center.shape[0], "Batch sizes must match"
    assert center.shape[1] == 2, "Center should be a tensor of shape (N, 2)"

    for i in range(N):
        cx, cy = center[i]

        # Convert normalized coordinates to pixel coordinates
        cx_pixel = ((cx + 1) / 2) * W
        cy_pixel = ((cy + 1) / 2) * H

        left = int(torch.round(cx_pixel - pw / 2))
        top = int(torch.round(cy_pixel - ph / 2))

        # Clamp to image bounds
        if left < 0:
            left = 0
        if top < 0:
            top = 0
        if left + pw > W:
            left = W - pw
        if top + ph > H:
            top = H - ph

        img[i, :, top:top+ph, left:left+pw] = patch[i]

    return img


class RandomPointConstrainedCrop:
    """
    Random crop that ensures a specific point remains in the crop.

    This is useful for ensuring the adversarial patch location is preserved
    during data augmentation.
    """

    def __init__(
        self,
        size: Tuple[int, int],
        scale: Tuple[float, float] = (0.5, 0.9),
        ratio: Tuple[float, float] = (0.5, 2.0),
        norm_coord: Tuple[float, float] = (0.0, 0.0),
        attempts: int = 10,
    ) -> None:
        """
        Args:
            size: Output size (height, width)
            scale: Range of size of cropped area relative to image
            ratio: Range of aspect ratio of cropped area
            norm_coord: Normalized coordinates [-1, 1] of point to preserve
            attempts: Number of random crop attempts before fallback
        """
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.norm_coord = norm_coord
        self.attempts = attempts

    @staticmethod
    def _clamp(val: int, low: int, high: int) -> int:
        return max(low, min(val, high))

    def _point_to_pixel(self, H: int, W: int) -> Tuple[int, int]:
        """Convert normalized coordinates to pixel coordinates."""
        x = (self.norm_coord[0] + 1) * 0.5 * W
        y = (self.norm_coord[1] + 1) * 0.5 * H
        # Convert to scalar if tensor
        if isinstance(x, torch.Tensor):
            x = x.item()
        if isinstance(y, torch.Tensor):
            y = y.item()
        return int(round(x)), int(round(y))

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: Tensor of shape (C, H, W) or (B, C, H, W)

        Returns:
            Cropped and resized tensor
        """
        if img.dim() == 4:
            batch_mode = True
            _, _, H, W = img.shape
        elif img.dim() == 3:
            batch_mode = False
            _, H, W = img.shape
        else:
            raise ValueError("Unsupported tensor shape; expect 3D or 4D.")

        x_p, y_p = self._point_to_pixel(H, W)
        total_area = H * W

        for _ in range(self.attempts):
            q = random.uniform(*self.scale)
            target_area = q * total_area

            log_ar = random.uniform(math.log(self.ratio[0]), math.log(self.ratio[1]))
            a = math.exp(log_ar)

            h = int(round(math.sqrt(target_area / a)))
            w = int(round(a * h))

            if h > H or w > W or h == 0 or w == 0:
                continue

            # Ensure point is in crop
            x_left_min = max(0, x_p - w)
            x_left_max = min(x_p, W - w)
            y_top_min = max(0, y_p - h)
            y_top_max = min(y_p, H - h)

            if x_left_max < x_left_min or y_top_max < y_top_min:
                continue

            x_left = random.randint(x_left_min, x_left_max)
            y_top = random.randint(y_top_min, y_top_max)

            return TF.resized_crop(img, y_top, x_left, h, w, self.size)

        # Fallback: center crop and resize
        side = min(H, W)
        return TF.resize(TF.center_crop(img, side), self.size)
