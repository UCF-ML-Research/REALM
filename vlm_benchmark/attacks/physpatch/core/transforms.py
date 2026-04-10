"""Image transformation utilities for data augmentation during attacks."""

import numpy as np
import torch
import torch.nn.functional as F
import torch_dct as dct
import scipy.stats as st


device = "cuda" if torch.cuda.is_available() else "cpu"


class My_T:
    """
    Ensemble of diverse image transformations for robust adversarial attacks.

    Applies random transformations including geometric, color, frequency-domain,
    and noise-based augmentations.
    """

    def __init__(self, num_copies=5, **kwargs):
        """
        Args:
            num_copies: Number of transformed copies to generate per input
        """
        self.num_copies = num_copies
        self.kernel = self.gkern()

        # Transformation operations
        self.op = [
            self.resize,
            self.vertical_shift,
            self.horizontal_shift,
            self.vertical_flip,
            self.horizontal_flip,
            self.rotate180,
            self.scale,
            self.add_noise,
            self.dct,
            self.drop_out,
            self.adjust_brightness,
            self.adjust_contrast
        ]

    def vertical_shift(self, x):
        """Randomly shift image vertically."""
        _, _, w, _ = x.shape
        step = np.random.randint(low=0, high=w // 10, dtype=np.int32)
        return x.roll(step, dims=2)

    def horizontal_shift(self, x):
        """Randomly shift image horizontally."""
        _, _, _, h = x.shape
        step = np.random.randint(low=0, high=h // 10, dtype=np.int32)
        return x.roll(step, dims=3)

    def vertical_flip(self, x):
        """Flip image vertically."""
        return x.flip(dims=(2,))

    def horizontal_flip(self, x):
        """Flip image horizontally."""
        return x.flip(dims=(3,))

    def rotate180(self, x):
        """Rotate image 180 degrees."""
        return x.rot90(k=2, dims=(2, 3))

    def scale(self, x):
        """Randomly scale pixel values."""
        return torch.rand(1)[0] * x

    def resize(self, x):
        """Resize with bilinear interpolation."""
        _, _, w, h = x.shape
        scale_factor = 0.8
        new_h = int(h * scale_factor) + 1
        new_w = int(w * scale_factor) + 1
        x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
        x = F.interpolate(x, size=(w, h), mode='bilinear', align_corners=False).clamp(0, 255)
        return x

    def dct(self, x):
        """Apply DCT-based low-pass filtering."""
        dctx = dct.dct_2d(x)
        _, _, w, h = dctx.shape
        low_ratio = 0.4
        low_w = int(w * low_ratio)
        low_h = int(h * low_ratio)
        # Zero out high frequency components
        dctx[:, :, -low_w:, :] = 0
        dctx[:, :, :, -low_h:] = 0
        idctx = dct.idct_2d(dctx)
        return idctx

    def add_noise(self, x):
        """Add uniform random noise."""
        return torch.clip(x + torch.zeros_like(x).uniform_(-16, 16), 0, 255)

    def gkern(self, kernel_size=3, nsig=3):
        """Generate Gaussian kernel."""
        x = np.linspace(-nsig, nsig, kernel_size)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        return torch.from_numpy(stack_kernel.astype(np.float32)).to(device)

    def adjust_brightness(self, x, factor=None):
        """
        Adjust brightness by scaling all pixels.

        Args:
            x: Input tensor
            factor: Brightness factor (>1: brighter, <1: darker)
        """
        if factor is None:
            factor = np.random.uniform(0.9, 1.1)
        return (x * factor).clamp(0, 255)

    def adjust_contrast(self, x, factor=None):
        """
        Adjust contrast by changing difference from mean.

        Args:
            x: Input tensor
            factor: Contrast factor (>1: more contrast, <1: less contrast)
        """
        if factor is None:
            factor = np.random.uniform(0.8, 1.2)
        mean = x.mean(dim=(2, 3), keepdim=True)
        return ((x - mean) * factor + mean).clamp(0, 255)

    def drop_out(self, x):
        """Apply 2D dropout."""
        return F.dropout2d(x, p=0.1, training=True)

    def blocktransform(self, x):
        """Apply a single random transformation."""
        i = torch.randint(0, len(self.op), [1]).item()
        trans = self.op[i]
        return trans(x)

    def transform(self, x, **kwargs):
        """
        Apply multiple random transformations.

        Args:
            x: Input tensor

        Returns:
            Concatenated transformed copies
        """
        return torch.cat([self.blocktransform(x) for _ in range(self.num_copies)])
