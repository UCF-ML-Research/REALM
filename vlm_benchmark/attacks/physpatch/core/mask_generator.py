"""Dynamic patch mask generation with gradient-guided refinement."""

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import numpy as np
from scipy import ndimage
import kornia.morphology as morph


def remove_small_regions(mask_np, min_area=500):
    """
    Remove small disconnected regions from binary mask.

    Args:
        mask_np: Binary numpy array
        min_area: Minimum region size to keep

    Returns:
        Filtered binary mask
    """
    labeled_mask, num_features = ndimage.label(mask_np)
    sizes = ndimage.sum(mask_np, labeled_mask, range(1, num_features + 1))
    output_mask = np.zeros_like(mask_np)

    for i, size in enumerate(sizes):
        if size >= min_area:
            output_mask[labeled_mask == (i + 1)] = 1

    return output_mask


class DynamicPatchGenerator(nn.Module):
    """
    Dynamic patch mask generator with gradient-guided spatial refinement.

    Starts with a Gaussian-centered mask and refines it during optimization
    based on gradient feedback to focus on the most effective patch regions.
    """

    def __init__(self, image_size=(1600, 900), initial_threshold=0.7, learning_rate=0.15, device="cuda"):
        """
        Args:
            image_size: Target image size (width, height)
            initial_threshold: Initial threshold for binarizing soft mask
            learning_rate: Step size for mask refinement
            device: Device to place tensors on
        """
        super(DynamicPatchGenerator, self).__init__()

        self.width, self.height = image_size
        self.initial_threshold = initial_threshold
        self.learning_rate = learning_rate

        self.potential_field = None
        self.device = device

        # Create coordinate grid for distance computations
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(-1, 1, self.height),
            torch.linspace(-1, 1, self.width),
            indexing='ij'
        )
        self.register_buffer('coord_grid', torch.stack([x_grid, y_grid], dim=0).unsqueeze(0))

    def initialize_gaussian_field(self, coord, sigma=0.2):
        """
        Initialize potential field with Gaussian centered at coord.

        Args:
            coord: Normalized coordinates tensor of shape (B, 2)
            sigma: Standard deviation of Gaussian

        Returns:
            Gaussian field tensor
        """
        batch_size = coord.shape[0]

        coord = coord.view(batch_size, 2, 1, 1)

        # Compute distance from each coordinate
        distance = torch.sum((self.coord_grid - coord) ** 2, dim=1, keepdim=True)

        # Gaussian field
        gaussian_field = torch.exp(-distance / (2 * sigma ** 2))

        self.potential_field = gaussian_field.mean(dim=0, keepdim=True).to(self.device)

        return gaussian_field

    def get_mask(self, threshold=None):
        """
        Convert potential field to binary mask with morphological processing.

        Args:
            threshold: Binarization threshold (default: self.initial_threshold)

        Returns:
            Binary mask tensor
        """
        if threshold is None:
            threshold = self.initial_threshold

        # Soft mask through sigmoid
        soft_mask = torch.sigmoid(self.potential_field)

        # Binarize
        t_m = (soft_mask[0] > threshold).float().cpu().numpy().astype(np.uint8)

        # Fill holes
        t_m = ndimage.binary_fill_holes(t_m).astype(np.uint8)

        # Morphological closing
        t_m = torch.tensor(t_m, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        struct_elem = torch.ones((11, 11), device=self.device)
        t_m = morph.closing(t_m, struct_elem)

        # Gaussian smoothing
        t_m = t_m.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
        t_m = ndimage.gaussian_filter(t_m, sigma=3)

        # Re-binarize
        t_m = (t_m > threshold).astype(np.float32)

        # Remove small regions
        t_m = remove_small_regions(t_m, min_area=800)

        # Final hole filling and smoothing
        t_m = ndimage.binary_fill_holes(t_m).astype(np.uint8)
        t_m = ndimage.gaussian_filter(t_m, sigma=3)

        t_m = torch.tensor(t_m, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        return t_m

    def update_potential_field(self, mask_grads, iter):
        """
        Update potential field based on mask gradients.

        Regions with high gradient magnitudes indicate areas where the mask
        boundary affects the attack effectiveness, so we adjust the field there.

        Args:
            mask_grads: Gradients of loss w.r.t. mask
            iter: Current iteration number
        """
        # Take absolute gradients and mask to current mask region
        abs_grads = torch.abs(mask_grads) * self.mask_expanded

        # Smooth gradients
        smoothed_grads = TF.gaussian_blur(abs_grads, kernel_size=3, sigma=1)

        # Normalize gradients
        if smoothed_grads.max() > 0:
            norm_grads = (smoothed_grads - smoothed_grads.mean()) / (smoothed_grads.max() + 1e-8)
        else:
            norm_grads = smoothed_grads

        norm_grads = torch.maximum(norm_grads, torch.tensor(0.0).to(self.device))

        # Update potential field
        self.potential_field = self.potential_field + self.learning_rate * norm_grads
        self.potential_field = TF.gaussian_blur(self.potential_field, kernel_size=3, sigma=1)

    def forward(self, coord=(0.5, 0.5), threshold=None):
        """
        Generate mask for given coordinates.

        Args:
            coord: Normalized coordinates (x, y) in [-1, 1]
            threshold: Binarization threshold

        Returns:
            Binary mask tensor with gradient tracking
        """
        if self.potential_field is None:
            self.initialize_gaussian_field(coord)

        binary_mask = self.get_mask(threshold)
        self.mask_expanded = binary_mask.clone().requires_grad_(True)
        return self.mask_expanded
