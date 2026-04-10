"""
Type validation utilities for VLM benchmark.

Provides functions to validate data types, shapes, and dimensions
to catch errors early with informative messages.
"""

import numpy as np
from PIL import Image
from typing import Union, List
import torch


class TypeValidationError(Exception):
    """Raised when type validation fails."""
    pass


def validate_trajectory(
    trajectory: Union[np.ndarray, List],
    min_length: int = 1,
    expected_dims: int = 2,
    name: str = "trajectory"
) -> np.ndarray:
    """Validate trajectory shape and convert to numpy array.

    Args:
        trajectory: Trajectory data (list or numpy array)
        min_length: Minimum number of trajectory points
        expected_dims: Expected dimensionality (default: 2 for [x, y])
        name: Name for error messages

    Returns:
        Validated trajectory as numpy array with shape [T, expected_dims]

    Raises:
        TypeValidationError: If validation fails
    """
    if trajectory is None:
        raise TypeValidationError(f"{name} is None")

    # Convert to numpy array if needed
    if isinstance(trajectory, list):
        try:
            trajectory = np.array(trajectory)
        except Exception as e:
            raise TypeValidationError(
                f"{name} could not be converted to numpy array: {e}"
            )

    if not isinstance(trajectory, np.ndarray):
        raise TypeValidationError(
            f"{name} must be numpy array or list, got {type(trajectory)}"
        )

    # Check shape
    if trajectory.ndim != 2:
        raise TypeValidationError(
            f"{name} must be 2D array with shape [T, {expected_dims}], "
            f"got {trajectory.ndim}D with shape {trajectory.shape}"
        )

    if trajectory.shape[1] != expected_dims:
        raise TypeValidationError(
            f"{name} must have {expected_dims} dimensions per point, "
            f"got {trajectory.shape[1]} (shape: {trajectory.shape})"
        )

    if len(trajectory) < min_length:
        raise TypeValidationError(
            f"{name} must have at least {min_length} points, "
            f"got {len(trajectory)}"
        )

    return trajectory


def validate_image(
    image: Union[Image.Image, np.ndarray, torch.Tensor],
    allow_batch: bool = False,
    name: str = "image"
) -> Union[Image.Image, np.ndarray, torch.Tensor]:
    """Validate image type and dimensions.

    Args:
        image: Image data (PIL Image, numpy array, or torch tensor)
        allow_batch: Whether to allow batch dimension [B, C, H, W]
        name: Name for error messages

    Returns:
        Validated image

    Raises:
        TypeValidationError: If validation fails
    """
    if image is None:
        raise TypeValidationError(f"{name} is None")

    # PIL Image
    if isinstance(image, Image.Image):
        if image.size[0] == 0 or image.size[1] == 0:
            raise TypeValidationError(
                f"{name} has zero dimensions: {image.size}"
            )
        return image

    # Numpy array
    if isinstance(image, np.ndarray):
        if image.ndim == 3:  # [H, W, C]
            if image.shape[0] == 0 or image.shape[1] == 0:
                raise TypeValidationError(
                    f"{name} has zero dimensions: {image.shape}"
                )
        elif image.ndim == 4 and allow_batch:  # [B, H, W, C] or [B, C, H, W]
            pass
        else:
            raise TypeValidationError(
                f"{name} numpy array must be 3D [H,W,C] "
                f"{'or 4D [B,H,W,C]' if allow_batch else ''}, "
                f"got {image.ndim}D with shape {image.shape}"
            )
        return image

    # Torch tensor
    if isinstance(image, torch.Tensor):
        if image.ndim == 3:  # [C, H, W]
            if image.shape[1] == 0 or image.shape[2] == 0:
                raise TypeValidationError(
                    f"{name} has zero dimensions: {image.shape}"
                )
        elif image.ndim == 4 and allow_batch:  # [B, C, H, W]
            pass
        else:
            raise TypeValidationError(
                f"{name} torch tensor must be 3D [C,H,W] "
                f"{'or 4D [B,C,H,W]' if allow_batch else ''}, "
                f"got {image.ndim}D with shape {image.shape}"
            )
        return image

    raise TypeValidationError(
        f"{name} must be PIL Image, numpy array, or torch tensor, "
        f"got {type(image)}"
    )
