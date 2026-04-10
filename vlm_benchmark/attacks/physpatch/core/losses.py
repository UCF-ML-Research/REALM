"""Loss functions for adversarial patch generation."""

import torch


def tv_loss(patch: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """
    Total Variation Loss

    Args:
        patch: Input tensor of shape (B, C, H, W) or (C, H, W)
        reduction: 'mean' or 'sum'

    Returns:
        Total variation loss value
    """
    if patch.dim() == 3:
        patch = patch.unsqueeze(0)

    dy = patch[:, :, 1:, :] - patch[:, :, :-1, :]
    dx = patch[:, :, :, 1:] - patch[:, :, :, :-1]

    if reduction == "mean":
        return (dy.abs().mean() + dx.abs().mean())
    elif reduction == "sum":
        return (dy.abs().sum() + dx.abs().sum())
    else:
        raise ValueError("reduction must be 'mean' or 'sum'")


# Printable color palette for NPS loss
palette = torch.tensor([
    [0,   0,   0  ],
    [255, 255, 255],
    [255, 0,   0  ],
    [0,   255, 0  ],
    [0,   0,   255],
    [255, 255, 0  ],
    [255, 0,   255],
    [0,   255, 255],
    [128, 128, 128],  # Grey
    [220, 130, 0  ],  # Orangish
    [160, 105, 55 ],  # Brownish
    [200, 175, 30 ],  # Goldish
    [220, 210, 50 ]   # Yellowish
], dtype=torch.float32)


def nps_loss(patch: torch.Tensor, palette: torch.Tensor = palette) -> torch.Tensor:
    """
    Non-Printability Score (NPS) Loss

    Measures how far patch colors are from a printable color palette.

    Args:
        patch: Input tensor of shape (B, C, H, W) or (C, H, W)
        palette: Color palette tensor of shape (K, 3) in RGB format

    Returns:
        NPS loss value (mean minimum distance to palette colors)
    """
    if patch.dim() == 3:
        patch = patch.unsqueeze(0)  # (1, 3, H, W)
    B, C, H, W = patch.shape

    # Reshape patch to (B, H*W, 3)
    patch_flat = patch.permute(0, 2, 3, 1).reshape(B, -1, 3)

    # Expand palette for batch
    palette = palette.to(patch.device).unsqueeze(0).expand(B, -1, -1)  # (B, K, 3)

    # Compute distances to all palette colors
    dist = torch.cdist(patch_flat, palette)

    # Return mean of minimum distances
    return dist.min(dim=-1).values.mean()
