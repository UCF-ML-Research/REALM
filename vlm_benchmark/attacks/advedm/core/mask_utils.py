"""
Mask Construction Utilities for ADVEDM-R

Text-guided mask construction and pixel-space masking (Equations 3-4)
"""

import torch
import torch.nn.functional as F

_COS_EPS = 1e-6


def compute_text_patch_similarity(
    patch_embeds: torch.Tensor,  # [B, 576, D]
    text_embed: torch.Tensor     # [1, D]
) -> torch.Tensor:
    """
    Compute cosine similarity between each patch and target text.

    Args:
        patch_embeds: Patch embeddings [B, 576, D]
        text_embed: Target text embedding [1, D]

    Returns:
        S: [B, 576] similarity scores
    """
    # Ensure dtype compatibility
    text_embed = text_embed.to(patch_embeds.dtype)

    # Normalize
    patch_norm = F.normalize(patch_embeds, dim=-1, eps=_COS_EPS)
    text_norm = F.normalize(text_embed, dim=-1, eps=_COS_EPS)

    # Cosine similarity
    S = torch.matmul(patch_norm, text_norm.T).squeeze(-1)  # [B, 576]
    return S


def construct_threshold_mask(
    similarity: torch.Tensor,  # [B, N]
    threshold: torch.Tensor | float
) -> torch.Tensor:
    """
    Eq.4 threshold mask:
        mask_i = 0 if s_i > xi else 1

    Args:
        similarity: Text-patch similarity scores [B, N]
        threshold: Scalar xi or per-sample tensor [B]

    Returns:
        mask: [B, N] binary (0=removal, 1=preserve)
    """
    B, N = similarity.shape
    mask = torch.ones(B, N, device=similarity.device)

    if not torch.is_tensor(threshold):
        threshold = torch.tensor(threshold, device=similarity.device, dtype=similarity.dtype)
    threshold = threshold.to(device=similarity.device, dtype=similarity.dtype)
    if threshold.ndim == 0:
        threshold = threshold.expand(B)
    if threshold.ndim != 1 or threshold.shape[0] != B:
        raise ValueError(f"threshold must be scalar or [B], got shape {tuple(threshold.shape)}")

    return torch.where(similarity > threshold.unsqueeze(1), torch.zeros_like(mask), mask)


def construct_top_k_mask(
    similarity: torch.Tensor,  # [B, N]
    k_ratio: float = 0.2
) -> torch.Tensor:
    """
    Build mask via Eq.4-style thresholding, with Appendix top-k ratio default.

    Eq.4 in the paper is threshold form:
        mask_i = 0 if s_i > xi else 1
    Appendix A sets removal to top 20% patches. We therefore derive per-sample
    threshold xi from top-k order statistics, then apply thresholding.
    A deterministic top-k fallback is used only when ties make threshold counts
    ambiguous.

    Args:
        similarity: Text-patch similarity scores [B, N]
        k_ratio: Ratio of patches to remove (default: 0.2 for 20%)

    Returns:
        mask: [B, N] binary (0=removal, 1=preserve)
    """
    if not (0.0 < k_ratio <= 1.0):
        raise ValueError(f"k_ratio must be in (0, 1], got {k_ratio}")

    B, N = similarity.shape
    # Use nearest integer count for the requested ratio (e.g., 20% of 576 -> 115).
    k = int(round(N * k_ratio))
    k = max(1, min(k, N))

    sorted_vals, _ = torch.sort(similarity, dim=1, descending=True)

    # Derive Eq.4 threshold xi between k-th and (k+1)-th scores.
    if k < N:
        xi = 0.5 * (sorted_vals[:, k - 1] + sorted_vals[:, k])  # [B]
    else:
        # Remove all patches: choose threshold below min score.
        xi = sorted_vals[:, -1] - 1e-6

    # Eq.4 thresholding: mask_i = 0 if s_i > xi else 1
    mask = construct_threshold_mask(similarity, xi)

    # Tie-handling fallback: enforce exact top-k count when threshold ties occur.
    for b in range(B):
        removal_count = int((mask[b] == 0).sum().item())
        if removal_count != k:
            top_k_indices = torch.topk(similarity[b], k, largest=True).indices
            mask[b].fill_(1.0)
            mask[b, top_k_indices] = 0.0

    return mask


def create_masked_image(
    image: torch.Tensor,       # [B, C, H, W] in [0,1]
    mask: torch.Tensor,        # [B, N] binary
    patch_size: int = 14,
    image_size: int = 336,
    grid_size: int | None = None,
) -> torch.Tensor:
    """
    Zero out pixels corresponding to removal patches (mask=0).

    Equation 4: M is created by masking image at patch locations

    Args:
        image: Input image [B, C, H, W] in [0, 1]
        mask: Binary mask [B, N] (0=removal, 1=preserve)
        patch_size: Patch size in pixels (default: 14)
        image_size: Image size in pixels (default: 336)
        grid_size: Patches per side. If None, computed as image_size //
                   patch_size. Required for models like SigLIP where
                   image_size (384) is not divisible by patch_size (14)
                   but the model produces a 27x27 grid.

    Returns:
        masked_image: [B, C, H, W] with removal regions zeroed
    """
    B, C, H, W = image.shape
    if image_size != H or image_size != W:
        raise ValueError(
            f"image_size argument ({image_size}) must match actual image size ({H}x{W})"
        )

    if grid_size is not None:
        grid_h = grid_w = grid_size
    else:
        if H % patch_size != 0 or W % patch_size != 0:
            raise ValueError(
                f"Image size ({H}x{W}) must be divisible by patch_size={patch_size}"
            )
        grid_h = H // patch_size
        grid_w = W // patch_size
    expected_patches = grid_h * grid_w
    if mask.shape[0] != B or mask.shape[1] != expected_patches:
        raise ValueError(
            f"Mask shape {tuple(mask.shape)} incompatible with image patches "
            f"{expected_patches} for image {H}x{W} and patch_size={patch_size}"
        )

    masked_image = image.clone()

    for b in range(B):
        removal_indices = (mask[b] == 0).nonzero(as_tuple=True)[0]

        for patch_idx in removal_indices:
            patch_idx_int = int(patch_idx.item())
            row = patch_idx_int // grid_w
            col = patch_idx_int % grid_w

            # Zero out 14x14 pixel region
            y_start = row * patch_size
            y_end = (row + 1) * patch_size
            x_start = col * patch_size
            x_end = (col + 1) * patch_size

            masked_image[b, :, y_start:y_end, x_start:x_end] = 0.0

    return masked_image
