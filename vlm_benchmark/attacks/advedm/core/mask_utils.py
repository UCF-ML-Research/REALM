"""Text-guided mask construction and pixel-space masking for ADVEDM-R."""

import torch
import torch.nn.functional as F

_COS_EPS = 1e-6


def compute_text_patch_similarity(
    patch_embeds: torch.Tensor,  # [B, 576, D]
    text_embed: torch.Tensor     # [1, D]
) -> torch.Tensor:
    """Compute cosine similarity between each patch and the target text."""
    text_embed = text_embed.to(patch_embeds.dtype)

    patch_norm = F.normalize(patch_embeds, dim=-1, eps=_COS_EPS)
    text_norm = F.normalize(text_embed, dim=-1, eps=_COS_EPS)

    S = torch.matmul(patch_norm, text_norm.T).squeeze(-1)
    return S


def construct_threshold_mask(
    similarity: torch.Tensor,  # [B, N]
    threshold: torch.Tensor | float
) -> torch.Tensor:
    """Eq.4 threshold mask: mask_i = 0 if s_i > xi else 1."""
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
    """Build a removal mask for the top-k most text-similar patches via Eq.4-style thresholding."""
    if not (0.0 < k_ratio <= 1.0):
        raise ValueError(f"k_ratio must be in (0, 1], got {k_ratio}")

    B, N = similarity.shape
    k = int(round(N * k_ratio))
    k = max(1, min(k, N))

    sorted_vals, _ = torch.sort(similarity, dim=1, descending=True)

    if k < N:
        xi = 0.5 * (sorted_vals[:, k - 1] + sorted_vals[:, k])
    else:
        xi = sorted_vals[:, -1] - 1e-6

    mask = construct_threshold_mask(similarity, xi)

    # Enforce exact top-k count when threshold ties occur.
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
    """Zero out pixels corresponding to removal patches (mask=0)."""
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

            y_start = row * patch_size
            y_end = (row + 1) * patch_size
            x_start = col * patch_size
            x_end = (col + 1) * patch_size

            masked_image[b, :, y_start:y_end, x_start:x_end] = 0.0

    return masked_image
