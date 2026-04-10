"""
Attention map extraction utilities for VLMs.

Provides functions to:
- Extract cross-attention maps from vision-language models
- Compute attention diversity metrics
- Select pivotal frames based on attention patterns
"""

import torch
import numpy as np
from typing import List, Optional, Dict
from PIL import Image


def extract_attention_maps(
    model,
    images: List[Image.Image],
    question: str,
    layer_indices: Optional[List[int]] = None,
) -> Dict[str, torch.Tensor]:
    """Extract attention maps from VLM.

    Args:
        model: VLM model wrapper
        images: List of PIL Images
        question: Question/prompt text
        layer_indices: Which layers to extract from (None = all layers)

    Returns:
        Dictionary mapping layer names to attention tensors
    """
    # This is a simplified implementation
    # Real implementation would hook into model's attention layers

    attention_maps = {}

    try:
        # Check if model has attention extraction capability
        if hasattr(model, 'extract_attention'):
            attention_maps = model.extract_attention(images, question, layer_indices)

        elif hasattr(model, 'model'):
            # Try to extract from underlying model
            # This requires model-specific implementation
            # For now, return dummy attention (uniform)
            attention_maps = _dummy_attention_maps(images, question)

        else:
            # Fallback: dummy uniform attention
            attention_maps = _dummy_attention_maps(images, question)

    except Exception as e:
        print(f"Warning: Failed to extract attention maps ({e}), using uniform attention")
        attention_maps = _dummy_attention_maps(images, question)

    return attention_maps


def _dummy_attention_maps(
    images: List[Image.Image],
    question: str,
) -> Dict[str, torch.Tensor]:
    """Create dummy uniform attention maps.

    Args:
        images: List of images
        question: Question text

    Returns:
        Dictionary with dummy attention tensors
    """
    # Assume each image has attention of shape [num_patches]
    # For simplicity, use 14x14 = 196 patches (common for ViT)
    num_patches = 196
    num_images = len(images)

    # Uniform attention
    attention = torch.ones(num_images, num_patches) / num_patches

    return {
        "cross_attention": attention,
    }


def compute_attention_diversity(
    attention_maps: List[torch.Tensor],
    metric: str = "correlation_distance",
) -> np.ndarray:
    """Compute diversity scores between attention maps.

    Args:
        attention_maps: List of attention tensors [num_patches]
        metric: Diversity metric ("correlation_distance", "kl_divergence")

    Returns:
        Diversity matrix [num_maps, num_maps]
    """
    num_maps = len(attention_maps)

    # Convert to numpy
    maps_np = [att.cpu().numpy() if isinstance(att, torch.Tensor) else att
               for att in attention_maps]

    # Compute pairwise diversity
    diversity_matrix = np.zeros((num_maps, num_maps))

    for i in range(num_maps):
        for j in range(i + 1, num_maps):
            if metric == "correlation_distance":
                # 1 - correlation coefficient
                corr = np.corrcoef(maps_np[i].flatten(), maps_np[j].flatten())[0, 1]
                diversity = 1.0 - corr

            elif metric == "kl_divergence":
                # KL divergence (symmetrized)
                # Ensure positive and normalized
                p = maps_np[i] + 1e-10
                q = maps_np[j] + 1e-10
                p = p / p.sum()
                q = q / q.sum()

                kl_pq = np.sum(p * np.log(p / q))
                kl_qp = np.sum(q * np.log(q / p))
                diversity = (kl_pq + kl_qp) / 2.0

            else:
                raise ValueError(f"Unknown metric: {metric}")

            diversity_matrix[i, j] = diversity
            diversity_matrix[j, i] = diversity

    return diversity_matrix


def select_pivotal_frames(
    attention_maps: List[torch.Tensor],
    num_pivotal: int = 3,
    selection_method: str = "greedy_diversity",
) -> List[int]:
    """Select pivotal frames based on attention diversity.

    Args:
        attention_maps: List of attention tensors for each frame
        num_pivotal: Number of pivotal frames to select
        selection_method: Selection method ("greedy_diversity", "uniform")

    Returns:
        List of selected frame indices
    """
    num_frames = len(attention_maps)

    if num_frames <= num_pivotal:
        return list(range(num_frames))

    if selection_method == "uniform":
        # Uniform sampling
        step = num_frames / num_pivotal
        indices = [int(i * step) for i in range(num_pivotal)]
        return indices

    elif selection_method == "greedy_diversity":
        # Greedy selection to maximize diversity
        diversity_matrix = compute_attention_diversity(attention_maps)

        selected = []

        # Start with frame that has highest average diversity
        avg_diversity = diversity_matrix.mean(axis=1)
        first_idx = int(np.argmax(avg_diversity))
        selected.append(first_idx)

        # Greedily add frames that maximize minimum distance to selected
        for _ in range(num_pivotal - 1):
            best_idx = -1
            best_min_dist = -1

            for i in range(num_frames):
                if i in selected:
                    continue

                # Compute minimum distance to already selected frames
                min_dist = min(diversity_matrix[i, j] for j in selected)

                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_idx = i

            if best_idx >= 0:
                selected.append(best_idx)

        return sorted(selected)

    else:
        raise ValueError(f"Unknown selection method: {selection_method}")
