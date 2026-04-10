"""
Attention utilities used by paper-faithful ADVEDM implementations.

This module intentionally keeps only the vector-form attention operations used
by the current ADVEDM-A / ADVEDM-R pipelines.
"""

import torch
import torch.nn.functional as F


def _extract_cls_to_patch_from_attentions(
    attentions,
    average_across_layers: bool = True,
    renormalize: bool = False,
) -> torch.Tensor:
    """
    Convert layer attentions to CLS->patch vector [B, N].

    Args:
        attentions: Sequence of [B, heads, seq, seq] tensors.
        average_across_layers: Average over all layers if True; else use last.
        renormalize: Re-normalize vector to sum to 1 if True.
    """
    if not attentions:
        raise ValueError("No attention tensors were provided.")

    cls_vectors = []
    for layer_attn in attentions:
        # Mean over heads: [B, seq, seq]
        attn_avg = layer_attn.mean(dim=1)
        cls_vectors.append(attn_avg[:, 0, 1:])

    if average_across_layers:
        cls_to_patches = torch.stack(cls_vectors, dim=0).mean(dim=0)
    else:
        cls_to_patches = cls_vectors[-1]

    if renormalize:
        denom = cls_to_patches.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        cls_to_patches = cls_to_patches / denom
    return cls_to_patches


def extract_cls_to_patch_attention(
    vision_encoder,
    images_normalized: torch.Tensor,
    average_across_layers: bool = True,
    renormalize: bool = False,
) -> torch.Tensor:
    """
    Extract CLS->patch attention as a vector per image.

    Args:
        vision_encoder: CLIP vision model (e.g., model.visual).
        images_normalized: Preprocessed images [B, C, H, W].
        average_across_layers: Average CLS->patch attention across all layers.
            If False, use only the last layer.
        renormalize: Optional implementation choice. If True, re-normalize the
            extracted CLS->patch vector to sum to 1.

    Returns:
        Tensor [B, N] where N is the number of patch tokens.
    """
    device = images_normalized.device
    conv1_dtype = vision_encoder.conv1.weight.dtype
    images_norm = images_normalized.to(dtype=conv1_dtype, device=device)

    x = vision_encoder.conv1(images_norm)
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)

    x = torch.cat(
        [
            vision_encoder.class_embedding.to(x.dtype)
            + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x,
        ],
        dim=1,
    )
    x = x + vision_encoder.positional_embedding.to(x.dtype)
    x = vision_encoder.ln_pre(x)
    x = x.permute(1, 0, 2)

    num_blocks = len(vision_encoder.transformer.resblocks)
    all_cls_attentions = []

    for layer_idx in range(num_blocks):
        block = vision_encoder.transformer.resblocks[layer_idx]
        x_ln = block.ln_1(x)

        seq_len, batch_size, dim = x_ln.shape
        num_heads = block.attn.num_heads
        head_dim = dim // num_heads

        qkv = F.linear(x_ln, block.attn.in_proj_weight, block.attn.in_proj_bias)
        qkv = qkv.reshape(seq_len, batch_size, 3, num_heads, head_dim)
        qkv = qkv.permute(2, 1, 3, 0, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]
        scale = head_dim ** -0.5
        # Compute attention in float32 for numerical stability.
        attn_scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
        attn_scores = attn_scores - attn_scores.amax(dim=-1, keepdim=True)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0, posinf=0.0, neginf=0.0)
        attn_denom = attn_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        attn_weights = attn_weights / attn_denom

        attn_avg = attn_weights.mean(dim=1)
        cls_to_patches = attn_avg[:, 0, 1:]
        all_cls_attentions.append(cls_to_patches)

        attn_output = torch.matmul(attn_weights, v.float())
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, dim).transpose(0, 1)
        attn_output = F.linear(
            attn_output.to(dtype=block.attn.out_proj.weight.dtype),
            block.attn.out_proj.weight,
            block.attn.out_proj.bias,
        )

        x = x + attn_output
        x = x + block.mlp(block.ln_2(x))

    if average_across_layers:
        all_cls_attentions = torch.stack(all_cls_attentions, dim=0)
        cls_to_patches_final = all_cls_attentions.mean(dim=0)
    else:
        cls_to_patches_final = all_cls_attentions[-1]

    if renormalize:
        denom = cls_to_patches_final.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        cls_to_patches_final = cls_to_patches_final / denom

    return cls_to_patches_final


def extract_cls_to_patch_attention_generic(
    vision_encoder,
    images_normalized: torch.Tensor,
    average_across_layers: bool = True,
    renormalize: bool = False,
) -> torch.Tensor:
    """
    Generic CLS->patch attention extraction for both OpenAI-CLIP and HF-style towers.

    Strategy:
    1) OpenAI-CLIP style (conv1+transformer): use exact manual extraction.
    2) HF-style modules: call forward(..., output_attentions=True) and aggregate.
    """
    if hasattr(vision_encoder, "conv1") and hasattr(vision_encoder, "transformer"):
        return extract_cls_to_patch_attention(
            vision_encoder,
            images_normalized,
            average_across_layers=average_across_layers,
            renormalize=renormalize,
        )

    kwargs = {"output_attentions": True, "return_dict": True}
    try:
        outputs = vision_encoder(images_normalized, **kwargs)
    except TypeError:
        outputs = vision_encoder(pixel_values=images_normalized, **kwargs)

    attentions = getattr(outputs, "attentions", None)
    if attentions is None and isinstance(outputs, tuple) and len(outputs) > 0:
        # Some models may return attentions as the last tuple element.
        maybe_attn = outputs[-1]
        if isinstance(maybe_attn, (list, tuple)):
            attentions = maybe_attn

    # Filter out None entries: sdpa-backed models (e.g. LLaVA) silently ignore
    # output_attentions=True and return a list of Nones instead of raising.
    if attentions is not None:
        attentions = [a for a in attentions if a is not None]

    if not attentions:
        raise ValueError(
            "HF-style attention extraction failed: no attentions returned by vision encoder."
        )

    return _extract_cls_to_patch_from_attentions(
        attentions,
        average_across_layers=average_across_layers,
        renormalize=renormalize,
    )


def reallocate_attention_vector(
    A_orig: torch.Tensor,
    A_ref: torch.Tensor,
    mask: torch.Tensor,
    beta: float = 0.4,
) -> torch.Tensor:
    """
    Reallocate attention per Equation 10 in vector form.

    Args:
        A_orig: Clean image CLS->patch attention [B, N].
        A_ref: Reference image CLS->patch attention [B, N].
        mask: Binary mask [B, N], 0=inject/remove, 1=preserve.
        beta: Reallocation scaling factor.

    Returns:
        Reallocated attention [B, N].
    """
    if A_orig.shape != A_ref.shape or A_orig.shape != mask.shape:
        raise ValueError(
            f"Shape mismatch: A_orig={tuple(A_orig.shape)}, "
            f"A_ref={tuple(A_ref.shape)}, mask={tuple(mask.shape)}"
        )

    return mask * (1 - beta) * A_orig + (1 - mask) * beta * A_ref


def compute_attention_weighted_features_vector(
    A_vec: torch.Tensor,
    patch_embeds: torch.Tensor,
) -> torch.Tensor:
    """
    Compute attention-weighted patch embeddings A * [patch].

    Args:
        A_vec: Attention weights [B, N].
        patch_embeds: Patch embeddings [B, N, D].

    Returns:
        Weighted patch features [B, N, D].
    """
    if A_vec.ndim != 2 or patch_embeds.ndim != 3:
        raise ValueError(
            f"Expected A_vec [B,N] and patch_embeds [B,N,D], got "
            f"{tuple(A_vec.shape)} and {tuple(patch_embeds.shape)}"
        )
    if A_vec.shape[0] != patch_embeds.shape[0] or A_vec.shape[1] != patch_embeds.shape[1]:
        raise ValueError(
            f"Shape mismatch: A_vec={tuple(A_vec.shape)}, patch_embeds={tuple(patch_embeds.shape)}"
        )

    return A_vec.unsqueeze(-1) * patch_embeds
