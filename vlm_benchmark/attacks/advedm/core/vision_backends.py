"""
Vision backend adapters for ADVEDM optimization.

This module provides a unified interface so ADVEDM-R can run either with:
- CLIP visual encoder (baseline/surrogate)
- Target VLM vision tower (paper-oriented path)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .attention_utils import (
    extract_cls_to_patch_attention,
    extract_cls_to_patch_attention_generic,
)
from .clip_contrastive_encoder import CLIPContrastiveEncoder


_CLIP_MEAN = torch.tensor([0.48145466, 0.45782750, 0.40821073]).view(1, 3, 1, 1)
_CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
_COS_EPS = 1e-6


def _normalize_clip(image: torch.Tensor) -> torch.Tensor:
    """Normalize image tensor in [0,1] using CLIP stats."""
    mean = _CLIP_MEAN.to(device=image.device, dtype=image.dtype)
    std = _CLIP_STD.to(device=image.device, dtype=image.dtype)
    return (image - mean) / std


def _openai_clip_forward_to_ln_post(
    vision_encoder: torch.nn.Module,
    image_normalized: torch.Tensor,
) -> torch.Tensor:
    """
    Shared forward pass for OpenAI-CLIP style ViT up to (and including) ln_post.

    Returns pre-projection hidden states [B, 1+N, D] (CLS at index 0, then patches).
    """
    conv1_dtype = vision_encoder.conv1.weight.dtype
    x = vision_encoder.conv1(image_normalized.to(conv1_dtype))
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
    x = vision_encoder.transformer(x)
    x = x.permute(1, 0, 2)
    x = vision_encoder.ln_post(x)
    return x  # [B, 1+N, D_inner]  (e.g. 1025 × 1024 for ViT-L/14@336px)


def _extract_openai_clip_patch_cls(
    vision_encoder: torch.nn.Module,
    image_normalized: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract PROJECTED patch/CLS embeddings (768-dim) for OpenAI-CLIP style ViT modules.

    Used for text-image alignment (masking, L_cls).
    """
    x = _openai_clip_forward_to_ln_post(vision_encoder, image_normalized)
    if getattr(vision_encoder, "proj", None) is not None:
        x = x @ vision_encoder.proj

    # Upcast to float32 before normalization: float16 forward with gradients
    # can produce NaN in CLS token during optimization (same fix as CLIPContrastiveEncoder).
    cls_embed = F.normalize(x[:, 0, :].float(), dim=-1, eps=_COS_EPS)
    patch_embeds = F.normalize(x[:, 1:, :].float(), dim=-1, eps=_COS_EPS)
    return patch_embeds, cls_embed


def _extract_openai_clip_patch_cls_noproj(
    vision_encoder: torch.nn.Module,
    image_normalized: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract PRE-PROJECTION patch/CLS embeddings (1024-dim) for spatial losses.

    Used for L_p and L_fix when the target VLM (e.g. LLaVA) uses the pre-projection
    hidden states rather than the CLIP contrastive projection output.
    """
    x = _openai_clip_forward_to_ln_post(vision_encoder, image_normalized)
    # No projection — return raw ln_post output
    cls_embed = F.normalize(x[:, 0, :].float(), dim=-1, eps=_COS_EPS)
    patch_embeds = F.normalize(x[:, 1:, :].float(), dim=-1, eps=_COS_EPS)
    return patch_embeds, cls_embed


def _forward_hf_style_vision(
    vision_encoder: torch.nn.Module,
    image_normalized: torch.Tensor,
    output_attentions: bool,
):
    """
    Forward helper for HF-style vision modules.

    Tries both positional and keyword arg calling conventions.
    """
    kwargs = {"output_attentions": output_attentions, "return_dict": True}
    try:
        return vision_encoder(image_normalized, **kwargs)
    except TypeError:
        return vision_encoder(pixel_values=image_normalized, **kwargs)


def _extract_hf_patch_cls(
    vision_encoder: torch.nn.Module,
    image_normalized: torch.Tensor,
    projection: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract patch/CLS embeddings for HF-style vision modules."""
    outputs = _forward_hf_style_vision(
        vision_encoder, image_normalized, output_attentions=False
    )

    if hasattr(outputs, "last_hidden_state"):
        tokens = outputs.last_hidden_state
    elif isinstance(outputs, tuple) and len(outputs) > 0:
        tokens = outputs[0]
    else:
        raise ValueError(
            "Cannot extract tokens from HF-style vision output. "
            f"Got type {type(outputs)}"
        )

    if projection is not None:
        tokens = tokens @ projection.to(dtype=tokens.dtype, device=tokens.device)

    cls_embed = F.normalize(tokens[:, 0, :].float(), dim=-1, eps=_COS_EPS)
    patch_embeds = F.normalize(tokens[:, 1:, :].float(), dim=-1, eps=_COS_EPS)
    return patch_embeds, cls_embed


@dataclass
class ClipVisionBackend:
    """CLIP vision backend used by ADVEDM-R baseline mode."""

    vision_encoder: torch.nn.Module
    device: str

    def __post_init__(self):
        image_size = getattr(self.vision_encoder, "image_size", 336)
        if isinstance(image_size, tuple):
            image_size = image_size[0]
        self.image_size = int(image_size)
        self.patch_size = int(self.vision_encoder.conv1.kernel_size[0])
        self.backend_name = "clip"
        self.uses_fallback_attention = False
        self.text_encoder = None

    def extract_patch_cls_embeddings(
        self,
        image: torch.Tensor,
        normalized: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract PROJECTED 768-dim features (for masking / L_cls text alignment)."""
        image_normalized = image if normalized else _normalize_clip(image)
        return _extract_openai_clip_patch_cls(self.vision_encoder, image_normalized)

    def extract_patch_cls_embeddings_noproj(
        self,
        image: torch.Tensor,
        normalized: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract PRE-PROJECTION 1024-dim features for spatial losses (L_p, L_fix)."""
        image_normalized = image if normalized else _normalize_clip(image)
        return _extract_openai_clip_patch_cls_noproj(self.vision_encoder, image_normalized)

    def extract_cls_to_patch_attention(
        self,
        image: torch.Tensor,
        normalized: bool = False,
    ) -> torch.Tensor:
        image_normalized = image if normalized else _normalize_clip(image)
        return extract_cls_to_patch_attention(self.vision_encoder, image_normalized)


class TargetVLMVisionBackend:
    """
    Target-VLM backend (paper-oriented ADVEDM-R path).

    Notes:
    - Uses target VLM vision tower for patch/attention extraction.
    - Uses CLIP text encoder for text embeddings and optional projection
      alignment when target tower outputs unprojected hidden states.
    """

    def __init__(
        self,
        model_path: str,
        device: str,
        text_clip_model: str = "ViT-L/14@336px",
    ):
        from .llava_vision_encoder import LLaVAVisionEncoder

        self.device = device
        self.backend_name = "target_vlm"
        self.uses_fallback_attention = False

        self.vlm_encoder = LLaVAVisionEncoder(model_path=model_path, device=device)
        self.vision_encoder = self.vlm_encoder.vision_tower
        self.image_size = int(self.vlm_encoder.image_size)

        # Use CLIP text encoder for Eq.3/5 text embeddings and optional projection.
        self.text_encoder = CLIPContrastiveEncoder(
            model_name=text_clip_model,
            device=device,
        )
        text_proj = getattr(self.text_encoder.model, "text_projection", None)
        self._text_dim = int(text_proj.shape[1]) if text_proj is not None else None
        self._alignment_proj = getattr(self.text_encoder.model.visual, "proj", None)
        self._alignment_in_dim = (
            int(self._alignment_proj.shape[0]) if self._alignment_proj is not None else None
        )
        self._cached_hf_projection: Optional[torch.Tensor] = None
        self._checked_hf_projection = False

        self._is_openai_clip_like = hasattr(self.vision_encoder, "conv1") and hasattr(
            self.vision_encoder, "transformer"
        )
        if self._is_openai_clip_like:
            self.patch_size = int(self.vision_encoder.conv1.kernel_size[0])
        else:
            patch_size = None
            if hasattr(self.vision_encoder, "config"):
                patch_size = getattr(self.vision_encoder.config, "patch_size", None)
            self.patch_size = int(patch_size) if patch_size is not None else 14

    def extract_patch_cls_embeddings(
        self,
        image: torch.Tensor,
        normalized: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract PROJECTED 768-dim features (for masking / L_cls text alignment)."""
        image_normalized = image if normalized else _normalize_clip(image)
        if self._is_openai_clip_like:
            return _extract_openai_clip_patch_cls(self.vision_encoder, image_normalized)

        patch_embeds, cls_embed = _extract_hf_patch_cls(
            self.vision_encoder,
            image_normalized,
            projection=self._resolve_projection_for_hf(image_normalized),
        )
        if self._text_dim is not None and patch_embeds.shape[-1] != self._text_dim:
            raise RuntimeError(
                "Target VLM patch/text dimensions are misaligned. "
                f"Patch dim={patch_embeds.shape[-1]}, text dim={self._text_dim}. "
                "Use a compatible target model or run with --vision_backend clip."
            )
        return patch_embeds, cls_embed

    def extract_patch_cls_embeddings_noproj(
        self,
        image: torch.Tensor,
        normalized: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract PRE-PROJECTION 1024-dim features for spatial losses (L_p, L_fix).

        Returns the vision tower's raw hidden states without any contrastive projection,
        matching the feature space that LLaVA's MLP projector actually receives.
        """
        image_normalized = image if normalized else _normalize_clip(image)
        if self._is_openai_clip_like:
            return _extract_openai_clip_patch_cls_noproj(self.vision_encoder, image_normalized)
        # HF-style tower (e.g. LLaVA's CLIPVisionModel): skip projection entirely
        return _extract_hf_patch_cls(self.vision_encoder, image_normalized, projection=None)

    def extract_cls_to_patch_attention(
        self,
        image: torch.Tensor,
        normalized: bool = False,
    ) -> torch.Tensor:
        image_normalized = image if normalized else _normalize_clip(image)
        try:
            return extract_cls_to_patch_attention_generic(self.vision_encoder, image_normalized)
        except Exception:
            # Fallback: return uniform attention without any additional forward pass.
            # extract_patch_cls_embeddings would be a full extra LLaVA forward pass just
            # to obtain the shape — wasteful and OOM-prone during long optimization loops.
            # Use the known image/patch dimensions from self instead.
            B = image.shape[0]
            N = (self.image_size // self.patch_size) ** 2
            self.uses_fallback_attention = True
            return torch.full(
                (B, N),
                1.0 / max(N, 1),
                device=image.device,
                dtype=torch.float32,
            )

    def _resolve_projection_for_hf(
        self,
        image_normalized: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        Provide projection for HF-style vision towers when available.

        If hidden width matches CLIP visual projection input width, we reuse the
        CLIP projection matrix to align patch features to text space.
        """
        if self._alignment_proj is None:
            return None
        if self._checked_hf_projection:
            return self._cached_hf_projection

        with torch.no_grad():
            outputs = _forward_hf_style_vision(
                self.vision_encoder,
                image_normalized[:1],
                output_attentions=False,
            )
            if hasattr(outputs, "last_hidden_state"):
                hidden = outputs.last_hidden_state
            elif isinstance(outputs, tuple) and len(outputs) > 0:
                hidden = outputs[0]
            else:
                return None

        hidden_dim = int(hidden.shape[-1])
        if hidden_dim != self._alignment_in_dim:
            self._checked_hf_projection = True
            self._cached_hf_projection = None
            return None
        self._checked_hf_projection = True
        self._cached_hf_projection = self._alignment_proj
        return self._cached_hf_projection
