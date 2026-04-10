"""
Multi-CLIP Surrogate Ensemble for Black-Box Transfer Attacks

Manages an ensemble of 6 CLIP vision-text encoders matching the reference
implementation (VLMTransfer/change_semantic.py):
  - ViT-L/14        (OpenAI CLIP)
  - ViT-B/32        (open_clip / LAION)
  - ViT-B/16        (open_clip / LAION)
  - convnext_large_d_320 (open_clip / LAION)
  - ViT-bigG-14     (open_clip / LAION)
  - ViT-SO400M-14-SigLIP-384 (open_clip / webli)

Each surrogate has different image_size, patch_size, grid, and embed_dim.
Feature extraction is differentiable for gradient flow through SSA-CWA.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

# CLIP normalization constants (ImageNet-based, used by OpenAI CLIP and most open_clip)
CLIP_MEAN = (0.48145466, 0.45782750, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# SigLIP uses different normalization (0.5, 0.5, 0.5)
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)

_COS_EPS = 1e-6


@dataclass
class CLIPModelSpec:
    """Specification for a single CLIP surrogate model."""
    name: str               # e.g. "ViT-L/14"
    source: str             # "openai_clip" or "open_clip"
    pretrained: str = ""    # open_clip pretrained tag (e.g. "laion2b_s34b_b88k")
    image_size: int = 224
    norm_mean: Tuple[float, ...] = CLIP_MEAN
    norm_std: Tuple[float, ...] = CLIP_STD


# Paper ensemble specifications — matches VLMTransfer/change_semantic.py exactly
PAPER_ENSEMBLE: List[CLIPModelSpec] = [
    CLIPModelSpec(
        name="ViT-L/14",
        source="openai_clip",
        image_size=224,
    ),
    CLIPModelSpec(
        name="ViT-B-32",
        source="open_clip",
        pretrained="laion2b_s34b_b79k",
        image_size=224,
    ),
    CLIPModelSpec(
        name="ViT-B-16",
        source="open_clip",
        pretrained="laion2b_s34b_b88k",
        image_size=224,
    ),
    # ConvNeXt excluded: ADVEDM needs ViT-style patch tokens for attention/mask;
    # ConvNeXt uses a different stem architecture incompatible with patch extraction.
    # CLIPModelSpec(
    #     name="convnext_large_d_320",
    #     source="open_clip",
    #     pretrained="laion2b_s29b_b131k_ft_soup",
    #     image_size=320,
    # ),
    CLIPModelSpec(
        name="ViT-bigG-14",
        source="open_clip",
        pretrained="laion2b_s39b_b160k",
        image_size=224,
    ),
    CLIPModelSpec(
        name="ViT-SO400M-14-SigLIP-384",
        source="open_clip",
        pretrained="webli",
        image_size=384,
        norm_mean=SIGLIP_MEAN,
        norm_std=SIGLIP_STD,
    ),
]


@dataclass
class SurrogateModel:
    """A loaded surrogate model with architecture metadata."""
    spec: CLIPModelSpec
    vision_encoder: torch.nn.Module
    text_encoder: object            # clip.model or open_clip model for tokenize+encode
    image_size: int
    patch_size: int
    grid_size: int                  # image_size // patch_size
    num_patches: int                # grid_size ** 2
    embed_dim: int                  # hidden dim of patch tokens
    device: torch.device
    is_openai_clip: bool            # True = OpenAI CLIP architecture
    has_cls: bool = True            # False for SigLIP (no CLS token)
    _full_model: object = None      # keep reference for text encoding


class EnsembleEncoder:
    """
    Load and manage an ensemble of CLIP surrogates for black-box transfer.

    Usage:
        ensemble = EnsembleEncoder(PAPER_ENSEMBLE, device="cuda:0")
        ensemble.load_all()
        for sm in ensemble.surrogates:
            patches, cls_emb = ensemble.extract_features(sm, image)
    """

    def __init__(
        self,
        specs: List[CLIPModelSpec] = None,
        device: str = "cuda",
    ):
        self.specs = specs or PAPER_ENSEMBLE
        self.device = torch.device(device)
        self.surrogates: List[SurrogateModel] = []

    def load_all(self) -> None:
        """Load all surrogate models."""
        for spec in self.specs:
            sm = self._load_one(spec)
            self.surrogates.append(sm)
            print(f"  Loaded {spec.name}: {sm.image_size}px, "
                  f"{sm.grid_size}x{sm.grid_size}={sm.num_patches} patches, "
                  f"dim={sm.embed_dim}")
        print(f"Ensemble: {len(self.surrogates)} surrogates loaded on {self.device}")

    def _load_one(self, spec: CLIPModelSpec) -> SurrogateModel:
        """Load a single surrogate model."""
        if spec.source == "openai_clip":
            return self._load_openai_clip(spec)
        elif spec.source == "open_clip":
            return self._load_open_clip(spec)
        else:
            raise ValueError(f"Unknown source: {spec.source}")

    def _load_openai_clip(self, spec: CLIPModelSpec) -> SurrogateModel:
        """Load an OpenAI CLIP model (ViT-L/14, ViT-B/32, etc.)."""
        import clip

        model, _ = clip.load(spec.name, device=self.device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        vis = model.visual
        patch_size = int(vis.conv1.kernel_size[0])
        grid_size = spec.image_size // patch_size
        num_patches = grid_size * grid_size
        embed_dim = int(vis.ln_post.normalized_shape[0])

        return SurrogateModel(
            spec=spec,
            vision_encoder=vis,
            text_encoder=model,
            image_size=spec.image_size,
            patch_size=patch_size,
            grid_size=grid_size,
            num_patches=num_patches,
            embed_dim=embed_dim,
            device=self.device,
            is_openai_clip=True,
            has_cls=True,  # OpenAI CLIP always has class_embedding
            _full_model=model,
        )

    def _load_open_clip(self, spec: CLIPModelSpec) -> SurrogateModel:
        """Load an open_clip model (ViT-G-14, SigLIP-384, etc.)."""
        import open_clip

        model, _, _ = open_clip.create_model_and_transforms(
            spec.name, pretrained=spec.pretrained, device=self.device,
        )
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        vis = model.visual
        # open_clip stores trunk as vis.trunk for timm-based models
        trunk = getattr(vis, "trunk", vis)

        # Detect patch_size
        if hasattr(trunk, "patch_embed"):
            pe = trunk.patch_embed
            if hasattr(pe, "proj"):
                patch_size = int(pe.proj.kernel_size[0])
            elif hasattr(pe, "patch_size"):
                ps = pe.patch_size
                patch_size = ps[0] if isinstance(ps, (tuple, list)) else int(ps)
            else:
                raise ValueError(f"Cannot detect patch_size for {spec.name}")
        elif hasattr(trunk, "conv1"):
            patch_size = int(trunk.conv1.kernel_size[0])
        elif hasattr(trunk, "stem"):
            # ConvNeXt: stem.0 is Conv2d with stride = patch_size
            for m in trunk.stem.modules():
                if hasattr(m, "kernel_size") and hasattr(m, "stride"):
                    patch_size = int(m.kernel_size[0]) if isinstance(m.kernel_size, tuple) else int(m.kernel_size)
                    break
            else:
                raise ValueError(f"Cannot detect patch_size from stem for {spec.name}")
        else:
            raise ValueError(f"Cannot detect patch_embed for {spec.name}")

        grid_size = spec.image_size // patch_size
        num_patches = grid_size * grid_size

        # Detect embed_dim
        if hasattr(trunk, "embed_dim"):
            embed_dim = int(trunk.embed_dim)
        elif hasattr(trunk, "num_features"):
            embed_dim = int(trunk.num_features)
        elif hasattr(trunk, "ln_post"):
            embed_dim = int(trunk.ln_post.normalized_shape[0])
        else:
            # Fallback: from first block
            blocks = getattr(trunk, "blocks", None)
            if blocks is not None and len(blocks) > 0:
                embed_dim = int(blocks[0].attn.qkv.in_features)
            else:
                raise ValueError(f"Cannot detect embed_dim for {spec.name}")

        # Detect if this model has a CLS token (SigLIP does not)
        _trunk_for_cls = getattr(vis, "trunk", vis)
        _has_cls = hasattr(_trunk_for_cls, "cls_token") and _trunk_for_cls.cls_token is not None

        return SurrogateModel(
            spec=spec,
            vision_encoder=vis,
            text_encoder=model,
            image_size=spec.image_size,
            patch_size=patch_size,
            grid_size=grid_size,
            num_patches=num_patches,
            embed_dim=embed_dim,
            device=self.device,
            is_openai_clip=False,
            has_cls=_has_cls,
            _full_model=model,
        )

    # ------------------------------------------------------------------
    # Differentiable image preprocessing
    # ------------------------------------------------------------------

    def resize_and_normalize(
        self,
        image: torch.Tensor,       # [B, 3, H, W] in [0,1]
        surrogate: SurrogateModel,
    ) -> torch.Tensor:
        """Resize and CLIP-normalize (differentiable)."""
        sz = surrogate.image_size
        if image.shape[2] != sz or image.shape[3] != sz:
            x = F.interpolate(image, size=(sz, sz), mode="bilinear", align_corners=False)
        else:
            x = image

        mean = torch.tensor(surrogate.spec.norm_mean, device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
        std = torch.tensor(surrogate.spec.norm_std, device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
        return (x - mean) / std

    # ------------------------------------------------------------------
    # Feature extraction (differentiable)
    # ------------------------------------------------------------------

    def extract_features(
        self,
        surrogate: SurrogateModel,
        image_normalized: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract patch and CLS embeddings from normalized image.

        Returns:
            patches: [B, N, D] normalized patch embeddings
            cls_emb: [B, D] normalized CLS embedding
        """
        # open_clip ViT models share OpenAI CLIP's internal structure (conv1, transformer)
        vis = surrogate.vision_encoder
        trunk = getattr(vis, "trunk", vis)
        if surrogate.is_openai_clip or hasattr(trunk, "conv1"):
            return self._extract_features_openai(surrogate, image_normalized)
        else:
            return self._extract_features_open_clip(surrogate, image_normalized)

    def _extract_features_openai(
        self,
        surrogate: SurrogateModel,
        image_normalized: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """OpenAI CLIP: conv1 → class_embedding → transformer → ln_post."""
        vis = surrogate.vision_encoder
        conv1_dtype = vis.conv1.weight.dtype
        x = vis.conv1(image_normalized.to(conv1_dtype))
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        x = torch.cat([
            vis.class_embedding.to(x.dtype) +
            torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x
        ], dim=1)
        x = x + vis.positional_embedding.to(x.dtype)
        x = vis.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = vis.transformer(x)
        x = x.permute(1, 0, 2)
        x = vis.ln_post(x)
        x = x.float()

        cls_emb = F.normalize(x[:, 0, :], dim=-1, eps=_COS_EPS)
        patches = F.normalize(x[:, 1:, :], dim=-1, eps=_COS_EPS)
        return patches, cls_emb

    def _extract_features_open_clip(
        self,
        surrogate: SurrogateModel,
        image_normalized: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """open_clip / timm ViT: patch_embed → blocks → norm."""
        vis = surrogate.vision_encoder
        trunk = getattr(vis, "trunk", vis)

        x = trunk.patch_embed(image_normalized)  # [B, N, D]

        # CLS token
        if hasattr(trunk, "cls_token") and trunk.cls_token is not None:
            cls_token = trunk.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls_token, x], dim=1)
            has_cls = True
        else:
            has_cls = False

        # Position embedding
        if hasattr(trunk, "pos_embed") and trunk.pos_embed is not None:
            x = x + trunk.pos_embed
        # Some open_clip models use positional_embedding
        elif hasattr(trunk, "positional_embedding") and trunk.positional_embedding is not None:
            x = x + trunk.positional_embedding

        # Pre-norm (some models)
        if hasattr(trunk, "patch_drop"):
            x = trunk.patch_drop(x)
        if hasattr(trunk, "norm_pre"):
            x = trunk.norm_pre(x)

        # Transformer blocks
        if hasattr(trunk, "blocks"):
            for blk in trunk.blocks:
                x = blk(x)
        elif hasattr(trunk, "resblocks"):
            x = x.permute(1, 0, 2)
            for blk in trunk.resblocks:
                x = blk(x)
            x = x.permute(1, 0, 2)

        # Final norm
        if hasattr(trunk, "norm"):
            x = trunk.norm(x)
        elif hasattr(trunk, "ln_post"):
            x = trunk.ln_post(x)

        x = x.float()

        if has_cls:
            cls_emb = F.normalize(x[:, 0, :], dim=-1, eps=_COS_EPS)
            patches = F.normalize(x[:, 1:, :], dim=-1, eps=_COS_EPS)
        else:
            # No CLS token (e.g., SigLIP): use mean-pool as CLS surrogate
            cls_emb = F.normalize(x.mean(dim=1), dim=-1, eps=_COS_EPS)
            patches = F.normalize(x, dim=-1, eps=_COS_EPS)

        return patches, cls_emb

    # ------------------------------------------------------------------
    # Attention extraction (differentiable-compatible, detached for use)
    # ------------------------------------------------------------------

    def extract_attention(
        self,
        surrogate: SurrogateModel,
        image_normalized: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract CLS→patch attention as [B, N] vector.

        For OpenAI CLIP: reuse existing manual extraction.
        For open_clip: use output_attentions or manual QKV extraction.
        """
        vis = surrogate.vision_encoder
        trunk = getattr(vis, "trunk", vis)
        if surrogate.is_openai_clip or hasattr(trunk, "conv1"):
            from .attention_utils import extract_cls_to_patch_attention
            return extract_cls_to_patch_attention(
                surrogate.vision_encoder, image_normalized,
                average_across_layers=True, renormalize=False,
            )
        else:
            return self._extract_attention_open_clip(surrogate, image_normalized)

    def _extract_attention_open_clip(
        self,
        surrogate: SurrogateModel,
        image_normalized: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract CLS→patch attention for open_clip models.

        Uses manual QKV decomposition through transformer blocks.
        """
        vis = surrogate.vision_encoder
        trunk = getattr(vis, "trunk", vis)

        x = trunk.patch_embed(image_normalized)

        has_cls = hasattr(trunk, "cls_token") and trunk.cls_token is not None
        if has_cls:
            cls_token = trunk.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls_token, x], dim=1)

        if hasattr(trunk, "pos_embed") and trunk.pos_embed is not None:
            x = x + trunk.pos_embed
        elif hasattr(trunk, "positional_embedding") and trunk.positional_embedding is not None:
            x = x + trunk.positional_embedding

        if hasattr(trunk, "patch_drop"):
            x = trunk.patch_drop(x)
        if hasattr(trunk, "norm_pre"):
            x = trunk.norm_pre(x)

        if not has_cls:
            # Without CLS, return uniform attention
            B = x.shape[0]
            N = surrogate.num_patches
            return torch.ones(B, N, device=x.device, dtype=x.dtype) / N

        # Walk through blocks, extract attention at each layer
        all_cls_attn = []
        if hasattr(trunk, "blocks"):
            for blk in trunk.blocks:
                # timm-style block: blk.attn is the attention module
                attn_mod = blk.attn
                x_norm = blk.norm1(x)

                # QKV projection
                B_size, seq_len, dim = x_norm.shape
                if hasattr(attn_mod, "qkv"):
                    qkv = attn_mod.qkv(x_norm)
                    num_heads = attn_mod.num_heads
                    head_dim = dim // num_heads
                    qkv = qkv.reshape(B_size, seq_len, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
                    q, k, v = qkv[0], qkv[1], qkv[2]
                else:
                    # Fallback: skip attention extraction for this block
                    x = blk(x)
                    continue

                scale = head_dim ** -0.5
                attn_scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
                attn_weights = F.softmax(attn_scores, dim=-1)
                attn_avg = attn_weights.mean(dim=1)  # [B, seq, seq]
                cls_to_patches = attn_avg[:, 0, 1:]  # [B, N]
                all_cls_attn.append(cls_to_patches)

                # Continue forward pass through block
                x = blk(x)

        if all_cls_attn:
            return torch.stack(all_cls_attn, dim=0).mean(dim=0)
        else:
            # Fallback: uniform attention
            B = image_normalized.shape[0]
            N = surrogate.num_patches
            return torch.ones(B, N, device=image_normalized.device) / N

    # ------------------------------------------------------------------
    # Text encoding
    # ------------------------------------------------------------------

    def encode_text(
        self,
        surrogate: SurrogateModel,
        texts: List[str],
    ) -> torch.Tensor:
        """Encode text using surrogate's text encoder. Returns [N, D] normalized."""
        if surrogate.is_openai_clip:
            import clip
            tokens = clip.tokenize(texts).to(surrogate.device)
            with torch.no_grad():
                text_features = surrogate._full_model.encode_text(tokens).float()
            return F.normalize(text_features, dim=-1, eps=_COS_EPS)
        else:
            import open_clip
            tokenizer = open_clip.get_tokenizer(surrogate.spec.name)
            tokens = tokenizer(texts).to(surrogate.device)
            with torch.no_grad():
                text_features = surrogate._full_model.encode_text(tokens).float()
            return F.normalize(text_features, dim=-1, eps=_COS_EPS)

    def project_patches(
        self,
        surrogate: SurrogateModel,
        patches: torch.Tensor,
    ) -> torch.Tensor:
        """
        Project patch features to the model's output embedding space.

        OpenAI CLIP and open_clip models with vis.proj multiply by a linear
        projection matrix (e.g. 1024→768 for ViT-L/14).  SigLIP uses an
        Identity head so no projection is needed and patches are already in
        the output space.

        Args:
            surrogate: Loaded surrogate model.
            patches: [B, N, D_inner] patch embeddings (pre-projection).

        Returns:
            [B, N, D_out] projected and L2-normalized patch embeddings.
        """
        vis = surrogate.vision_encoder
        proj = getattr(vis, "proj", None)
        if proj is not None:
            # proj: [D_inner, D_out] — apply to each patch
            projected = patches @ proj.to(patches.dtype)
        else:
            # No projection (e.g. SigLIP) — already in output space
            projected = patches
        return F.normalize(projected, dim=-1, eps=_COS_EPS)

    # ------------------------------------------------------------------
    # Automatic background region detection (replaces GPT annotator)
    # ------------------------------------------------------------------

    def clip_bbox_from_attention(
        self,
        image_tensor: torch.Tensor,
        region_pixels: int = 100,
        source_size: int = 224,
    ) -> Tuple[int, int, int, int]:
        """
        Find a background region for semantic injection using CLIP attention.

        Selects the contiguous k×k patch block with the lowest CLS→patch
        attention (i.e., the region the model "looks at" least).  This
        replaces the paper's "manual selection" / GPT annotation step.

        Args:
            image_tensor: [1, 3, H, W] in [0,1] at source_size resolution.
            region_pixels: Desired side length of the injection region in pixels.
            source_size: Pixel size the returned bbox refers to.

        Returns:
            (x, y, w, h) bounding box in source_size pixel coordinates.
        """
        for sm in self.surrogates:
            if not sm.has_cls:
                continue  # SigLIP returns uniform attention; skip
            with torch.no_grad():
                img_norm = self.resize_and_normalize(image_tensor, sm)
                A = self.extract_attention(sm, img_norm)  # [1, N]
            A_map = A.squeeze(0).cpu().float().reshape(sm.grid_size, sm.grid_size)

            # How many patches fit in region_pixels
            k = max(1, min(round(region_pixels / sm.patch_size), sm.grid_size))

            best_sum = float("inf")
            best_r, best_c = 0, 0
            for r in range(sm.grid_size - k + 1):
                for c in range(sm.grid_size - k + 1):
                    s = A_map[r:r + k, c:c + k].sum().item()
                    if s < best_sum:
                        best_sum, best_r, best_c = s, r, c

            # Convert patch → pixel at surrogate scale, then rescale to source_size
            scale = source_size / sm.image_size
            x = int(best_c * sm.patch_size * scale)
            y = int(best_r * sm.patch_size * scale)
            w = max(1, int(k * sm.patch_size * scale))
            h = max(1, int(k * sm.patch_size * scale))
            return (x, y, w, h)

        raise RuntimeError(
            "clip_bbox_from_attention: no CLS-capable surrogate found in ensemble. "
            "Ensure at least one of the CLIP models has a CLS token (e.g. ViT-L/14)."
        )

    # ------------------------------------------------------------------
    # Bbox → patch index mapping
    # ------------------------------------------------------------------

    @staticmethod
    def bbox_to_target_indices(
        bbox_pixels: Tuple[int, int, int, int],
        surrogate: SurrogateModel,
        source_image_size: int,
    ) -> torch.Tensor:
        """
        Convert pixel bbox (x, y, w, h) to patch indices for a given surrogate.

        Args:
            bbox_pixels: (x, y, w, h) in source image coordinates
            surrogate: Target surrogate model
            source_image_size: Size of the source image the bbox refers to

        Returns:
            Tensor of patch indices [k]
        """
        x, y, w, h = bbox_pixels
        gs = surrogate.grid_size
        ps = surrogate.patch_size
        sz = surrogate.image_size

        # Scale bbox to surrogate's image size
        scale = sz / source_image_size
        x_s = int(x * scale)
        y_s = int(y * scale)
        w_s = max(1, int(w * scale))
        h_s = max(1, int(h * scale))

        # Clamp
        x_s = max(0, min(x_s, sz - 1))
        y_s = max(0, min(y_s, sz - 1))
        w_s = min(w_s, sz - x_s)
        h_s = min(h_s, sz - y_s)

        # Pixel → patch coords
        px_start = x_s // ps
        py_start = y_s // ps
        px_end = min((x_s + w_s + ps - 1) // ps, gs)
        py_end = min((y_s + h_s + ps - 1) // ps, gs)

        indices = []
        for i in range(py_start, py_end):
            for j in range(px_start, px_end):
                indices.append(i * gs + j)

        return torch.tensor(indices, dtype=torch.long)
