"""
CLIP Contrastive Space Encoder

This module extracts vision and text features in CLIP's properly aligned
768-dim contrastive space (after projection), NOT the mismatched hidden states.

Key fix: Both vision patches and text are in the same 768-dim space where
cosine similarity is semantically meaningful.
"""

import torch
import clip
from typing import Tuple


class CLIPContrastiveEncoder:
    """
    Extracts vision patch features and text features in CLIP's contrastive space (768-dim)
    """

    def __init__(self, model_name: str = "ViT-L/14@336px", device: str = "cuda"):
        """
        Initialize CLIP encoder

        Args:
            model_name: CLIP model name
            device: Device to run on
        """
        self.device = device
        print(f"Loading CLIP model: {model_name}...")

        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()

        # Explicitly move all model components to device
        self.model = self.model.to(device)
        self.model.visual = self.model.visual.to(device)
        if device == "cpu":
            # Avoid half-precision instability on CPU (can produce NaNs in optimization).
            self.model = self.model.float()
            self.model.visual = self.model.visual.float()

        # Image size for this model
        if "336" in model_name:
            self.image_size = 336
        elif "224" in model_name:
            self.image_size = 224
        else:
            self.image_size = 224  # default
        patch_size = int(self.model.visual.conv1.kernel_size[0])
        self._grid_size = self.image_size // patch_size
        self._num_patches = self._grid_size * self._grid_size

        print(f"✓ CLIP model loaded")
        print(f"  Image size: {self.image_size}x{self.image_size}")
        print(f"  Patch grid: {self._grid_size}x{self._grid_size} = {self._num_patches} patches")
        print(f"  Contrastive dimension: 768")

    def get_num_patches(self) -> int:
        """Get number of patches (excluding CLS token)"""
        return self._num_patches

    def get_patch_embeddings(
        self,
        images: torch.Tensor,
        normalize: bool = True,
        return_cls: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract patch embeddings in CLIP's 768-dim contrastive space

        Args:
            images: Input images [B, C, H, W] in pixel space [0, 1]
            normalize: Whether to normalize images (CLIP preprocessing)
            return_cls: Whether to return CLS token

        Returns:
            - Patch embeddings [B, 576, 768] in contrastive space
            - CLS token [B, 768] if return_cls=True, else None
        """
        # Ensure images are on the correct device
        images = images.to(self.device)

        # CLIP preprocessing: normalize
        if normalize:
            mean = torch.tensor([0.48145466, 0.45782750, 0.40821073]).view(1, 3, 1, 1).to(images.device)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(images.device)
            images_norm = (images - mean) / std
        else:
            images_norm = images

        # Convert to model dtype and ensure on correct device
        conv1_dtype = self.model.visual.conv1.weight.dtype
        images_norm = images_norm.to(dtype=conv1_dtype, device=self.device)

        # Extract vision features through CLIP's vision encoder
        with torch.set_grad_enabled(images.requires_grad):
            # Step 1: Conv1 and reshape
            x = self.model.visual.conv1(images_norm)
            x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, 1024, 576]
            x = x.permute(0, 2, 1)  # [B, 576, 1024]

            # Step 2: Add CLS token and positional embedding
            x = torch.cat([
                self.model.visual.class_embedding.to(x.dtype).to(x.device) +
                torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x
            ], dim=1)  # [B, 577, 1024]
            x = x + self.model.visual.positional_embedding.to(x.dtype).to(x.device)

            # Step 3: Pre-layer norm
            x = self.model.visual.ln_pre(x)

            # Step 4: Transformer blocks
            x = x.permute(1, 0, 2)  # [577, B, 1024]
            x = self.model.visual.transformer(x)
            x = x.permute(1, 0, 2)  # [B, 577, 1024]

            # Step 5: Post-layer norm
            x = self.model.visual.ln_post(x)

            # Step 6: Project to contrastive space (1024 → 768)
            if self.model.visual.proj is not None:
                # Apply projection to all tokens (CLS + patches)
                x = x @ self.model.visual.proj  # [B, 577, 768]

            # Split CLS and patches
            cls_token = x[:, 0, :]  # [B, 768]
            patch_embeds = x[:, 1:, :]  # [B, 576, 768]

        if return_cls:
            return patch_embeds.float(), cls_token.float()
        else:
            return patch_embeds.float(), None

    def encode_text(self, texts: list) -> torch.Tensor:
        """
        Encode text in CLIP's 768-dim contrastive space

        Args:
            texts: List of text strings

        Returns:
            Text embeddings [num_texts, 768] in contrastive space
        """
        text_tokens = clip.tokenize(texts).to(self.device)

        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens).float()
            # Normalize as CLIP does
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features


if __name__ == "__main__":
    print("Testing CLIPContrastiveEncoder...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = CLIPContrastiveEncoder(model_name="ViT-L/14@336px", device=device)

    # Test vision encoding
    dummy_image = torch.rand(1, 3, 336, 336).to(device)
    patch_embeds, cls_token = encoder.get_patch_embeddings(dummy_image, normalize=True, return_cls=True)

    print(f"\nVision encoding:")
    print(f"  Patch embeddings: {patch_embeds.shape}")  # [1, 576, 768]
    print(f"  CLS token: {cls_token.shape}")  # [1, 768]

    # Test text encoding
    text_embeds = encoder.encode_text(["a pedestrian", "traffic light"])
    print(f"\nText encoding:")
    print(f"  Text embeddings: {text_embeds.shape}")  # [2, 768]

    # Test cosine similarity
    patch_norm = patch_embeds / patch_embeds.norm(dim=-1, keepdim=True)
    text_norm = text_embeds[0:1] / text_embeds[0:1].norm(dim=-1, keepdim=True)

    similarity = (patch_norm @ text_norm.T).squeeze()  # [576]
    print(f"\nCosine similarity test:")
    print(f"  Similarity shape: {similarity.shape}")  # [576]
    print(f"  Mean: {similarity.mean().item():.4f}")
    print(f"  Range: [{similarity.min().item():.4f}, {similarity.max().item():.4f}]")

    print("\n✓ CLIPContrastiveEncoder working correctly!")
