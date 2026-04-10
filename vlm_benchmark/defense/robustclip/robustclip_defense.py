"""RobustCLIP Defense: FARE + LEAF encoder swap for adversarial robustness.

This defense replaces the standard CLIP vision/text encoder with adversarially
fine-tuned versions (FARE for vision, LEAF for text).  It operates in the
**embedding space** — the `clean()` method re-encodes an adversarial image
through the robust encoder and reconstructs a denoised image via the inverse
CLIP visual pipeline, yielding a cleaned image whose CLIP features are closer
to the true clean features.

For VLM-level integration (encoder swap without image reconstruction), use
`get_robust_model()` to get the loaded CLIPModel and pass it directly to
the VLM evaluation pipeline.

References:
  - FARE: Schlarmann et al., "Robust CLIP", ICML 2024
  - LEAF: Schlarmann et al., "CLIP Needs a Robust Text Encoder", NeurIPS 2025
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from ..base_defense import BaseDefense, DefenseConfig, DefenseResult


@dataclass
class RobustCLIPDefenseConfig(DefenseConfig):
    """RobustCLIP-specific configuration."""

    # Model selection
    clip_model_name: str = "ViT-L-14"
    hf_model_id: str = "LEAF-CLIP/CLIP-ViT-L-rho50-k1-constrained-FARE2"
    local_checkpoint: Optional[str] = None

    # Encoder mode: "vision", "text", "both"
    encoder_mode: str = "both"

    # Device
    device: str = "cuda:0"


# CLIP ImageNet normalization constants
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class RobustCLIPDefense(BaseDefense):
    """RobustCLIP defense via FARE + LEAF encoder swap.

    Two usage modes:

    1. **Image cleaning** (``clean()``):
       Passes the adversarial image through the robust CLIP vision encoder,
       then reconstructs a cleaned image via pixel optimization that matches
       the robust embedding through the original encoder.

    2. **Encoder swap** (``get_robust_model()``):
       Returns the loaded HuggingFace CLIPModel / OpenCLIP model so the
       evaluation pipeline can inject it directly into the VLM, bypassing
       image reconstruction entirely.  This is the recommended mode for
       benchmark evaluation.
    """

    def __init__(self, config: RobustCLIPDefenseConfig):
        super().__init__(config)
        self.config: RobustCLIPDefenseConfig = config

        # Lazy-loaded models
        self._robust_model = None       # CLIPModel (HF) or OpenCLIP model
        self._processor = None          # CLIPProcessor / image transform
        self._original_model = None     # frozen original CLIP for reference embeddings
        self._is_hf = False             # True if loaded via HuggingFace
        self._input_res = 224           # CLIP input resolution (set during model load)

    def requires_model(self) -> bool:
        return False

    # ── Model loading ────────────────────────────────────────────────────

    def _initialize_models(self):
        """Lazy load robust CLIP and frozen original CLIP."""
        if self._robust_model is not None:
            return

        device = self.config.device

        if self.config.local_checkpoint:
            self._load_openclip(device)
        else:
            self._load_hf(device)

    def _load_hf(self, device: str):
        """Load LEAF+FARE checkpoint from HuggingFace (recommended)."""
        from transformers import CLIPModel, CLIPProcessor

        hf_id = self.config.hf_model_id
        proc_id = _hf_processor_id(self.config.clip_model_name)
        mode = self.config.encoder_mode

        print(f"Loading robust CLIP from HuggingFace: {hf_id} (mode={mode})")

        robust = CLIPModel.from_pretrained(hf_id).to(device).eval()
        original = CLIPModel.from_pretrained(proc_id).to(device).eval()

        # Apply encoder_mode: selectively swap back to original weights
        if mode == "vision":
            # Keep robust vision, restore original text
            robust.text_model.load_state_dict(original.text_model.state_dict())
            robust.text_projection.load_state_dict(original.text_projection.state_dict())
        elif mode == "text":
            # Keep robust text, restore original vision
            robust.vision_model.load_state_dict(original.vision_model.state_dict())
            robust.visual_projection.load_state_dict(original.visual_projection.state_dict())
        # mode == "both": keep everything from robust checkpoint

        for p in robust.parameters():
            p.requires_grad_(False)
        for p in original.parameters():
            p.requires_grad_(False)

        self._robust_model = robust
        self._original_model = original
        self._processor = CLIPProcessor.from_pretrained(proc_id)
        self._is_hf = True
        # Extract input resolution from processor config
        self._input_res = getattr(
            self._processor.image_processor, "size", {}
        ).get("shortest_edge", 224)

        print("RobustCLIP models loaded (HuggingFace)")

    def _load_openclip(self, device: str):
        """Load FARE vision-only .pt checkpoint via OpenCLIP."""
        import open_clip

        arch = self.config.clip_model_name
        ckpt = self.config.local_checkpoint

        if not Path(ckpt).exists():
            raise FileNotFoundError(f"FARE checkpoint not found: {ckpt}")

        print(f"Loading OpenCLIP {arch} with FARE checkpoint: {ckpt}")

        # Robust model — load base then swap vision weights
        model, _, preprocess = open_clip.create_model_and_transforms(
            arch, pretrained="openai",
        )
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        model.visual.load_state_dict(state)
        model = model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self._robust_model = model
        self._processor = preprocess

        # Original model for reference embeddings
        orig, _, _ = open_clip.create_model_and_transforms(
            arch, pretrained="openai",
        )
        orig = orig.to(device).eval()
        for p in orig.parameters():
            p.requires_grad_(False)
        self._original_model = orig
        self._is_hf = False
        # Extract input resolution from the vision encoder
        self._input_res = model.visual.image_size
        if isinstance(self._input_res, (tuple, list)):
            self._input_res = self._input_res[0]

        print("RobustCLIP models loaded (OpenCLIP)")

    # ── Public API ───────────────────────────────────────────────────────

    def get_robust_model(self):
        """Return the loaded robust CLIP model for direct VLM encoder swap.

        Returns:
            (robust_model, processor)
        """
        self._initialize_models()
        return self._robust_model, self._processor

    def get_robust_vision_state_dict(self):
        """Return the robust vision encoder state_dict for direct injection.

        Useful when only the vision encoder needs swapping (e.g. LLaVA).
        """
        self._initialize_models()
        if hasattr(self._robust_model, "visual"):
            return self._robust_model.visual.state_dict()
        else:
            return self._robust_model.vision_model.state_dict()

    def clean(self, image_path: str, **kwargs) -> DefenseResult:
        """Clean an adversarial image via robust CLIP embedding guidance.

        Strategy: optimize in raw [0,1] pixel space so the resulting image's
        CLIP embedding (through the *original* encoder) matches the robust
        encoder's embedding of the adversarial input.

        Args:
            image_path: path to adversarial image
            **kwargs: unused

        Returns:
            DefenseResult with cleaned PIL image
        """
        self._initialize_models()

        device = self.config.device
        pil_img = Image.open(image_path).convert("RGB")
        original_size = pil_img.size

        # Encode adversarial image through the ROBUST encoder → target embedding
        robust_emb = self._encode_image(pil_img, device)

        # Pixel-space optimization: find image whose ORIGINAL encoder
        # embedding matches the robust embedding
        cleaned_tensor = self._reconstruct(pil_img, robust_emb, device)

        # Convert [0,1] tensor back to PIL
        cleaned_np = (
            cleaned_tensor.squeeze(0)
            .clamp(0, 1)
            .mul(255)
            .byte()
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
        cleaned_pil = Image.fromarray(cleaned_np).resize(
            original_size, Image.LANCZOS,
        )

        return DefenseResult(
            cleaned_sample=cleaned_pil,
            original_image_path=image_path,
            detection_confidence=0.0,
            regions_removed=0,
            metadata={
                "hf_model_id": self.config.hf_model_id,
                "encoder_mode": self.config.encoder_mode,
                "clip_model_name": self.config.clip_model_name,
                "purification_method": "robustclip",
            },
        )

    # ── Internal helpers ─────────────────────────────────────────────────

    def _encode_image(self, pil_img: Image.Image, device: str):
        """Encode a PIL image through the robust CLIP vision encoder."""
        if self._is_hf:
            inputs = self._processor(images=pil_img, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                return self._robust_model.get_image_features(**inputs)
        else:
            x = self._processor(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                return self._robust_model.encode_image(x)

    def _pil_to_tensor(self, pil_img: Image.Image, device: str) -> torch.Tensor:
        """Convert PIL image to [1, C, H, W] tensor in [0, 1] at CLIP resolution."""
        res = self._input_res
        transform = transforms.Compose([
            transforms.Resize(res, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(res),
            transforms.ToTensor(),
        ])
        return transform(pil_img).unsqueeze(0).to(device)

    @staticmethod
    def _normalize_clip(x: torch.Tensor) -> torch.Tensor:
        """Apply CLIP ImageNet normalization to [0,1] tensor."""
        mean = torch.tensor(_CLIP_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor(_CLIP_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        return (x - mean) / std

    def _encode_raw(self, model, x_raw: torch.Tensor) -> torch.Tensor:
        """Encode a [0,1] raw tensor through a CLIP vision encoder.

        Handles normalization internally so the optimization loop can
        operate entirely in [0,1] pixel space.
        """
        x_norm = self._normalize_clip(x_raw)
        if self._is_hf:
            return model.get_image_features(pixel_values=x_norm)
        else:
            return model.encode_image(x_norm)

    def _reconstruct(
        self,
        pil_img: Image.Image,
        target_emb: torch.Tensor,
        device: str,
        steps: int = 200,
        lr: float = 0.01,
    ) -> torch.Tensor:
        """Reconstruct a clean image via pixel-space optimization.

        Minimizes  || f_orig(normalize(x')) - target_emb ||^2  starting from
        the adversarial image, keeping pixel values in [0, 1].

        Returns:
            [1, C, H, W] tensor in [0, 1]
        """
        # Start from the adversarial image in raw [0, 1] space
        x = self._pil_to_tensor(pil_img, device)
        x = x.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([x], lr=lr)
        target_emb = target_emb.detach()

        for _ in range(steps):
            optimizer.zero_grad()
            emb = self._encode_raw(self._original_model, x)
            # Match FARE/LEAF loss: sum over embedding dim, mean over batch
            loss = F.mse_loss(emb, target_emb, reduction="none").sum(dim=-1).mean()
            loss.backward()
            optimizer.step()

            # Clamp to valid [0, 1] pixel range
            with torch.no_grad():
                x.clamp_(0, 1)

        return x.detach()


# ── Utilities ────────────────────────────────────────────────────────────────

_HF_PROCESSOR_MAP = {
    "ViT-L-14": "openai/clip-vit-large-patch14",
    "ViT-H-14": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    "ViT-g-14": "laion/CLIP-ViT-g-14-laion2B-s12B-b42K",
    "ViT-bigG-14": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    "ViT-B-16": "openai/clip-vit-base-patch16",
    "ViT-B-32": "openai/clip-vit-base-patch32",
}


def _hf_processor_id(clip_model_name: str) -> str:
    """Map OpenCLIP model name → HuggingFace processor ID."""
    return _HF_PROCESSOR_MAP.get(clip_model_name, "openai/clip-vit-large-patch14")
