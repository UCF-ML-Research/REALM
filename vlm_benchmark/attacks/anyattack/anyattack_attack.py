"""AnyAttack: single-forward-pass perturbation via learned Decoder.

Flow: target_image → CLIP.encode_img() → Decoder() → noise → clamp(±ε) → clean + noise → adversarial

No iterative optimization at inference time — the Decoder was trained offline.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image

from ..base_attack import AttackConfig, AttackResult, BaseAttack
from ...data.base_dataset import Sample

_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")


@dataclass
class AnyAttackConfig(AttackConfig):
    epsilon: float = 16.0 / 255.0       # perturbation bound in [0,1] range
    decoder_checkpoint: str = "coco_bi"  # checkpoint name or full path
    target_images_dir: str = ""          # flat dir with {index}.jpg targets


class AnyAttack(BaseAttack):
    """AnyAttack: single-forward-pass perturbation via learned Decoder."""

    def __init__(self, config: AnyAttackConfig):
        super().__init__(config)
        self.config: AnyAttackConfig = config
        self._clip_encoder = None
        self._decoder = None

    def _initialize_models(self) -> None:
        if self._clip_encoder is not None:
            return

        from .core.model import CLIPEncoder, Decoder

        device = self.config.device

        # Load CLIP encoder
        self._clip_encoder = CLIPEncoder(model_name="ViT-B/32")
        self._clip_encoder = self._clip_encoder.to(device).eval()

        # Resolve checkpoint path
        ckpt_name = self.config.decoder_checkpoint
        if os.path.isfile(ckpt_name):
            ckpt_path = ckpt_name
        else:
            ckpt_path = os.path.join(_ASSETS_DIR, "checkpoints", f"{ckpt_name}.pt")

        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"Decoder checkpoint not found: {ckpt_path}\n"
                f"Available: {os.listdir(os.path.join(_ASSETS_DIR, 'checkpoints'))}"
            )

        # Load decoder
        # CLIP ViT-B/32 embed_dim = 512; legacy training uses embed_dim=512.
        # Auto-detect from checkpoint FC layer below.
        checkpoint = torch.load(ckpt_path, map_location="cpu")

        # Determine embed_dim from checkpoint FC layer
        decoder_state = checkpoint.get("decoder_state_dict", checkpoint)
        fc_weight_key = "fc.0.weight"
        if fc_weight_key in decoder_state:
            embed_dim = decoder_state[fc_weight_key].shape[1]
        else:
            embed_dim = 512  # default for CLIP ViT-B/32 checkpoints

        self._decoder = Decoder(embed_dim=embed_dim)

        # Strip DDP "module." prefix if present
        cleaned_state = {}
        for k, v in decoder_state.items():
            cleaned_state[k.removeprefix("module.")] = v
        self._decoder.load_state_dict(cleaned_state)

        self._decoder = self._decoder.to(device).eval()

    def _load_target_image(self, sample: Sample) -> torch.Tensor:
        """Load target image matching sample.id from target_images_dir."""
        tgt_dir = self.config.target_images_dir
        if not tgt_dir:
            raise ValueError("target_images_dir not set in AnyAttackConfig")

        # Stem-matched lookup: {sample.id}.jpg
        tgt_path = Path(tgt_dir) / f"{sample.id}.jpg"
        if not tgt_path.exists():
            # Try .png fallback
            tgt_path = Path(tgt_dir) / f"{sample.id}.png"
        if not tgt_path.exists():
            raise FileNotFoundError(f"Target image not found: {tgt_dir}/{sample.id}.*")

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),  # [0, 1]
        ])
        img = Image.open(tgt_path).convert("RGB")
        return transform(img).unsqueeze(0)  # [1, 3, 224, 224]

    def generate(self, model, sample: Sample, **kwargs) -> AttackResult:
        self._initialize_models()

        device = self.config.device
        eps = self.config.epsilon

        # 1. Prepare clean image: PIL → Resize(256) → CenterCrop(224) → tensor [0,1]
        clean_pil = sample.images[0]
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        clean_tensor = transform(clean_pil).unsqueeze(0).to(device)  # [1, 3, 224, 224]

        # 2. Load and encode target image
        target_tensor = self._load_target_image(sample).to(device)

        with torch.no_grad():
            # 3. target → CLIP encode → Decoder → noise
            target_features = self._clip_encoder.encode_img(target_tensor)
            noise = self._decoder(target_features)

            # 4. Clamp noise to ±epsilon
            noise = noise.clamp(-eps, eps)

            # 5. Apply perturbation
            adv_tensor = (clean_tensor + noise).clamp(0, 1)

        # 6. tensor → PIL
        adv_arr = adv_tensor.squeeze(0).cpu().mul(255).byte().permute(1, 2, 0).numpy()
        adv_pil = Image.fromarray(adv_arr)

        perturbation_norm = noise.abs().max().item()

        return AttackResult(
            success=True,
            adversarial_sample=adv_pil,
            original_output="",
            adversarial_output="",
            perturbation_norm=perturbation_norm,
            metadata={
                "epsilon": eps,
                "decoder_checkpoint": self.config.decoder_checkpoint,
            },
        )

    def is_gradient_based(self) -> bool:
        return False
