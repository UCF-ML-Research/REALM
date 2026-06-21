"""AnyAttack: single-forward-pass perturbation via a learned Decoder."""

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
    epsilon: float = 16.0 / 255.0
    decoder_checkpoint: str = "coco_bi"
    target_images_dir: str = ""


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

        self._clip_encoder = CLIPEncoder(model_name="ViT-B/32")
        self._clip_encoder = self._clip_encoder.to(device).eval()

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

        checkpoint = torch.load(ckpt_path, map_location="cpu")

        # Determine embed_dim from checkpoint FC layer.
        decoder_state = checkpoint.get("decoder_state_dict", checkpoint)
        fc_weight_key = "fc.0.weight"
        if fc_weight_key in decoder_state:
            embed_dim = decoder_state[fc_weight_key].shape[1]
        else:
            embed_dim = 512

        self._decoder = Decoder(embed_dim=embed_dim)

        # Strip DDP "module." prefix if present.
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

        tgt_path = Path(tgt_dir) / f"{sample.id}.jpg"
        if not tgt_path.exists():
            tgt_path = Path(tgt_dir) / f"{sample.id}.png"
        if not tgt_path.exists():
            raise FileNotFoundError(f"Target image not found: {tgt_dir}/{sample.id}.*")

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        img = Image.open(tgt_path).convert("RGB")
        return transform(img).unsqueeze(0)

    def generate(self, model, sample: Sample, **kwargs) -> AttackResult:
        self._initialize_models()

        device = self.config.device
        eps = self.config.epsilon

        clean_pil = sample.images[0]
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        clean_tensor = transform(clean_pil).unsqueeze(0).to(device)

        target_tensor = self._load_target_image(sample).to(device)

        with torch.no_grad():
            target_features = self._clip_encoder.encode_img(target_tensor)
            noise = self._decoder(target_features)

            noise = noise.clamp(-eps, eps)

            adv_tensor = (clean_tensor + noise).clamp(0, 1)

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
