"""PA-Attack: Prototype and Attention-guided adversarial attack on CLIP ViT."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from ..base_attack import AttackConfig, AttackResult, BaseAttack

_PROTO_DIR = Path(__file__).parent / "prototypes"


@dataclass
class PAAttackConfig(AttackConfig):
    """PA-Attack configuration.

    Epsilon and stepsize are in [0, 1] scale (e.g. 4/255 ~= 0.0157).
    """
    epsilon: float = 8.0 / 255.0
    max_iterations: int = 100
    stepsize: float = 1.0 / 255.0
    momentum: float = 0.9
    attn_layer: int = 12
    attn_temp: float = 20.0
    phase1_iters: int = 50
    prototype_path: Optional[str] = None
    clip_model_name: str = "ViT-L-14"
    clip_pretrained: str = "openai"
    output_normalize: bool = False
    input_res: int = 224


class PAAttack(BaseAttack):
    """Gray-box untargeted attack using CLIP ViT attention + OOD prototypes."""

    def __init__(self, config: PAAttackConfig):
        super().__init__(config)
        self._clip_model = None
        self._proto_tokens = None
        self._preprocess = transforms.Compose([
            transforms.Resize(config.input_res, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(config.input_res),
            transforms.ToTensor(),
        ])

    def _initialize_models(self):
        """Lazy-load CLIP ViT and prototypes."""
        if self._clip_model is not None:
            return

        import open_clip
        from .core.clip_vision import ClipVisionModel

        cfg = self.config
        model, _, preprocess = open_clip.create_model_and_transforms(
            cfg.clip_model_name, pretrained=cfg.clip_pretrained
        )
        model = model.visual
        normalize = preprocess.transforms[-1]  # Normalize transform
        self._clip_model = ClipVisionModel(model, normalize).to(cfg.device).eval()

        # Load prototypes
        proto_path = cfg.prototype_path
        if proto_path is None:
            proto_path = str(_PROTO_DIR / "prototypes_tokens_3000_20_1024.pt")
        self._proto_tokens = torch.load(proto_path, map_location=cfg.device, weights_only=True)

    def _prepare_image(self, pil_img: Image.Image) -> torch.Tensor:
        """Resize + CenterCrop + ToTensor -> [1, 3, H, W] in [0, 1]."""
        tensor = self._preprocess(pil_img)
        return tensor.unsqueeze(0).to(self.config.device)

    def generate(self, model, sample, **kwargs) -> AttackResult:
        self._initialize_models()
        cfg = self.config

        from .core.pgd import pgd_veattack
        from .core.loss import ComputeLossWrapper

        pil_img = sample.images[0]
        img_tensor = self._prepare_image(pil_img)

        # --- Extract clean tokens + attention ---
        with torch.no_grad():
            embedding_orig, tokens_orig, attentions = self._clip_model(
                vision=img_tensor, output_normalize=cfg.output_normalize,
                tokens=True, attention=True,
            )
            # CLS-to-patch attention from target layer, averaged over heads
            cls2patch = attentions[cfg.attn_layer][:, :, 0, 1:].mean(dim=1)
            tokens_mask = F.softmax(cfg.attn_temp * cls2patch, dim=-1)

        # --- Select most dissimilar prototype ---
        tokens_norm = F.normalize(tokens_orig, dim=-1)
        proto_tokens_norm = F.normalize(self._proto_tokens, dim=-1)
        token_sims = torch.einsum('bld,nld->bnl', tokens_norm, proto_tokens_norm)
        emb_similarity = token_sims.mean(dim=-1)
        min_sim, min_idx = torch.min(emb_similarity, dim=1)
        target_proto_tokens = self._proto_tokens[min_idx]

        # --- Phase 1: PGD with clean attention mask ---
        loss_fn = ComputeLossWrapper(
            embedding_orig, tokens_orig, target_proto_tokens, tokens_mask, 'none',
        )
        phase1_out = pgd_veattack(
            forward=self._clip_model,
            loss_fn=loss_fn,
            data_clean=img_tensor,
            norm='linf',
            eps=cfg.epsilon,
            iterations=cfg.phase1_iters,
            stepsize=cfg.stepsize,
            output_normalize=cfg.output_normalize,
            perturbation=torch.zeros_like(img_tensor).uniform_(-cfg.epsilon, cfg.epsilon).requires_grad_(True),
            mode='max',
            momentum=cfg.momentum,
        )

        # --- Phase 2: recompute attention from phase 1 output, fresh PGD ---
        with torch.no_grad():
            _, _, attentions2 = self._clip_model(
                vision=phase1_out, output_normalize=cfg.output_normalize,
                tokens=True, attention=True,
            )
            cls2patch2 = attentions2[cfg.attn_layer][:, :, 0, 1:].mean(dim=1)
            tokens_mask2 = F.softmax(cfg.attn_temp * cls2patch2, dim=-1)

        loss_fn2 = ComputeLossWrapper(
            embedding_orig, tokens_orig, target_proto_tokens, tokens_mask2, 'none',
        )
        phase2_out = pgd_veattack(
            forward=self._clip_model,
            loss_fn=loss_fn2,
            data_clean=phase1_out,
            norm='linf',
            eps=cfg.epsilon,
            iterations=cfg.max_iterations,
            stepsize=cfg.stepsize,
            output_normalize=cfg.output_normalize,
            perturbation=torch.zeros_like(phase1_out).uniform_(-cfg.epsilon, cfg.epsilon).requires_grad_(True),
            mode='max',
            momentum=cfg.momentum,
        )

        # --- Convert back to PIL ---
        adv_tensor = phase2_out.squeeze(0).clamp(0, 1).cpu()
        # Resize back to original size
        orig_w, orig_h = pil_img.size
        adv_pil = transforms.ToPILImage()(adv_tensor)
        if (adv_pil.size[0] != orig_w) or (adv_pil.size[1] != orig_h):
            adv_pil = adv_pil.resize((orig_w, orig_h), Image.BICUBIC)

        # Perturbation norm (per-phase is within epsilon; total from original may exceed)
        pert_norm = (phase2_out - img_tensor).abs().max().item()

        return AttackResult(
            success=True,
            adversarial_sample=adv_pil,
            original_output="",
            adversarial_output="",
            perturbation_norm=pert_norm,
            metadata={
                "min_prototype_similarity": min_sim.item(),
                "prototype_index": min_idx.item(),
                "phase1_iters": cfg.phase1_iters,
                "phase2_iters": cfg.max_iterations,
            },
        )

    def is_gradient_based(self) -> bool:
        return True
