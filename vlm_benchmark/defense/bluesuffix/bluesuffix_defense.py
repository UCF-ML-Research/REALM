"""BlueSuffix Defense Wrapper.

Three-step defense (ICLR 2025):
  1. Image Purifier  — Diffusion denoising (BlueSuffix's own diffusion core)
  2. Text Purifier   — GPT-4o prompt rewriting
  3. Suffix Generator — GPT-2 LoRA defensive suffix

Each component can be independently enabled/disabled.
"""

import os
from dataclasses import dataclass
from typing import Optional

import torch
from PIL import Image
from torchvision import transforms

from ..base_defense import BaseDefense, DefenseConfig, DefenseResult


@dataclass
class BlueSuffixDefenseConfig(DefenseConfig):
    """BlueSuffix-specific configuration."""
    # Component toggles
    enable_image_purifier: bool = True
    enable_text_purifier: bool = True
    enable_suffix_generator: bool = True

    # Diffusion denoising params
    max_timesteps: str = "100"
    num_denoising_steps: str = "20"
    sampling_method: str = "ddim"
    diffusion_checkpoint: Optional[str] = None

    # Suffix generator
    suffix_generator_dir: Optional[str] = None

    # Text purifier
    openai_api_key: Optional[str] = None
    text_purifier_model: str = "gpt-4o"

    # Device
    device: str = "cuda:0"

    def __post_init__(self):
        # Parse string booleans from CLI
        for field_name in ("enable_image_purifier", "enable_text_purifier", "enable_suffix_generator"):
            val = getattr(self, field_name)
            if isinstance(val, str):
                setattr(self, field_name, val.lower() not in ("false", "0", "no"))


class BlueSuffixDefense(BaseDefense):
    """BlueSuffix defense wrapper."""

    def __init__(self, config: BlueSuffixDefenseConfig):
        super().__init__(config)
        self.config: BlueSuffixDefenseConfig = config
        self._purification_forward = None
        self._suffix_model = None
        self._suffix_tokenizer = None

    def requires_model(self) -> bool:
        """BlueSuffix does not require a VLM."""
        return False

    def _initialize_image_purifier(self):
        """Lazy load BlueSuffix's diffusion model (own core copy)."""
        if self._purification_forward is not None:
            return

        from .core.load_diffusion_model import load_diffusion_models
        from .core.purification import PurificationForward
        from .core.purify import get_diffusion_params

        if self.config.diffusion_checkpoint is None:
            from .config import get_diffusion_checkpoint_path
            self.config.diffusion_checkpoint = get_diffusion_checkpoint_path()

        print("Loading BlueSuffix image purifier (diffusion denoising)...")

        diffusion = load_diffusion_models(
            self.config.diffusion_checkpoint,
            torch.device(self.config.device),
        )

        def_max_timesteps, def_diffusion_steps = get_diffusion_params(
            self.config.max_timesteps,
            self.config.num_denoising_steps,
        )

        print(f"  Max timesteps: {def_max_timesteps}")
        print(f"  Diffusion steps: {def_diffusion_steps}")
        print(f"  Sampling method: {self.config.sampling_method}")

        self._purification_forward = PurificationForward(
            diffusion,
            def_max_timesteps,
            def_diffusion_steps,
            self.config.sampling_method,
            is_imagenet=True,
            device=torch.device(self.config.device),
        )

        print("BlueSuffix image purifier loaded")

    def _initialize_suffix_generator(self):
        """Lazy load GPT-2 LoRA suffix generator."""
        if self._suffix_model is not None:
            return

        from .core.suffix_generator import load_suffix_generator
        from .config import get_suffix_generator_path

        suffix_dir = self.config.suffix_generator_dir
        if suffix_dir is None:
            suffix_dir = get_suffix_generator_path()

        print("Loading BlueSuffix suffix generator...")
        self._suffix_model, self._suffix_tokenizer = load_suffix_generator(
            suffix_dir, device=self.config.device,
        )
        print("BlueSuffix suffix generator loaded")

    def clean(self, image_path: str, **kwargs) -> DefenseResult:
        """Clean adversarial image using BlueSuffix three-step defense.

        Args:
            image_path: Path to adversarial image
            **kwargs:
                prompt: Text prompt to purify and append suffix to

        Returns:
            DefenseResult with cleaned image and metadata including final_prompt
        """
        prompt = kwargs.get("prompt")
        steps_applied = []

        # ── Step 1: Image Purifier ──
        purified_pil = None
        if self.config.enable_image_purifier:
            self._initialize_image_purifier()

            image = Image.open(image_path).convert("RGB")
            original_size = image.size
            transform = transforms.ToTensor()

            with torch.no_grad():
                x = transform(image).unsqueeze(0).to(torch.float32).to(self.config.device)
                purified_tensor = self._purification_forward(x)

                mean = torch.tensor([0.485, 0.456, 0.406], device=self.config.device)
                std = torch.tensor([0.229, 0.224, 0.225], device=self.config.device)
                vis = purified_tensor * std[:, None, None] + mean[:, None, None]
                vis = vis.permute(0, 2, 3, 1)
                vis = torch.clamp(vis, 0, 1)

                purified_np = vis[0].cpu().numpy()
                purified_pil = Image.fromarray((purified_np * 255).astype("uint8"))

            steps_applied.append("image_purifier")
        else:
            purified_pil = Image.open(image_path).convert("RGB")
            original_size = purified_pil.size

        # ── Step 2: Text Purifier ──
        purified_prompt = prompt
        if self.config.enable_text_purifier and prompt is not None:
            api_key = self.config.openai_api_key or os.environ.get("OPENAI_API_KEY")
            if api_key:
                from .core.text_purifier import purify_text
                try:
                    purified_prompt = purify_text(
                        prompt, api_key, model=self.config.text_purifier_model,
                    )
                    steps_applied.append("text_purifier")
                except Exception as e:
                    print(f"[BlueSuffix] Text purifier failed, using original prompt: {e}")
                    purified_prompt = prompt
            # else: silently skip — no API key

        # ── Step 3: Suffix Generator ──
        suffix = ""
        if self.config.enable_suffix_generator and prompt is not None:
            self._initialize_suffix_generator()
            from .core.suffix_generator import generate_suffix
            target_prompt = purified_prompt if purified_prompt is not None else prompt
            suffix = generate_suffix(
                self._suffix_model, self._suffix_tokenizer,
                target_prompt, device=self.config.device,
            )
            steps_applied.append("suffix_generator")

        # Build final prompt
        final_prompt = prompt
        if prompt is not None:
            base = purified_prompt if purified_prompt is not None else prompt
            final_prompt = (base + " " + suffix).strip() if suffix else base

        return DefenseResult(
            cleaned_sample=purified_pil,
            original_image_path=image_path,
            detection_confidence=0.0,
            regions_removed=0,
            metadata={
                "original_prompt": prompt,
                "purified_prompt": purified_prompt,
                "suffix": suffix,
                "final_prompt": final_prompt,
                "steps_applied": steps_applied,
                "enable_image_purifier": self.config.enable_image_purifier,
                "enable_text_purifier": self.config.enable_text_purifier,
                "enable_suffix_generator": self.config.enable_suffix_generator,
                "max_timesteps": self.config.max_timesteps,
                "num_denoising_steps": self.config.num_denoising_steps,
                "sampling_method": self.config.sampling_method,
                "purification_method": "bluesuffix",
            },
        )
