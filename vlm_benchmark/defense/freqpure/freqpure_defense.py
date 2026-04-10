"""FreqPure Defense Wrapper."""

from dataclasses import dataclass
from typing import Optional

from ..base_defense import BaseDefense, DefenseConfig, DefenseResult


@dataclass
class FreqPureDefenseConfig(DefenseConfig):
    """FreqPure-specific configuration."""
    amplitude_cut_range: int = 10
    phase_cut_range: int = 10
    delta: float = 0.3
    forward_noise_steps: int = 50
    max_timesteps: str = "50,50,50,50,50,50,50,50"
    num_denoising_steps: str = "5,5,5,5,5,5,5,5"
    sampling_method: str = "ddpm"
    diffusion_checkpoint: Optional[str] = None
    device: str = "cuda:0"


class FreqPureDefense(BaseDefense):
    """FreqPure defense wrapper."""

    def __init__(self, config: FreqPureDefenseConfig):
        super().__init__(config)
        self.config: FreqPureDefenseConfig = config
        self._purifier = None     # lazy — PurificationForward instance

    def requires_model(self) -> bool:
        return False

    def _initialize_models(self):
        if self._purifier is not None:
            return

        import torch
        from .core.load_model import load_models
        from .core.purification import PurificationForward
        from .core.purify import get_diffusion_params

        if self.config.diffusion_checkpoint is None:
            from .config import get_diffusion_checkpoint_path
            self.config.diffusion_checkpoint = get_diffusion_checkpoint_path()

        print("Loading FreqPure diffusion model...")

        diffusion = load_models(
            self.config.diffusion_checkpoint,
            torch.device(self.config.device)
        )

        max_timestep_list, diffusion_steps = get_diffusion_params(
            self.config.max_timesteps, self.config.num_denoising_steps
        )

        self._purifier = PurificationForward(
            diffusion=diffusion,
            max_timestep=max_timestep_list,
            attack_steps=diffusion_steps,
            sampling_method=self.config.sampling_method,
            is_imagenet=True,
            device=torch.device(self.config.device),
            amplitude_cut_range=self.config.amplitude_cut_range,
            phase_cut_range=self.config.phase_cut_range,
            delta=self.config.delta,
            forward_noise_steps=self.config.forward_noise_steps,
        )

        print("FreqPure model loaded successfully")

    def clean(self, image_path: str, **kwargs) -> DefenseResult:
        import torch
        import torch.nn.functional as F
        from PIL import Image
        from torchvision import transforms
        from .core.utils import clf2diff, diff2clf

        self._initialize_models()

        image = Image.open(image_path).convert('RGB')
        original_size = image.size  # (W, H)

        x = transforms.ToTensor()(image).unsqueeze(0).to(self.config.device)

        with torch.no_grad():
            x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
            x_diff = clf2diff(x)
            for i in range(len(self._purifier.max_timestep)):
                noised_x = self._purifier.get_noised_x(x_diff, self._purifier.max_timestep[i])
                x_diff = self._purifier.denoising_process(x_diff, noised_x, self._purifier.attack_steps[i])
            purified = diff2clf(x_diff).squeeze(0).clamp(0, 1)

        purified_np = (purified.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
        purified_pil = Image.fromarray(purified_np)

        return DefenseResult(
            cleaned_sample=purified_pil,
            original_image_path=image_path,
            detection_confidence=0.0,
            regions_removed=0,
            metadata={
                "output_size": "256x256",
                "original_size": f"{original_size[0]}x{original_size[1]}",
                "amplitude_cut_range": self.config.amplitude_cut_range,
                "phase_cut_range": self.config.phase_cut_range,
                "delta": self.config.delta,
                "forward_noise_steps": self.config.forward_noise_steps,
                "max_timesteps": self.config.max_timesteps,
                "num_denoising_steps": self.config.num_denoising_steps,
                "sampling_method": self.config.sampling_method,
                "purification_method": "freqpure",
            }
        )
