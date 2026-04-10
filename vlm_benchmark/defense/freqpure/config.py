from pathlib import Path

BASE_DIR = Path(__file__).parent

DEFAULT_CONFIG = {
    # FreqPure frequency-domain params
    "amplitude_cut_range": 10,
    "phase_cut_range": 10,
    "delta": 0.3,
    "forward_noise_steps": 50,
    # Diffusion schedule (same meaning as BlueSuffix)
    "max_timesteps": "50,50,50,50,50,50,50,50",
    "num_denoising_steps": "5,5,5,5,5,5,5,5",
    "sampling_method": "ddpm",
    # Infra
    "diffusion_checkpoint": None,    # auto-resolved from assets/
    "device": "cuda:0",
}

CLI_ARGUMENTS = [
    {"name": "--amplitude_cut_range", "type": int, "default": None,
     "help": "FreqPure low-freq amplitude swap radius (default: 10)"},
    {"name": "--phase_cut_range", "type": int, "default": None,
     "help": "FreqPure low-freq phase clip radius (default: 10)"},
    {"name": "--delta", "type": float, "default": None,
     "help": "FreqPure max phase deviation (default: 0.3)"},
    {"name": "--forward_noise_steps", "type": int, "default": None,
     "help": "FreqPure reference noise timestep (default: 50)"},
    # Shared diffusion params (also used by BlueSuffix — registered here once to
    # avoid argparse duplicate-flag errors; omitted from bluesuffix/config.py)
    {"name": "--max_timesteps", "type": str, "default": None,
     "help": "Max diffusion timesteps, comma-separated (default varies by defense)"},
    {"name": "--num_denoising_steps", "type": str, "default": None,
     "help": "Number of denoising steps, comma-separated (default varies by defense)"},
    {"name": "--sampling_method", "type": str, "default": None,
     "choices": ["ddim", "ddpm"],
     "help": "Diffusion sampling method: ddim or ddpm"},
]

def get_default_config():
    return DEFAULT_CONFIG.copy()

def get_diffusion_checkpoint_path():
    checkpoint = BASE_DIR / "assets" / "pretrained" / "guided_diffusion" / "256x256_diffusion_uncond.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"FreqPure checkpoint not found at {checkpoint}")
    return str(checkpoint)
