"""BlueSuffix defense configuration."""

from pathlib import Path

BASE_DIR = Path(__file__).parent

DEFAULT_CONFIG = {
    # Component toggles
    "enable_image_purifier": True,
    "enable_text_purifier": True,
    "enable_suffix_generator": True,

    # Diffusion denoising params
    "max_timesteps": "100",
    "num_denoising_steps": "20",
    "sampling_method": "ddim",
    "diffusion_checkpoint": None,       # auto-resolved from assets/

    # Suffix generator
    "suffix_generator_dir": None,       # auto-resolved from assets/suffix_generator/

    # Text purifier
    "openai_api_key": None,             # fallback to $OPENAI_API_KEY
    "text_purifier_model": "gpt-4o",

    # Infra
    "device": "cuda:0",
}

# Only BlueSuffix-unique CLI args. Shared diffusion args (max_timesteps,
# num_denoising_steps, sampling_method) are omitted to avoid argparse
# duplicate-flag errors — they are registered by FreqPure's
# CLI_ARGUMENTS. They are still listed in cli_params in registry.py so
# DefenseRegistry.create() picks them up correctly.
CLI_ARGUMENTS = [
    {
        "name": "--enable_image_purifier",
        "type": str,
        "default": None,
        "help": "Enable BlueSuffix image purifier (True/False, default: True)",
    },
    {
        "name": "--enable_text_purifier",
        "type": str,
        "default": None,
        "help": "Enable BlueSuffix text purifier (True/False, default: True)",
    },
    {
        "name": "--enable_suffix_generator",
        "type": str,
        "default": None,
        "help": "Enable BlueSuffix suffix generator (True/False, default: True)",
    },
    {
        "name": "--openai_api_key",
        "type": str,
        "default": None,
        "help": "OpenAI API key for text purifier (default: $OPENAI_API_KEY)",
    },
]


def get_default_config():
    """Get default BlueSuffix configuration."""
    return DEFAULT_CONFIG.copy()


def get_diffusion_checkpoint_path():
    """Get diffusion model checkpoint path."""
    checkpoint = BASE_DIR / "assets" / "256x256_diffusion_uncond.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"BlueSuffix diffusion checkpoint not found at {checkpoint}")
    return str(checkpoint)


def get_suffix_generator_path():
    """Get suffix generator directory path."""
    suffix_dir = BASE_DIR / "assets" / "suffix_generator"
    if not suffix_dir.exists():
        raise FileNotFoundError(f"Suffix generator not found at {suffix_dir}")
    return str(suffix_dir)
