"""BlueSuffix defense configuration."""

from pathlib import Path

BASE_DIR = Path(__file__).parent

DEFAULT_CONFIG = {
    "enable_image_purifier": True,
    "enable_text_purifier": True,
    "enable_suffix_generator": True,

    "max_timesteps": "100",
    "num_denoising_steps": "20",
    "sampling_method": "ddim",
    "diffusion_checkpoint": None,

    "suffix_generator_dir": None,

    "openai_api_key": None,
    "text_purifier_model": "gpt-4o",

    "device": "cuda:0",
}

# Only BlueSuffix-unique CLI args; shared diffusion args are registered by FreqPure to avoid argparse duplicates.
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
