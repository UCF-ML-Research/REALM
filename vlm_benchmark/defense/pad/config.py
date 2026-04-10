"""PAD defense configuration."""

from pathlib import Path

BASE_DIR = Path(__file__).parent

DEFAULT_CONFIG = {
    "iou_threshold": 0.5,
    "ratio_mi": 0.5,
    "kernel_param": 80,
    "thresh_param": 80,
    "sam_model_type": "vit_l",
    "sam_checkpoint": None,  # Auto-set to pad/assets/models path
    "device": "cuda:0",
}

# CLI argument specifications for PAD defense
CLI_ARGUMENTS = [
    {
        "name": "--iou_threshold",
        "type": float,
        "default": None,
        "help": "IOU threshold for PAD (default: 0.5)"
    },
    {
        "name": "--ratio_mi",
        "type": float,
        "default": None,
        "help": "Ratio MI parameter for PAD (default: 0.5)"
    },
    {
        "name": "--kernel_param",
        "type": int,
        "default": None,
        "help": "Kernel parameter for PAD (default: 80)"
    },
    {
        "name": "--thresh_param",
        "type": int,
        "default": None,
        "help": "Threshold parameter for PAD (default: 80)"
    },
]


def get_default_config():
    """Get default PAD configuration."""
    return DEFAULT_CONFIG.copy()


def get_sam_checkpoint_path():
    """Get SAM checkpoint path under PAD assets."""
    sam_checkpoint = BASE_DIR / "assets" / "models" / "sam_vit_l_0b3195.pth"

    if not sam_checkpoint.exists():
        raise FileNotFoundError(
            f"SAM checkpoint not found at {sam_checkpoint}. "
            f"Place the checkpoint under pad/assets/models or pass --sam_checkpoint."
        )

    return str(sam_checkpoint)
