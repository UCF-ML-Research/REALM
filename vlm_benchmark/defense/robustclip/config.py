"""RobustCLIP defense configuration."""

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    # Model selection
    "clip_model_name": "ViT-L-14",
    "hf_model_id": "LEAF-CLIP/CLIP-ViT-L-rho50-k1-constrained-FARE2",
    "local_checkpoint": None,      # optional: path to a local .pt file (FARE vision-only)

    # Encoder mode: "vision", "text", "both"
    #   vision = FARE only (swap vision encoder)
    #   text   = LEAF only (swap text encoder)
    #   both   = FARE + LEAF (recommended)
    "encoder_mode": "both",

    # Device
    "device": "cuda:0",
}

# ── CLI arguments ─────────────────────────────────────────────────────────────

CLI_ARGUMENTS = [
    {
        "name": "--clip_model_name",
        "type": str,
        "default": None,
        "help": "OpenCLIP model architecture (default: ViT-L-14)",
    },
    {
        "name": "--hf_model_id",
        "type": str,
        "default": None,
        "help": (
            "HuggingFace model ID for LEAF+FARE checkpoint "
            "(default: LEAF-CLIP/CLIP-ViT-L-rho50-k1-constrained-FARE2)"
        ),
    },
    {
        "name": "--local_checkpoint",
        "type": str,
        "default": None,
        "help": "Path to local FARE .pt checkpoint (overrides hf_model_id for vision)",
    },
    {
        "name": "--encoder_mode",
        "type": str,
        "default": None,
        "help": "Which encoders to robustify: vision | text | both (default: both)",
    },
]

# ── Training defaults (used by train_fare.py / train_leaf.py) ─────────────────

FARE_TRAIN_DEFAULTS = {
    "epochs": 2,
    "batch_size": 128,
    "lr": 1e-5,
    "weight_decay": 1e-4,
    "warmup_steps": 1400,      # 7% of 20k steps (matches official: github.com/chs20/RobustVLM)
    "epsilon": 2 / 255,        # L-inf bound in [0, 1] space
    "pgd_steps": 10,
    "pgd_alpha": 1 / 255,      # PGD step size
    "clip_model_name": "ViT-L-14",
    "pretrained": "openai",     # starting checkpoint
}

FARE_QWEN_TRAIN_DEFAULTS = {
    "epochs": 2,
    "batch_size": 64,
    "lr": 1e-5,
    "weight_decay": 1e-4,
    "warmup_steps": 5000,
    "epsilon": 2 / 255,        # L-inf bound in [0, 1] space
    "pgd_steps": 10,
    "pgd_alpha": 1 / 255,      # PGD step size
    "model_id": "Qwen/Qwen3-VL-8B-Instruct",
    "image_size": 448,         # standard Qwen-VL resolution
}

LEAF_TRAIN_DEFAULTS = {
    "epochs": 30,
    "batch_size": 128,
    "lr": 1e-5,
    "weight_decay": 1e-4,
    "warmup_steps": 1400,
    "k_adv": 1,                 # Levenshtein distance budget
    "rho": 50,                  # random candidate positions per step
    "constrain": True,          # block new dictionary words
    "clip_model_name": "ViT-L-14",
    "pretrained": "hf-hub:chs20/fare2-clip",  # start from FARE checkpoint
    "data_samples": 80_000,     # subset of DataComp-small
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_default_config():
    return DEFAULT_CONFIG.copy()
