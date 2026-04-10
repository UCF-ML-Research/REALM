"""
Centralized configuration for VLM Benchmark.

Paths are configurable via environment variables or defaults.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Data root - configurable via environment variable
# Try environment variables first, then fall back to local ./data directory
_data_root_env = os.environ.get("VLM_DATA_ROOT") or os.environ.get("VLM_BENCHMARK_DATA")

if _data_root_env:
    DATA_ROOT = _data_root_env
else:
    # Use local ./data as default (portable across systems)
    DATA_ROOT = str(PROJECT_ROOT / "data")
    # Print warning on first import
    import warnings
    warnings.warn(
        f"VLM_DATA_ROOT environment variable not set. Using local default: {DATA_ROOT}\n"
        f"To suppress this warning, set: export VLM_DATA_ROOT=/path/to/your/data",
        UserWarning,
        stacklevel=2
    )

# Dataset paths - relative to DATA_ROOT
DATASET_PATHS = {
    "drivebench": os.path.join(DATA_ROOT, "DriveBench"),
}

# Results directory
RESULTS_DIR = os.environ.get(
    "VLM_RESULTS_DIR",
    str(PROJECT_ROOT / "results")
)

# Cache directory
CACHE_DIR = os.environ.get(
    "VLM_CACHE_DIR",
    str(Path.home() / ".cache" / "vlm_benchmark")
)

# Create directories if they don't exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


def get_dataset_path(dataset_name: str) -> str:
    """Get path for a dataset.

    Args:
        dataset_name: Name of dataset ('drivelm', 'nuscenes_qa', etc.)

    Returns:
        Path to dataset

    Raises:
        ValueError: If dataset name is unknown
    """
    dataset_name = dataset_name.lower().replace("-", "").replace("_", "")

    # Normalize names
    name_mapping = {
        "drivebench": "drivebench",
    }

    normalized_name = name_mapping.get(dataset_name)
    if normalized_name is None:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return DATASET_PATHS.get(normalized_name, os.path.join(DATA_ROOT, dataset_name))


def print_config():
    """Print current configuration."""
    print("=== VLM Benchmark Configuration ===")
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"DATA_ROOT:    {DATA_ROOT}")
    print(f"RESULTS_DIR:  {RESULTS_DIR}")
    print(f"CACHE_DIR:    {CACHE_DIR}")
    print("\nDataset Paths:")
    for name, path in DATASET_PATHS.items():
        exists = "✓" if os.path.exists(path) else "✗"
        print(f"  {name:15s}: {exists} {path}")
    print("\nTo customize paths, set environment variables:")
    print("  export VLM_DATA_ROOT=/path/to/data")
    print("  export VLM_RESULTS_DIR=/path/to/results")
    print("  export VLM_CACHE_DIR=/path/to/cache")


# ── Constants moved from scripts/run_vlm_benchmark.py ──────────

# Supported model types (used only for CLI --model choices)
MODEL_TYPES = ["qwen", "qwen_vl", "dolphins", "llava", "internvl"]

# Registry of available datasets
DATASET_REGISTRY = {
    "drivebench": "vlm_benchmark.data.DriveBenchDataset",
    "robo2vlm": "vlm_benchmark.data.Robo2VLMDataset",
    "physpatch": "vlm_benchmark.data.PhysPatchDataset",
}

# Base data path
_DATA_BASE = "/orange/qi855292.ucf/yi217029.ucf/Autonomous"

# Default data paths and constructor kwargs per dataset
DATA_PATH_DEFAULTS = {
    "drivebench": {
        "data_root": f"{_DATA_BASE}/DriveBench/data/corruption",
        "qa_json_path": f"{_DATA_BASE}/DriveBench/data/drivebench-test.json",
        "nuscenes_root": f"{_DATA_BASE}/DriveBench/data/nuscenes",
    },
    "robo2vlm": {
        "data_root": f"{_DATA_BASE}/Robo2VLM/train",
        "json_path": f"{_DATA_BASE}/Robo2VLM/train/robo2vlm_train.json",
    },
    "physpatch": {
        "data_root": f"{_DATA_BASE}/PhysPatch",
        "mode": "clean",  # Default to clean baseline
        "target": "stop_sign",  # Default adversarial target
    },
}

# Per-dataset max_new_tokens (legacy — prefer DECODING_MODES below)
DATASET_MAX_TOKENS = {
    "drivebench": 512,  # Official DriveBench uses 512
    "physpatch": 512,   # PhysPatch uses same as GPT-4o (1024 in paper, but 512 is sufficient)
}

# Per-dataset temperature (legacy — prefer DECODING_MODES below)
DATASET_TEMPERATURE = {
    "drivebench": 0.2,  # Official DriveBench uses 0.2
    "physpatch": 0.0,   # PhysPatch uses temperature=0 (greedy, deterministic)
}

# Per-dataset top_p (legacy — prefer DECODING_MODES below)
DATASET_TOP_P = {
    "drivebench": 0.95,  # Qwen2.5-VL recommended
}

# Per-dataset repetition penalty (legacy — prefer DECODING_MODES below)
DATASET_REPETITION_PENALTY = {
    "drivebench": 1.03,  # Qwen2.5-VL recommended
}

# Two-mode decoding configuration
# "paper" — use Qwen2.5-VL recommended parameters for best model performance
# "standard" — uniform greedy decoding for fair cross-dataset comparison
DECODING_MODES = {
    "paper": {
        "drivebench":    {
            "max_tokens": 512,      # Official DriveBench default
            "temperature": 0.2,     # Qwen2.5-VL recommended
            "top_p": 0.95,          # Qwen2.5-VL recommended (was 0.2 DriveBench)
            "repetition_penalty": 1.03,  # Qwen2.5-VL recommended (was 1.3)
        },
    },
    "standard": {
        "drivebench":    {
            "max_tokens": 512,      # Official DriveBench default
            "temperature": 0.0,     # Greedy decoding for reproducible evaluation
        },
    },
}


def get_decoding_params(dataset_type: str, mode: str = "paper") -> dict:
    """Get decoding parameters for a dataset under a given mode.

    Args:
        dataset_type: Dataset name (e.g. "drivelm", "drivebench")
        mode: "paper" for paper-aligned params, "standard" for uniform greedy

    Returns:
        Dict with keys max_tokens, temperature, and optionally num_beams
    """
    if mode not in DECODING_MODES:
        raise ValueError(f"Unknown decoding mode: {mode}. Choose 'paper' or 'standard'.")
    params = DECODING_MODES[mode].get(dataset_type)
    if params is None:
        # Fallback to legacy dicts
        return {
            "max_tokens": DATASET_MAX_TOKENS.get(dataset_type, 128),
            "temperature": DATASET_TEMPERATURE.get(dataset_type, 0.0),
        }
    return dict(params)  # return a copy
