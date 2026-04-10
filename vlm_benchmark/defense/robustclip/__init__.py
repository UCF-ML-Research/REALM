"""RobustCLIP Defense (FARE + LEAF).

FARE: Unsupervised adversarial fine-tuning of the CLIP vision encoder (ICML 2024).
LEAF: Adversarial fine-tuning of the CLIP text encoder (NeurIPS 2025).

Combined, they provide dual-domain adversarial robustness as a plug-and-play
encoder swap for any VLM that uses CLIP.
"""

from .robustclip_defense import RobustCLIPDefense, RobustCLIPDefenseConfig
from .config import get_default_config

__all__ = [
    "RobustCLIPDefense",
    "RobustCLIPDefenseConfig",
    "get_default_config",
]
