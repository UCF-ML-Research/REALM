"""Base attack module for adversarial testing of VLMs."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..data.base_dataset import Sample
from .attack_logger import get_attack_logger

logger = get_attack_logger("base")


@dataclass
class AttackConfig:
    """Configuration for adversarial attacks."""
    epsilon: float = 8.0/255.0
    attack_type: str = "image"  # "image", "text", "multimodal"
    seed: int = 42
    max_iterations: int = 10
    alpha: Optional[float] = None  # defaults to epsilon/10 if None
    device: str = "cuda"

    def __post_init__(self):
        """Set default alpha if not provided."""
        if self.alpha is None:
            self.alpha = self.epsilon / 10.0


@dataclass
class AttackResult:
    """Result from an adversarial attack."""
    success: bool
    adversarial_sample: Any
    original_output: str
    adversarial_output: str
    perturbation_norm: float
    queries: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAttack(ABC):
    """Abstract base class for all adversarial attacks."""

    def __init__(self, config: AttackConfig):
        """Initialize attack with configuration."""
        self.config = config

        import random
        import numpy as np
        import torch
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

    @abstractmethod
    def generate(
        self,
        model,
        sample: Sample,
        **kwargs
    ) -> AttackResult:
        """Generate adversarial example for the given sample."""
        pass

    def _run_inference_multi(
        self,
        model,
        sample: Sample,
        question: str,
        **kwargs,
    ) -> str:
        """Run model inference on all sample images."""
        images = sample.images or []
        output = model.inference(images, question, **kwargs)
        return output.text
