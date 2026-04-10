"""
Base attack module for adversarial testing of VLMs.

Defines the unified AttackConfig and AttackResult dataclasses, plus the
BaseAttack abstract class that all attack implementations must inherit from.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..data.base_dataset import Sample
from .attack_logger import get_attack_logger

logger = get_attack_logger("base")


@dataclass
class AttackConfig:
    """Configuration for adversarial attacks.

    Attributes:
        epsilon: L_inf perturbation bound (default: 8/255 ≈ 0.031)
        attack_type: Type of attack - "image", "text", or "multimodal"
        targeted: Whether attack is targeted (False = untargeted)
        seed: Random seed for reproducibility
        max_iterations: Maximum iterations for iterative attacks
        alpha: Step size for iterative attacks (default: epsilon/10)
        device: Device to run attack on ("cuda" or "cpu")
        save_adversarial_examples: Whether to save successful adversarial examples
        adversarial_save_dir: Directory to save adversarial examples
    """
    epsilon: float = 8.0/255.0
    attack_type: str = "image"  # "image", "text", "multimodal"
    targeted: bool = False
    seed: int = 42
    max_iterations: int = 10
    alpha: Optional[float] = None  # Will default to epsilon/10 if None
    device: str = "cuda"
    save_adversarial_examples: bool = True
    adversarial_save_dir: str = "results/adversarial_examples"

    def __post_init__(self):
        """Set default alpha if not provided."""
        if self.alpha is None:
            self.alpha = self.epsilon / 10.0


@dataclass
class AttackResult:
    """Result from an adversarial attack.

    Attributes:
        success: Whether the attack succeeded (changed model output)
        adversarial_sample: Modified image/text/both
        original_output: Model output on clean input
        adversarial_output: Model output on adversarial input
        perturbation_norm: L_inf norm of perturbation
        queries: Number of model queries used (for black-box attacks)
        metadata: Additional attack-specific information
    """
    success: bool
    adversarial_sample: Any  # Modified image, text, or both
    original_output: str
    adversarial_output: str
    perturbation_norm: float
    queries: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAttack(ABC):
    """Abstract base class for all adversarial attacks.

    All attack implementations should inherit from this class and implement
    the required abstract methods.
    """

    def __init__(self, config: AttackConfig):
        """Initialize attack with configuration.

        Args:
            config: Attack configuration
        """
        self.config = config

        # Set random seed for reproducibility
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
        """Generate adversarial example for the given sample.

        Args:
            model: VLM model wrapper (must have inference method)
            sample: Sample to attack
            **kwargs: Additional attack-specific parameters

        Returns:
            AttackResult containing adversarial example and metadata
        """
        pass

    @abstractmethod
    def is_gradient_based(self) -> bool:
        """Return whether this attack requires gradients.

        Returns:
            True if attack needs gradients, False otherwise
        """
        pass

    def _run_inference_multi(
        self,
        model,
        sample: Sample,
        question: str,
        **kwargs,
    ) -> str:
        """Run model inference on all sample images.

        Args:
            model: VLM model wrapper
            sample: Sample (for images)
            question: Question to ask
            **kwargs: Extra keyword arguments forwarded to model.inference()

        Returns:
            Model output text
        """
        images = sample.images or []
        output = model.inference(images, question, **kwargs)
        return output.text
