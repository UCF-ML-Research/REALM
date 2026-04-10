"""Base classes for defense methods."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict
import torch
import random
import numpy as np


@dataclass
class DefenseConfig:
    """Base configuration for defenses."""
    device: str = "cuda"
    seed: int = 42


@dataclass
class DefenseResult:
    """Result from defense cleaning."""
    cleaned_sample: Any              # PIL Image
    original_image_path: str
    detection_confidence: float = 0.0
    regions_removed: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseDefense(ABC):
    """Abstract base class for all defenses."""

    def __init__(self, config: DefenseConfig):
        self.config = config
        self._set_seed(config.seed)

    def _set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    @abstractmethod
    def clean(self, image_path: str, **kwargs) -> DefenseResult:
        """
        Clean adversarial image.

        Args:
            image_path: Path to adversarial image
            **kwargs: Additional defense-specific parameters

        Returns:
            DefenseResult with cleaned image and metadata
        """
        pass

    @abstractmethod
    def requires_model(self) -> bool:
        """
        Return whether defense needs a VLM for cleaning.

        Returns:
            True if defense requires VLM, False otherwise
        """
        pass

    def get_system_prompt(self) -> str | None:
        """Return a hardened system prompt, or None to use the default.

        Override in prompt-level defenses (e.g. SystemPromptDefense).
        Image-purification defenses leave this as None.
        """
        return None
