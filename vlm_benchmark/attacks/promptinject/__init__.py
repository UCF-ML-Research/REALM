"""PromptInject text-level adversarial suffix injection attack."""

from .promptinject_attack import PromptInjectAttack
from .config import PromptInjectConfig

__all__ = ["PromptInjectAttack", "PromptInjectConfig"]
