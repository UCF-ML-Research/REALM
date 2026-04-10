"""
FOA-Attack Module
=================

Full-image adversarial perturbation using Optimal Transport loss.
"""

from .foa_attack import FOAAttack, FOAAttackConfig

__all__ = [
    "FOAAttack",
    "FOAAttackConfig",
]
