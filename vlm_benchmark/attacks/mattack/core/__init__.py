"""M-Attack core functions."""

from .attacks import fgsm_attack, mifgsm_attack, pgd_attack

__all__ = ["fgsm_attack", "mifgsm_attack", "pgd_attack"]
