"""
Minimal logging utilities for attacks module.
"""

import logging


def get_attack_logger(name: str) -> logging.Logger:
    """Get a logger for the attacks module."""
    return logging.getLogger(f"vlm_benchmark.attacks.{name}")
