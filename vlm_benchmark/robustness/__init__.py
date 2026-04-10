"""Robustness testing for VLM benchmark.

Implements DriveBench-style robustness evaluation:
- Clean inputs: Normal evaluation
- Corrupted inputs: Test sensitivity to image degradation
- Text-only inputs: Expose fake visual grounding
"""

from .image_corruption import ImageCorruptor, CorruptionType
from .robustness_evaluator import RobustnessEvaluator, RobustnessResult

__all__ = [
    "ImageCorruptor",
    "CorruptionType",
    "RobustnessEvaluator",
    "RobustnessResult",
]
