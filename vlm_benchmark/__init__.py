"""
VLM Benchmark for Autonomous Driving
=====================================

Minimal framework for PhysPatch adversarial evaluation.
"""

__version__ = "0.2.0"

# Data loading
from .data import (
    Sample,
    DrivingSample,  # backward-compatible alias
    BaseDataset,
    BehaviorLabel,
    PhysPatchDataset,
)

__all__ = [
    # Data - Base
    "Sample",
    "DrivingSample",  # backward-compatible alias
    "BaseDataset",
    "BehaviorLabel",
    # Data - PhysPatch dataset
    "PhysPatchDataset",
]
