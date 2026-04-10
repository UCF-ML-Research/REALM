"""Data loading modules for VLM benchmark."""

from .base_dataset import Sample, DrivingSample, BaseDataset, BehaviorLabel
from .drivebench_dataset import DriveBenchDataset
from .robo2vlm_dataset import Robo2VLMDataset
from .physpatch_dataset import PhysPatchDataset

__all__ = [
    # Base classes
    "Sample",
    "DrivingSample",  # backward-compatible alias (deprecated)
    "BaseDataset",
    "BehaviorLabel",
    # Dataset implementations
    "DriveBenchDataset",
    "Robo2VLMDataset",
    "PhysPatchDataset",
]
