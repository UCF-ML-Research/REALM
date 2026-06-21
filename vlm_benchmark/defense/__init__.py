"""Defense Methods Module for VLM Benchmark"""

__version__ = "0.1.0"

from .base_defense import DefenseConfig, DefenseResult, BaseDefense
from .pad.pad_defense import PADDefenseConfig, PADDefense
from .freqpure.freqpure_defense import FreqPureDefenseConfig, FreqPureDefense
from .bluesuffix.bluesuffix_defense import BlueSuffixDefenseConfig, BlueSuffixDefense
from .registry import DefenseRegistry, DefenseSpec, register_all_defenses

register_all_defenses()

__all__ = [
    "DefenseConfig", "DefenseResult", "BaseDefense",
    "PADDefenseConfig", "PADDefense",
    "FreqPureDefenseConfig", "FreqPureDefense",
    "BlueSuffixDefenseConfig", "BlueSuffixDefense",
    "DefenseRegistry", "DefenseSpec",
]
