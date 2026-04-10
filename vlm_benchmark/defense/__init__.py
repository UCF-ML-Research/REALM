"""Defense Methods Module for VLM Benchmark"""

__version__ = "0.1.0"

# Base classes
from .base_defense import DefenseConfig, DefenseResult, BaseDefense

# PAD defense
from .pad.pad_defense import PADDefenseConfig, PADDefense

# FreqPure defense
from .freqpure.freqpure_defense import FreqPureDefenseConfig, FreqPureDefense

# BlueSuffix defense
from .bluesuffix.bluesuffix_defense import BlueSuffixDefenseConfig, BlueSuffixDefense

# SystemPrompt defense
from .systemprompt.systemprompt_defense import SystemPromptDefenseConfig, SystemPromptDefense

# RobustCLIP defense (FARE + LEAF)
from .robustclip.robustclip_defense import RobustCLIPDefenseConfig, RobustCLIPDefense

# Registry
from .registry import DefenseRegistry, DefenseSpec, register_all_defenses

# Register all defenses on import
register_all_defenses()

__all__ = [
    "DefenseConfig", "DefenseResult", "BaseDefense",
    "PADDefenseConfig", "PADDefense",
    "FreqPureDefenseConfig", "FreqPureDefense",
    "BlueSuffixDefenseConfig", "BlueSuffixDefense",
    "SystemPromptDefenseConfig", "SystemPromptDefense",
    "RobustCLIPDefenseConfig", "RobustCLIPDefense",
    "DefenseRegistry", "DefenseSpec",
]
