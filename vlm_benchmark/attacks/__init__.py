"""
Adversarial Attacks Module for VLM Benchmark
==============================================

Implements 14 adversarial attacks for evaluating VLM robustness.

Usage:
    from vlm_benchmark.attacks import AttackRegistry

    attack = AttackRegistry.create("foa", epsilon=16, max_iterations=300)
    result = attack.generate(model=None, sample=sample)
"""

__version__ = "0.1.0"

# Base classes
from .base_attack import (
    AttackConfig,
    AttackResult,
    BaseAttack,
)

# Physical attacks
from .physpatch.physpatch_attack import (
    PhysPatchConfig,
    PhysPatchAttack,
)
from .foa.foa_attack import (
    FOAAttackConfig,
    FOAAttack,
)
from .mattack.mattack_attack import (
    MAttackConfig,
    MAttack,
)
from .coa.coa_attack import (
    COAAttackConfig,
    COAAttack,
)
from .advdiffvlm.advdiffvlm_attack import (
    AdvDiffVLMConfig,
    AdvDiffVLMAttack,
)
from .advedm.advedm_attack import (
    ADVEDMConfig,
    ADVEDMAttack,
    ADVEDMRConfig,
    ADVEDMRAttack,
)

# V-Attack (text-guided gradient)
from .vattack.vattack_attack import (
    VAttackConfig,
    VAttack,
)

# AnyAttack (learned decoder)
from .anyattack.anyattack_attack import (
    AnyAttackConfig,
    AnyAttack,
)

# PA-Attack (prototype + attention guided)
from .paattack.paattack_attack import (
    PAAttackConfig,
    PAAttack,
)

# Typographic attacks
from .figstep.figstep_attack import (
    FigStepConfig,
    FigStepAttack,
)

# Text attacks
from .promptinject.promptinject_attack import (
    PromptInjectConfig,
    PromptInjectAttack,
)

# Baselines
from .corruption.corruption_attack import (
    CorruptionConfig,
    CorruptionAttack,
)
from .imagemix.imagemix_attack import (
    ImageMixConfig,
    ImageMixAttack,
)

# Attack registry
from .registry import AttackRegistry, AttackSpec, register_all_attacks

# Register all attacks on import
register_all_attacks()

__all__ = [
    # Base
    "AttackConfig",
    "AttackResult",
    "BaseAttack",
    # Physical attacks
    "PhysPatchConfig",
    "PhysPatchAttack",
    "FOAAttackConfig",
    "FOAAttack",
    "MAttackConfig",
    "MAttack",
    "COAAttackConfig",
    "COAAttack",
    "VAttackConfig",
    "VAttack",
    "AnyAttackConfig",
    "AnyAttack",
    # Diffusion attacks
    "AdvDiffVLMConfig",
    "AdvDiffVLMAttack",
    # ADVEDM (SSA-CWA ensemble)
    "ADVEDMConfig",
    "ADVEDMAttack",
    "ADVEDMRConfig",
    "ADVEDMRAttack",
    # PA-Attack
    "PAAttackConfig",
    "PAAttack",
    # Typographic attacks
    "FigStepConfig",
    "FigStepAttack",
    # Text attacks
    "PromptInjectConfig",
    "PromptInjectAttack",
    # Baselines
    "CorruptionConfig",
    "CorruptionAttack",
    "ImageMixConfig",
    "ImageMixAttack",
    # Registry
    "AttackRegistry",
    "AttackSpec",
]
