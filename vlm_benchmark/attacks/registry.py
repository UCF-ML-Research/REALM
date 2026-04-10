"""Attack registry for dynamic attack creation and configuration."""

from dataclasses import dataclass, field
from typing import Type, Dict, Any, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .base_attack import BaseAttack, AttackConfig


@dataclass
class AttackSpec:
    """Specification for an attack's parameters."""
    name: str
    category: str  # "visual", "physical", etc.
    attack_class: Type["BaseAttack"]
    config_class: Type["AttackConfig"]
    defaults: Dict[str, Any] = field(default_factory=dict)


class AttackRegistry:
    """Central registry for adversarial attacks."""

    _registry: Dict[str, AttackSpec] = {}

    @classmethod
    def register(cls, spec: AttackSpec) -> None:
        """Register an attack.

        Args:
            spec: Attack specification
        """
        cls._registry[spec.name] = spec

    @classmethod
    def create(cls, name: str, **config_kwargs) -> "BaseAttack":
        """Create attack instance with validation.

        Args:
            name: Attack name
            **config_kwargs: Configuration parameters

        Returns:
            Attack instance

        Raises:
            ValueError: If attack name is unknown
        """
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown attack: {name}. Available: {available}")

        spec = cls._registry[name]
        # Apply defaults, then override with provided kwargs
        final_kwargs = {**spec.defaults, **config_kwargs}
        config = spec.config_class(**final_kwargs)
        return spec.attack_class(config)

    @classmethod
    def get_spec(cls, name: str) -> Optional[AttackSpec]:
        """Get attack specification by name.

        Args:
            name: Attack name

        Returns:
            AttackSpec or None if not found
        """
        return cls._registry.get(name)

    @classmethod
    def list_attacks(cls, category: Optional[str] = None) -> List[str]:
        """List registered attack names.

        Args:
            category: Optional category filter

        Returns:
            List of attack names
        """
        if category:
            return [n for n, s in cls._registry.items() if s.category == category]
        return list(cls._registry.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if an attack is registered.

        Args:
            name: Attack name

        Returns:
            True if registered
        """
        return name in cls._registry


def register_all_attacks():
    """Register all attacks on module import."""
    from .physpatch.physpatch_attack import PhysPatchAttack, PhysPatchConfig
    from .foa.foa_attack import FOAAttack, FOAAttackConfig
    from .mattack.mattack_attack import MAttack, MAttackConfig
    from .coa.coa_attack import COAAttack, COAAttackConfig
    from .advdiffvlm.advdiffvlm_attack import AdvDiffVLMAttack, AdvDiffVLMConfig

    # === Physical Attacks (surrogate-based, transferable) ===
    AttackRegistry.register(AttackSpec(
        name="physpatch",
        category="physical",
        attack_class=PhysPatchAttack,
        config_class=PhysPatchConfig,
    ))

    # === FOA-Attack (full-image perturbation with OT loss) ===
    AttackRegistry.register(AttackSpec(
        name="foa",
        category="physical",
        attack_class=FOAAttack,
        config_class=FOAAttackConfig,
    ))

    # === M-Attack (simple cosine similarity, NO OT) ===
    AttackRegistry.register(AttackSpec(
        name="mattack",
        category="physical",
        attack_class=MAttack,
        config_class=MAttackConfig,
    ))

    # === CoA (Chain of Attack - iterative caption generation) ===
    AttackRegistry.register(AttackSpec(
        name="coa",
        category="physical",
        attack_class=COAAttack,
        config_class=COAAttackConfig,
    ))

    # === AdvDiffVLM (diffusion-based adversarial examples with AEGE) ===
    AttackRegistry.register(AttackSpec(
        name="advdiffvlm",
        category="diffusion",
        attack_class=AdvDiffVLMAttack,
        config_class=AdvDiffVLMConfig,
    ))

    # === ADVEDM-A / ADVEDM-R (semantic addition / removal) ===
    from .advedm.advedm_attack import (
        ADVEDMAttack,
        ADVEDMConfig,
        ADVEDMRAttack,
        ADVEDMRConfig,
    )
    # SSA-CWA + 4 CLIP ensemble (black-box transferable)
    AttackRegistry.register(AttackSpec(
        name="advedm",
        category="visual",
        attack_class=ADVEDMAttack,
        config_class=ADVEDMConfig,
    ))

    AttackRegistry.register(AttackSpec(
        name="advedm_r",
        category="visual",
        attack_class=ADVEDMRAttack,
        config_class=ADVEDMRConfig,
    ))

    # === FigStep (typographic prompt injection for AD hallucination) ===
    from .figstep.figstep_attack import FigStepAttack
    from .figstep.config import FigStepConfig

    AttackRegistry.register(AttackSpec(
        name="figstep",
        category="typographic",
        attack_class=FigStepAttack,
        config_class=FigStepConfig,
        defaults={"epsilon": 0.0, "max_iterations": 1},
    ))

    # === V-Attack (text-guided Value feature manipulation) ===
    from .vattack.vattack_attack import VAttack, VAttackConfig
    AttackRegistry.register(AttackSpec(
        name="vattack",
        category="physical",
        attack_class=VAttack,
        config_class=VAttackConfig,
    ))

    # === PromptInject (text-level false premise injection) ===
    from .promptinject.promptinject_attack import PromptInjectAttack
    from .promptinject.config import PromptInjectConfig

    AttackRegistry.register(AttackSpec(
        name="promptinject",
        category="text",
        attack_class=PromptInjectAttack,
        config_class=PromptInjectConfig,
        defaults={"epsilon": 0.0, "max_iterations": 1},
    ))

    # === AnyAttack (learned Decoder, single forward pass) ===
    from .anyattack.anyattack_attack import AnyAttack, AnyAttackConfig

    AttackRegistry.register(AttackSpec(
        name="anyattack",
        category="physical",
        attack_class=AnyAttack,
        config_class=AnyAttackConfig,
    ))

    # === PA-Attack (Prototype + Attention guided, untargeted) ===
    from .paattack.paattack_attack import PAAttack, PAAttackConfig

    AttackRegistry.register(AttackSpec(
        name="paattack",
        category="visual",
        attack_class=PAAttack,
        config_class=PAAttackConfig,
    ))

    # === ImageMix (alpha-blend / cutmix pixel perturbation baseline) ===
    from .imagemix.imagemix_attack import ImageMixAttack, ImageMixConfig

    AttackRegistry.register(AttackSpec(
        name="imagemix",
        category="physical",
        attack_class=ImageMixAttack,
        config_class=ImageMixConfig,
        defaults={"epsilon": 0.0, "max_iterations": 1},
    ))

    # === Corruption (natural image corruptions, benign baseline) ===
    from .corruption.corruption_attack import CorruptionAttack
    from .corruption.config import CorruptionConfig

    AttackRegistry.register(AttackSpec(
        name="corruption",
        category="natural",
        attack_class=CorruptionAttack,
        config_class=CorruptionConfig,
        defaults={"epsilon": 0.0, "max_iterations": 1},
    ))
