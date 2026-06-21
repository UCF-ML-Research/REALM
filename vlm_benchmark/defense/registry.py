"""Defense registry for managing defense methods."""

from dataclasses import dataclass, field
from typing import Type, Dict, Any, List, Optional
from .base_defense import BaseDefense, DefenseConfig


@dataclass
class DefenseSpec:
    """Defense specification."""
    name: str
    category: str                     # "purification", "detection", etc.
    defense_class: Type[BaseDefense]
    config_class: Type[DefenseConfig]
    cli_params: List[str] = field(default_factory=list)
    defaults: Dict[str, Any] = field(default_factory=dict)
    cli_arguments: List[Dict[str, Any]] = field(default_factory=list)  # CLI arg specs


class DefenseRegistry:
    """Central registry for defenses."""

    _registry: Dict[str, DefenseSpec] = {}

    @classmethod
    def register(cls, spec: DefenseSpec) -> None:
        """Register a defense."""
        if spec.name in cls._registry:
            raise ValueError(f"Defense '{spec.name}' already registered")
        cls._registry[spec.name] = spec

    @classmethod
    def create(cls, name: str, **config_kwargs) -> BaseDefense:
        """Create a defense instance by name with the given config parameters."""
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Unknown defense: {name}. Available: {available}"
            )

        spec = cls._registry[name]
        final_kwargs = {**spec.defaults, **config_kwargs}
        config = spec.config_class(**final_kwargs)
        return spec.defense_class(config)

    @classmethod
    def get_spec(cls, name: str) -> DefenseSpec:
        """Get defense specification."""
        if name not in cls._registry:
            raise ValueError(f"Unknown defense: {name}")
        return cls._registry[name]

    @classmethod
    def get_all_cli_arguments(cls) -> Dict[str, List[Dict[str, Any]]]:
        """Return a mapping of defense name to its CLI argument specs."""
        return {
            name: spec.cli_arguments
            for name, spec in cls._registry.items()
        }


def register_all_defenses():
    """Register all defenses (called on module import)."""
    from .pad.pad_defense import PADDefense, PADDefenseConfig
    from .pad.config import CLI_ARGUMENTS as PAD_CLI_ARGS

    DefenseRegistry.register(DefenseSpec(
        name="pad",
        category="purification",
        defense_class=PADDefense,
        config_class=PADDefenseConfig,
        cli_params=["iou_threshold", "ratio_mi", "kernel_param", "thresh_param"],
        defaults={
            "iou_threshold": 0.5,
            "ratio_mi": 0.5,
            "kernel_param": 80,
            "thresh_param": 80,
        },
        cli_arguments=PAD_CLI_ARGS,
    ))

    from .freqpure.freqpure_defense import FreqPureDefense, FreqPureDefenseConfig
    from .freqpure.config import CLI_ARGUMENTS as FREQPURE_CLI_ARGS

    DefenseRegistry.register(DefenseSpec(
        name="freqpure",
        category="purification",
        defense_class=FreqPureDefense,
        config_class=FreqPureDefenseConfig,
        cli_params=[
            "amplitude_cut_range", "phase_cut_range", "delta", "forward_noise_steps",
            "max_timesteps", "num_denoising_steps", "sampling_method",
        ],
        defaults={
            "amplitude_cut_range": 10,
            "phase_cut_range": 10,
            "delta": 0.3,
            "forward_noise_steps": 50,
            "max_timesteps": "50,50,50,50,50,50,50,50",
            "num_denoising_steps": "5,5,5,5,5,5,5,5",
            "sampling_method": "ddpm",
        },
        cli_arguments=FREQPURE_CLI_ARGS,
    ))

    from .bluesuffix.bluesuffix_defense import BlueSuffixDefense, BlueSuffixDefenseConfig
    from .bluesuffix.config import CLI_ARGUMENTS as BLUESUFFIX_CLI_ARGS

    DefenseRegistry.register(DefenseSpec(
        name="bluesuffix",
        category="purification",
        defense_class=BlueSuffixDefense,
        config_class=BlueSuffixDefenseConfig,
        cli_params=[
            "enable_image_purifier", "enable_text_purifier",
            "enable_suffix_generator", "openai_api_key",
            "max_timesteps", "num_denoising_steps", "sampling_method",
        ],
        defaults={
            "enable_image_purifier": True,
            "enable_text_purifier": True,
            "enable_suffix_generator": True,
            "max_timesteps": "100",
            "num_denoising_steps": "20",
            "sampling_method": "ddim",
        },
        cli_arguments=BLUESUFFIX_CLI_ARGS,
    ))
