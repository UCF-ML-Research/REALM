"""Corruption attack configuration."""

from dataclasses import dataclass

from ..base_attack import AttackConfig


VALID_MODES = ("brightness", "fog", "lowlight", "motionblur", "watersplash", "saturate")


def resolve_cli_kwargs(attack_name, args, context):
    """Build kwargs for AttackRegistry.create() from CLI args."""
    kwargs = {"device": args.device}
    if args.corruption_mode is not None:
        kwargs["mode"] = args.corruption_mode
    if args.corruption_severity is not None:
        kwargs["severity"] = args.corruption_severity
    return kwargs


def add_cli_args(parser):
    parser.add_argument('--corruption_mode', type=str, default='fog',
                        choices=list(VALID_MODES),
                        help='Corruption type (default: fog)')
    parser.add_argument('--corruption_severity', type=int, default=3,
                        choices=[1, 2, 3, 4, 5],
                        help='Corruption severity 1-5 (default: 3)')


def resolve_generate_kwargs(attack_name, args):
    return {
        "device": args.device,
        "mode": args.corruption_mode,
        "severity": args.corruption_severity,
    }


def get_eval_target(attack_name, config):
    from ..cli_utils import EVAL_QUERY
    mode = getattr(config, "mode", "fog")
    severity = getattr(config, "severity", 3)
    return {
        "description": f"Corruption ({mode}, severity={severity})",
        "reference_text": "Natural corruption baseline",
        "target_image": None,
        "evaluation_query": EVAL_QUERY,
    }


@dataclass
class CorruptionConfig(AttackConfig):
    mode: str = "fog"        # brightness, fog, lowlight, motionblur, watersplash, saturate
    severity: int = 3        # 1-5
