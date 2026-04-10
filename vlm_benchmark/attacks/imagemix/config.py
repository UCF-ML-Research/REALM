"""ImageMix attack configuration."""

from dataclasses import dataclass

from ..base_attack import AttackConfig


def resolve_cli_kwargs(attack_name, args, context):
    """Build kwargs for AttackRegistry.create() from CLI args."""
    kwargs = {
        "device": args.device,
        "target_images_dir": context["target_dir"],
    }
    if getattr(args, "imagemix_alpha", None) is not None:
        kwargs["alpha"] = args.imagemix_alpha
    if getattr(args, "imagemix_type", None) is not None:
        kwargs["mix_type"] = args.imagemix_type
    return kwargs


def add_cli_args(parser):
    parser.add_argument('--imagemix_alpha', type=float, default=0.3,
                        help='Blend ratio for mixup (default: 0.3)')
    parser.add_argument('--imagemix_type', type=str, default='mixup',
                        choices=['mixup', 'cutmix'],
                        help='Mix type: mixup | cutmix (default: mixup)')


def resolve_generate_kwargs(attack_name, args):
    return {
        "device": args.device,
        "alpha": args.imagemix_alpha,
        "mix_type": args.imagemix_type,
        "target_images_dir": "",
    }


def get_eval_target(attack_name, config):
    from ..cli_utils import EVAL_QUERY
    mix_type = getattr(config, "mix_type", "mixup")
    alpha = getattr(config, "alpha", 0.3)
    return {
        "description": f"ImageMix ({mix_type}, alpha={alpha})",
        "reference_text": "Image mixing attack",
        "target_image": None,
        "evaluation_query": EVAL_QUERY,
    }


@dataclass
class ImageMixConfig(AttackConfig):
    alpha: float = 0.3
    mix_type: str = "mixup"       # "mixup" or "cutmix"
    target_images_dir: str = ""   # flat dir with {sample_id}.jpg targets
