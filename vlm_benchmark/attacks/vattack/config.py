"""V-Attack configuration."""


def resolve_cli_kwargs(attack_name, args, context):
    """Build kwargs for AttackRegistry.create() from CLI args."""
    kwargs = {
        "device": args.device,
        "source_text": getattr(args, "source_text", "") or "",
        "target_text": getattr(args, "target_text", "") or "",
    }
    if getattr(args, "epsilon", None) is not None:
        kwargs["epsilon"] = args.epsilon
    if getattr(args, "steps", None) is not None:
        kwargs["max_iterations"] = args.steps
    if getattr(args, "alpha", None) is not None:
        kwargs["alpha"] = args.alpha
    return kwargs


def add_cli_args(parser):
    parser.add_argument('--vattack_backbone', type=str, nargs='+',
                        default=None,
                        help='CLIP backbones: B16, B32, Laion (default: all three)')
    parser.add_argument('--source_text', type=str, default='',
                        help='Source text description for V-Attack')
    parser.add_argument('--target_text', type=str, default='',
                        help='Target text description for V-Attack')


def resolve_generate_kwargs(attack_name, args):
    kwargs = {"device": args.device}
    if args.epsilon is not None:
        kwargs["epsilon"] = args.epsilon
    if args.num_steps is not None:
        kwargs["max_iterations"] = args.num_steps
    if args.alpha is not None:
        kwargs["alpha"] = args.alpha
    if args.vattack_backbone is not None:
        kwargs["backbone"] = args.vattack_backbone
    if args.source_text:
        kwargs["source_text"] = args.source_text
    if args.target_text:
        kwargs["target_text"] = args.target_text
    return kwargs


def get_eval_target(attack_name, config):
    from ..cli_utils import EVAL_QUERY
    return {
        "description": "V-Attack text-guided perturbation",
        "reference_text": getattr(config, "target_text", ""),
        "target_image": None,
        "evaluation_query": EVAL_QUERY,
    }
