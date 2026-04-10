"""AnyAttack configuration."""


def resolve_cli_kwargs(attack_name, args, context):
    """Build kwargs for AttackRegistry.create() from CLI args."""
    kwargs = {
        "device": args.device,
        "target_images_dir": context["target_dir"],
    }
    if getattr(args, "epsilon", None) is not None:
        kwargs["epsilon"] = args.epsilon
    return kwargs


def add_cli_args(parser):
    parser.add_argument('--anyattack_checkpoint', type=str, default=None,
                        help='Decoder checkpoint name or path (default: coco_bi)')


def resolve_generate_kwargs(attack_name, args):
    kwargs = {"device": args.device}
    if args.epsilon is not None:
        kwargs["epsilon"] = args.epsilon
    if args.anyattack_checkpoint is not None:
        kwargs["decoder_checkpoint"] = args.anyattack_checkpoint
    if hasattr(args, 'target') and args.target:
        kwargs["target_images_dir"] = args.target
    return kwargs


def get_eval_target(attack_name, config):
    from ..cli_utils import EVAL_QUERY
    return {
        "description": "AnyAttack decoder-based perturbation",
        "reference_text": "Target-guided adversarial noise",
        "target_image": None,
        "evaluation_query": EVAL_QUERY,
    }
