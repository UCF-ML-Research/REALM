"""FOA (Full-Image Adversarial) attack configuration."""

from pathlib import Path

BASE_DIR = Path(__file__).parent


def resolve_cli_kwargs(attack_name, args, context):
    """Build kwargs for AttackRegistry.create() from CLI args."""
    kwargs = {
        "device": args.device,
        "attack_method": "pgd",
        "target_images_dir": context["target_dir"],
    }
    if getattr(args, "epsilon", None) is not None:
        kwargs["epsilon"] = args.epsilon
    if getattr(args, "steps", None) is not None:
        kwargs["max_iterations"] = args.steps
    if getattr(args, "alpha", None) is not None:
        kwargs["alpha"] = args.alpha
    return kwargs


def add_cli_args(parser):
    parser.add_argument('--target_strategy', type=str, default=None,
                        help='Target strategy: stop_sign, plane, etc. (default: stop_sign)')
    parser.add_argument('--cluster_number', type=int, default=3,
                        help='Number of k-means clusters for FOA (default: 3)')


def resolve_generate_kwargs(attack_name, args):
    kwargs = {"device": args.device}
    if args.epsilon is not None:
        kwargs["epsilon"] = args.epsilon
    if args.num_steps is not None:
        kwargs["max_iterations"] = args.num_steps
    if args.alpha is not None:
        kwargs["alpha"] = args.alpha
    kwargs["target_images_dir"] = get_target_images_dir(args.clean_images)
    if args.cluster_number is not None:
        kwargs["cluster_number"] = args.cluster_number
    return kwargs


def get_eval_target(attack_name, config):
    from ..cli_utils import STANDARD_TARGETS, EVAL_QUERY
    strategy = getattr(config, "target_strategy", "stop_sign")
    base = STANDARD_TARGETS.get(strategy, {
        "description": f"Target: {strategy}",
        "reference_text": f"A {strategy} is visible",
        "target_image": None,
    })
    return {**base, "evaluation_query": EVAL_QUERY}


def get_target_images_dir(clean_images_dir: str):
    """Auto-infer target images directory from clean images directory."""
    clean_path = Path(clean_images_dir)
    dataset_root = clean_path.parent.parent
    target_dir = dataset_root / "images" / "target"
    return str(target_dir)
