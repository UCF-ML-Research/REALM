"""Chain of Attack (CoA) configuration."""

from pathlib import Path

BASE_DIR = Path(__file__).parent

CLIPCAP_WEIGHTS = str(BASE_DIR / "assets" / "conceptual_weights.pt")
CLEAN_CAPTIONS_FILENAME = "clean_captions_qwen.txt"


def resolve_cli_kwargs(attack_name, args, context):
    """Build kwargs for AttackRegistry.create() from CLI args."""
    kwargs = {
        "device": args.device,
        "clean_images_dir": context["source_dir"],
        "target_images_dir": context["target_dir"],
    }
    if getattr(args, "epsilon", None) is not None:
        kwargs["epsilon"] = args.epsilon
    if getattr(args, "steps", None) is not None:
        kwargs["max_iterations"] = args.steps
    if getattr(args, "target_captions", None):
        kwargs["target_captions_path"] = args.target_captions
    if getattr(args, "clean_captions", None):
        kwargs["clean_captions_path"] = args.clean_captions
    return kwargs


def add_cli_args(parser):
    parser.add_argument('--target_strategy', type=str, default=None,
                        help='Target strategy: stop_sign, plane, etc. (default: stop_sign)')


def resolve_generate_kwargs(attack_name, args):
    kwargs = {"device": args.device}
    if args.epsilon is not None:
        kwargs["epsilon"] = args.epsilon
    if args.num_steps is not None:
        kwargs["max_iterations"] = args.num_steps
    if args.alpha is not None:
        kwargs["alpha"] = args.alpha
    kwargs["target_images_dir"] = get_target_images_dir(args.clean_images)
    kwargs["target_captions_path"] = get_target_captions_path(args.clean_images)
    kwargs["clipcap_weights_path"] = CLIPCAP_WEIGHTS
    kwargs["clean_images_dir"] = args.clean_images
    kwargs["clean_captions_path"] = get_clean_captions_path(args.clean_images)
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


def get_clean_captions_path(clean_images_dir: str):
    """Get clean captions path based on clean images directory."""
    clean_path = Path(clean_images_dir)
    dataset_root = clean_path.parent.parent
    captions_dir = dataset_root / "captions"
    captions_dir.mkdir(parents=True, exist_ok=True)
    return str(captions_dir / CLEAN_CAPTIONS_FILENAME)


def get_target_images_dir(clean_images_dir: str) -> str:
    """Infer target images directory from clean images directory."""
    clean_path = Path(clean_images_dir)
    return str(clean_path.parent / "target")


def get_target_captions_path(clean_images_dir: str) -> str:
    """Infer target captions path from clean images directory."""
    clean_path = Path(clean_images_dir)
    dataset_root = clean_path.parent.parent
    return str(dataset_root / "captions" / "target_captions.txt")
