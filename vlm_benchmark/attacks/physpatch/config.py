"""PhysPatch attack configuration."""

from pathlib import Path

BASE_DIR = Path(__file__).parent


def resolve_cli_kwargs(attack_name, args, context):
    """Build kwargs for AttackRegistry.create() from CLI args."""
    kwargs = {
        "device": args.device,
        "target_images_dir": context["target_dir"],
        "source_images_dir": context["source_dir"],
    }
    if getattr(args, "epsilon", None) is not None:
        kwargs["epsilon"] = args.epsilon
    if getattr(args, "steps", None) is not None:
        kwargs["max_iterations"] = args.steps
    if getattr(args, "coords_file", None):
        kwargs["coords_file"] = args.coords_file
    else:
        # Auto-detect coordinates file from source directory
        kwargs["coords_file"] = _find_coords_file(context["source_dir"])
    return kwargs


def _find_coords_file(source_dir: str) -> str:
    """Auto-find coordinates file from dataset structure.

    Looks for coordinates/*.txt in the dataset root (parent of source dir).
    """
    dataset_root = Path(source_dir).parent
    coords_dir = dataset_root / "coordinates"
    if coords_dir.is_dir():
        txts = sorted(coords_dir.glob("*.txt"))
        if txts:
            return str(txts[0])
    # Legacy fallback
    return str(dataset_root / "coordinates" / "full.txt")


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
    kwargs["coords_file"] = get_coords_file(args.clean_images)
    kwargs["target_images_dir"] = str(Path(args.clean_images).parent.parent / "images" / "target")
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


def get_coords_file(clean_images_dir: str):
    """Auto-infer coordinates file from clean images directory."""
    clean_path = Path(clean_images_dir)
    dataset_root = clean_path.parent.parent
    coords_file = dataset_root / "coordinates" / "full.txt"
    return str(coords_file)
