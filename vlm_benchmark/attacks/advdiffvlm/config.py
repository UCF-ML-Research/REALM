"""AdvDiffVLM attack configuration."""

from pathlib import Path

BASE_DIR = Path(__file__).parent


def resolve_cli_kwargs(attack_name, args, context):
    """Build kwargs for AttackRegistry.create() from CLI args."""
    masks_dir = str(Path(context["output_dir"]) / "_gradcam_masks")
    kwargs = {
        "device": args.device,
        "target_images_dir": context["target_dir"],
        "auto_generate_masks": True,
        "gradcam_masks_dir": masks_dir,
    }
    if getattr(args, "class_label", None) is not None:
        kwargs["class_label"] = args.class_label
    return kwargs


def add_cli_args(parser):
    parser.add_argument('--target_strategy', type=str, default=None,
                        help='Target strategy: stop_sign, plane, etc. (default: stop_sign)')
    parser.add_argument('--ddim_steps', type=int, default=None,
                        help='DDIM sampling steps (AdvDiffVLM only)')
    parser.add_argument('--gradient_scale', type=float, default=None,
                        help='AEGE gradient scale (AdvDiffVLM only)')
    parser.add_argument('--refinement_iterations', type=int, default=None,
                        help='Number of refinement loops (AdvDiffVLM only)')


def resolve_generate_kwargs(attack_name, args):
    kwargs = {"device": args.device}
    for key in ("ddim_steps", "gradient_scale", "refinement_iterations"):
        val = getattr(args, key, None)
        if val is not None:
            kwargs[key] = val
    if args.target_strategy is not None:
        kwargs["target_strategy"] = args.target_strategy
    if args.epsilon is not None:
        kwargs["epsilon"] = args.epsilon
    if args.num_steps is not None:
        kwargs["max_iterations"] = args.num_steps
    if args.alpha is not None:
        kwargs["alpha"] = args.alpha
    kwargs["target_images_dir"] = get_target_images_dir(args.clean_images)
    kwargs["gradcam_masks_dir"] = get_gradcam_masks_dir(args.clean_images)
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


def get_checkpoint_path():
    """Get path to Latent Diffusion checkpoint (1.8 GB)."""
    checkpoint = BASE_DIR / "assets" / "checkpoints" / "ldm_cin256-v2.ckpt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    return str(checkpoint)


def get_config_yaml_path():
    """Get path to Latent Diffusion config YAML."""
    config_yaml = BASE_DIR / "assets" / "configs" / "latent-diffusion_cin256-v2.yaml"
    if not config_yaml.exists():
        raise FileNotFoundError(f"Config not found: {config_yaml}")
    return str(config_yaml)


def get_clip_cache_dir():
    """Get path to CLIP cache directory (2.1 GB)."""
    clip_cache = BASE_DIR / "assets" / "clip_cache"
    clip_cache.mkdir(parents=True, exist_ok=True)
    return str(clip_cache)


def get_gradcam_masks_dir(clean_images_dir: str):
    """Auto-infer GradCAM masks directory from clean images directory."""
    clean_path = Path(clean_images_dir)
    dataset_root = clean_path.parent.parent
    masks_dir = dataset_root / "gradcam_masks"
    return str(masks_dir)


def get_target_images_dir(clean_images_dir: str):
    """Auto-infer target images directory from clean images directory."""
    clean_path = Path(clean_images_dir)
    dataset_root = clean_path.parent.parent
    target_dir = dataset_root / "images" / "target"
    return str(target_dir)


def infer_class_label_from_target(target_strategy: str) -> int:
    """Map target strategy to ImageNet class label for GradCAM."""
    TARGET_TO_LABEL = {
        "stop_sign": 920,
        "plane": 404,
        "traffic_light": 920,
    }
    return TARGET_TO_LABEL.get(target_strategy, 920)
