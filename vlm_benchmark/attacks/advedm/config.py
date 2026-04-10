"""ADVEDM attack configuration (addition + removal variants)."""
from pathlib import Path
from typing import Optional

_ADVEDM_DIR = Path(__file__).parent / "assets"


def resolve_cli_kwargs(attack_name, args, context):
    """Build kwargs for AttackRegistry.create() from CLI args."""
    if attack_name == "advedm":
        # Support both single reference image and per-sample target directory
        ref_path = context.get("target_path", "")
        target_dir = context.get("target_dir", "")
        kwargs = {
            "device": args.device,
            "reference_image_path": ref_path,
            "target_images_dir": target_dir,
        }
    else:
        # advedm_r
        kwargs = {
            "device": args.device,
            "target_text": getattr(args, "target_text", ""),
        }
    # Pass through SSA-CWA overrides (epsilon is [0,1] scale, e.g. 16/255)
    if getattr(args, "epsilon", None) is not None:
        kwargs["epsilon"] = args.epsilon
    if getattr(args, "steps", None) is not None:
        kwargs["num_iters"] = args.steps
    return kwargs


def add_cli_args(parser):
    parser.add_argument('--target_strategy', type=str, default=None,
                        help='Target strategy: stop_sign, plane, etc. (default: stop_sign)')
    parser.add_argument('--target_text', type=str, default=None,
                        help='Target semantic text (required for advedm_r)')
    parser.add_argument('--advedm_annotations_file', type=str, default=None,
                        help='Path to ADVEDM annotation JSON (legacy GPT format)')
    parser.add_argument('--advedm_region_size', type=int, default=100,
                        help='ADVEDM GPT annotation region size (default: 100)')
    parser.add_argument('--advedm_gpt_model', type=str, default='gpt-5-mini-2025-08-07',
                        help='ADVEDM GPT model for background region selection')
    parser.add_argument('--advedm_api_key_env', type=str, default='OPENAI_API_KEY',
                        help='Environment variable name holding OpenAI API key')
    parser.add_argument('--advedm_selection_mode', type=str, default='legacy_strict',
                        choices=['legacy_strict'],
                        help='ADVEDM region selection mode (default: legacy_strict)')
    parser.add_argument('--advedm_r_vision_backend', type=str, default='target_vlm',
                        choices=['target_vlm', 'clip'],
                        help='ADVEDM-R vision backend (default: target_vlm)')
    parser.add_argument('--advedm_r_target_vlm_model', type=str, default='liuhaotian/llava-v1.5-7b',
                        help='Target VLM model path/id for ADVEDM-R target_vlm backend')
    parser.add_argument('--advedm_r_clip_model', type=str, default='ViT-L/14@336px',
                        help='CLIP model used by ADVEDM-R text/alignment backend')
    parser.add_argument('--advedm_r_w1', type=float, default=None,
                        help='ADVEDM-R lambda_cls (default: 0.5)')
    parser.add_argument('--advedm_r_w2', type=float, default=None,
                        help='ADVEDM-R lambda_local (default: 2.0)')
    parser.add_argument('--advedm_r_w3', type=float, default=None,
                        help='ADVEDM-R lambda_fix (default: 0.2)')
    parser.add_argument('--advedm_r_k_ratio', type=float, default=None,
                        help='ADVEDM-R top-k removal ratio (default: 0.2)')
    parser.add_argument('--advedm_r_mask_threshold', type=float, default=None,
                        help='ADVEDM-R optional Eq.4 threshold (overrides k-ratio mode)')
    # Blackbox SSA-CWA params
    parser.add_argument('--advedm_bb_num_iters', type=int, default=None,
                        help='SSA-CWA outer iterations (default: 30)')
    parser.add_argument('--advedm_bb_epsilon', type=float, default=None,
                        help='Blackbox L-inf budget (default: 16/255)')
    parser.add_argument('--advedm_bb_inner_step', type=float, default=None,
                        help='SSA-CWA inner step size (default: 250)')
    parser.add_argument('--advedm_bb_ssa_N', type=int, default=None,
                        help='SSA number of augmented samples (default: 20)')
    parser.add_argument('--advedm_bb_ssa_rho', type=float, default=None,
                        help='SSA spectral mask parameter (default: 0.5)')


def resolve_generate_kwargs(attack_name, args):
    kwargs = {"device": args.device}

    if attack_name == "advedm":
        if args.target_strategy:
            kwargs["target_strategy"] = args.target_strategy
        kwargs["reference_image_path"] = get_reference_image_path(
            args.target_strategy or "stop_sign"
        )
        if args.advedm_annotations_file:
            kwargs["annotations_file"] = args.advedm_annotations_file
        _apply_bb_cli_args(kwargs, args)
        return kwargs

    if attack_name == "advedm_r":
        if not args.target_text:
            raise ValueError("--target_text is required for --attack advedm_r")
        kwargs["target_text"] = args.target_text
        for dest, src in [
            ("lambda_cls", "advedm_r_w1"),
            ("lambda_local", "advedm_r_w2"),
            ("lambda_fix", "advedm_r_w3"),
            ("k_ratio", "advedm_r_k_ratio"),
        ]:
            val = getattr(args, src, None)
            if val is not None:
                kwargs[dest] = val
        _apply_bb_cli_args(kwargs, args)
        return kwargs

    raise ValueError(f"Unknown ADVEDM attack: {attack_name}")


def _apply_bb_cli_args(kwargs, args):
    """Apply blackbox SSA-CWA CLI overrides."""
    for dest, src in [
        ("num_iters", "advedm_bb_num_iters"),
        ("epsilon", "advedm_bb_epsilon"),
        ("inner_step_size", "advedm_bb_inner_step"),
        ("ssa_N", "advedm_bb_ssa_N"),
        ("ssa_rho", "advedm_bb_ssa_rho"),
    ]:
        val = getattr(args, src, None)
        if val is not None:
            kwargs[dest] = val


def get_eval_target(attack_name, config):
    from ..cli_utils import STANDARD_TARGETS, EVAL_QUERY
    if attack_name == "advedm_r":
        txt = getattr(config, "target_text", "unknown")
        return {
            "description": f"Semantic removal: {txt}",
            "reference_text": f"Remove semantic: {txt}",
            "target_image": None,
            "evaluation_query": EVAL_QUERY,
        }
    strategy = getattr(config, "target_strategy", "stop_sign")
    base = STANDARD_TARGETS.get(strategy, {
        "description": f"Target: {strategy}",
        "reference_text": f"A {strategy} is visible",
        "target_image": None,
    })
    return {**base, "evaluation_query": EVAL_QUERY}


def get_reference_image_path(target_strategy: str) -> str:
    """Return path to reference image for given target strategy."""
    ref_dir = _ADVEDM_DIR / "reference"
    candidates = [
        ref_dir / f"{target_strategy}.png",
        ref_dir / f"{target_strategy}.jpg",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        f"Reference image not found for '{target_strategy}' in {ref_dir}"
    )


def get_annotations_file(clean_images_dir: str) -> Optional[str]:
    """Infer annotations JSON from dataset structure, or return None."""
    dataset_root = Path(clean_images_dir).parent.parent
    candidates = [
        dataset_root / "annotations" / "advedm_bboxes.json",
        dataset_root / "advedm_annotations.json",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None
