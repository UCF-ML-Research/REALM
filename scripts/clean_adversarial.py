#!/usr/bin/env python3
"""
Clean adversarial images using registered defenses.

Usage:
    python scripts/clean_adversarial.py --defense pad --adversarial_images adv/ -o out/
    python scripts/clean_adversarial.py --defense freqpure --adversarial_images adv_scene.png -o out/
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from vlm_benchmark.defense import DefenseRegistry


def parse_args():
    p = argparse.ArgumentParser(description="Clean adversarial images")
    p.add_argument("--defense", required=True, help="Defense name (e.g. pad, freqpure, bluesuffix)")
    p.add_argument("--adversarial_images", required=True, help="Image file or directory")
    p.add_argument("--output_dir", "-o", required=True, help="Output directory")
    p.add_argument("--max_samples", type=int, default=100)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--prompt", type=str, default=None,
                   help="Text prompt (passed to defense via kwargs, used by bluesuffix)")

    # Add defense-specific CLI args from registry
    for cli_args in DefenseRegistry.get_all_cli_arguments().values():
        for spec in cli_args:
            name = spec.pop("name")
            p.add_argument(name, **spec)
            spec["name"] = name  # restore
    return p.parse_args()


def load_images(path_str: str, max_samples: int) -> list[Path]:
    p = Path(path_str)
    if p.is_file():
        return [p]

    # Check for manifest from generate_adversarial.py
    for manifest_name in ("results.json", "attack_results.json"):
        manifest = p / manifest_name
        if not manifest.exists():
            continue
        data = json.loads(manifest.read_text())
        files = []
        for r in data.get("results", [])[:max_samples]:
            for key in ("output", "adversarial_image_path"):
                if key in r:
                    fp = Path(r[key])
                    if not fp.is_absolute():
                        fp = p / fp
                    if fp.exists():
                        files.append(fp)
                    break
        if files:
            return files

    # Plain directory
    files = sorted(list(p.glob("*.jpg")) + list(p.glob("*.png")) + list(p.glob("*.jpeg")))[:max_samples]
    if not files:
        raise ValueError(f"No images found in {path_str}")
    return files


def main():
    args = parse_args()

    print(f"Defense: {args.defense}")
    print(f"Input:   {args.adversarial_images}")
    print(f"Output:  {args.output_dir}")

    images = load_images(args.adversarial_images, args.max_samples)
    print(f"Images:  {len(images)}")

    # Create defense
    spec = DefenseRegistry.get_spec(args.defense)
    config_kwargs = {"device": args.device}
    for param in spec.cli_params:
        val = getattr(args, param, None)
        if val is not None:
            config_kwargs[param] = val
    defense = DefenseRegistry.create(args.defense, **config_kwargs)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    t0 = datetime.now()

    # Build kwargs for defense.clean()
    clean_kwargs = {}
    if args.prompt is not None:
        clean_kwargs["prompt"] = args.prompt

    for img_path in tqdm(images, desc="Cleaning"):
        try:
            result = defense.clean(str(img_path), **clean_kwargs)
            out_path = out_dir / img_path.name
            result.cleaned_sample.save(out_path)
            entry = {
                "original": str(img_path),
                "cleaned": out_path.name,
                "detection_confidence": result.detection_confidence,
                "regions_removed": result.regions_removed,
                "metadata": result.metadata,
            }
            if result.metadata.get("final_prompt"):
                entry["final_prompt"] = result.metadata["final_prompt"]
            results.append(entry)
        except Exception as e:
            import traceback
            print(f"\n[ERROR] {img_path.name}: {e}")
            traceback.print_exc()

    duration = (datetime.now() - t0).total_seconds()

    with open(out_dir / "defense_results.json", "w") as f:
        json.dump({
            "defense": args.defense,
            "n_samples": len(results),
            "duration_s": round(duration, 1),
            "results": results,
        }, f, indent=2)

    avg = f"{duration/len(results):.1f}s/img" if results else "N/A"
    print(f"\nDone: {len(results)} images in {duration:.1f}s ({avg})")


if __name__ == "__main__":
    main()
