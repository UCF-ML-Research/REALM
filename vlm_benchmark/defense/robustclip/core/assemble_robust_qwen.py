#!/usr/bin/env python3
"""Assemble a serveable robust Qwen3-VL model from a FARE-trained vision checkpoint.

Two assembly modes:
  1. AutoModel (default): loads full model, swaps vision weights, saves.
     Requires transformers to support the model_type (e.g. qwen3_vl).
  2. Shard-patch (--patch-shard): patches safetensors shard directly.
     Works even when transformers doesn't support the model_type.

Usage:
    python -m vlm_benchmark.defense.robustclip.core.assemble_robust_qwen \
        --model_id Qwen/Qwen3-VL-8B-Instruct \
        --vision_checkpoint checkpoints/fare_qwen_vision_final.pt \
        --output_dir /path/to/robust-Qwen3-VL-8B
"""

import argparse
import json
import os
from pathlib import Path

import torch


def assemble_via_automodel(args):
    """Load full model, swap vision weights, save_pretrained."""
    from transformers import AutoTokenizer, AutoProcessor
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading full model: {args.model_id}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        device_map="cpu",
    )

    print(f"Loading vision checkpoint: {args.vision_checkpoint}")
    vision_state = torch.load(
        args.vision_checkpoint, map_location="cpu", weights_only=True,
    )

    # Find the visual module
    if hasattr(model, "model") and hasattr(model.model, "visual"):
        visual_module = model.model.visual
    elif hasattr(model, "visual"):
        visual_module = model.visual
    else:
        raise AttributeError("Cannot find vision encoder at model.model.visual or model.visual")

    missing, unexpected = visual_module.load_state_dict(vision_state, strict=True)
    print(f"Vision encoder weights replaced ({len(vision_state)} keys, strict=True)")

    print(f"Saving robust model to: {output_dir}")
    model.save_pretrained(output_dir, safe_serialization=True)

    # Save tokenizer + processor
    for cls, name in [(AutoTokenizer, "tokenizer"), (AutoProcessor, "processor")]:
        try:
            obj = cls.from_pretrained(args.model_id)
            obj.save_pretrained(output_dir)
        except Exception as e:
            print(f"Warning: could not save {name}: {e}")

    print(f"\nDone. Serve with: vllm serve {output_dir}")


def assemble_via_shard_patch(args):
    """Patch safetensors shard directly — no AutoModel needed."""
    from huggingface_hub import hf_hub_download, HfApi
    from safetensors.torch import load_file, save_file

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_id = args.model_id

    print(f"Loading vision checkpoint: {args.vision_checkpoint}")
    vision_state = torch.load(
        args.vision_checkpoint, map_location="cpu", weights_only=True,
    )
    prefixed_state = {f"model.visual.{k}": v for k, v in vision_state.items()}
    print(f"  {len(prefixed_state)} visual weight keys")

    # Find visual shards
    index_path = hf_hub_download(model_id, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)

    visual_shards = set()
    for key, shard in index["weight_map"].items():
        if key.startswith("model.visual."):
            visual_shards.add(shard)

    # Symlink all repo files, then patch visual shards
    api = HfApi()
    for filename in api.list_repo_files(model_id):
        dest = output_dir / filename
        if dest.exists() or dest.is_symlink():
            continue
        try:
            cached = hf_hub_download(model_id, filename)
        except Exception:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        if filename in visual_shards:
            continue  # will be patched below
        os.symlink(cached, dest)

    # Patch visual shards
    for shard_name in visual_shards:
        print(f"  Patching: {shard_name}")
        shard_weights = load_file(hf_hub_download(model_id, shard_name))

        replaced = 0
        for key in list(shard_weights.keys()):
            if key in prefixed_state:
                shard_weights[key] = prefixed_state[key]
                replaced += 1

        print(f"    Replaced {replaced} / {len(prefixed_state)} visual weights")

        dest = output_dir / shard_name
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        save_file(shard_weights, str(dest))

    print(f"\nDone. Serve with: vllm serve {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Assemble robust Qwen3-VL from FARE vision checkpoint",
    )
    parser.add_argument("--model_id", default="Qwen/Qwen3-VL-8B-Instruct",
                        help="Base HuggingFace model ID")
    parser.add_argument("--vision_checkpoint", required=True,
                        help="Path to FARE-trained vision encoder .pt file")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for the assembled model")
    parser.add_argument("--patch-shard", action="store_true",
                        help="Patch safetensors directly instead of using AutoModel")
    args = parser.parse_args()

    if args.patch_shard:
        assemble_via_shard_patch(args)
    else:
        assemble_via_automodel(args)


if __name__ == "__main__":
    main()
