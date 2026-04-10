#!/usr/bin/env python3
"""
PhysPatch Coordinate Generation Pipeline

Integrates SoM (Set-of-Mark) → SAM → GPT-4V pipeline for automatic
patch placement coordinate generation.

Pipeline:
    1. Check if coordinates exist → use them
    2. If not, check if SoM/labels exist → run GPT-4V selection
    3. If not, generate SoM labels → run GPT-4V selection
    4. Clean images → SAM segmentation → SoM images + label coords → GPT-4V → final coords

Based on legacy code:
    - SoM/batch_som.py (SAM segmentation)
    - som_gpt.py (GPT-4V coordinate selection)
"""

import os
import json
import base64
import re
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import center_of_mass


def validate_pipeline_data(
    clean_images_dir: str,
    coords_file: Optional[str] = None,
    som_dir: Optional[str] = None,
    sam_labels_dir: Optional[str] = None,
) -> dict:
    """
    Validate what data exists in the pipeline.

    Returns dict with status:
        - coordinates_exist: bool
        - som_exist: bool
        - labels_exist: bool
        - clean_images_exist: bool
        - num_images: int
        - missing_components: list
    """
    status = {
        "coordinates_exist": False,
        "som_exist": False,
        "labels_exist": False,
        "clean_images_exist": False,
        "num_images": 0,
        "missing_components": []
    }

    # Check clean images
    clean_path = Path(clean_images_dir)
    if clean_path.exists():
        image_files = list(clean_path.glob("*.jpg")) + list(clean_path.glob("*.png"))
        status["clean_images_exist"] = len(image_files) > 0
        status["num_images"] = len(image_files)
    else:
        status["missing_components"].append("clean_images")

    # Check coordinates
    if coords_file and Path(coords_file).exists():
        with open(coords_file, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]
        status["coordinates_exist"] = len(lines) > 0
    else:
        status["missing_components"].append("coordinates")

    # Check SoM images
    if som_dir and Path(som_dir).exists():
        som_files = list(Path(som_dir).glob("*.jpg")) + list(Path(som_dir).glob("*.png"))
        status["som_exist"] = len(som_files) > 0
    else:
        status["missing_components"].append("som_images")

    # Check SAM labels
    if sam_labels_dir and Path(sam_labels_dir).exists():
        label_files = list(Path(sam_labels_dir).glob("*.txt"))
        status["labels_exist"] = len(label_files) > 0
    else:
        status["missing_components"].append("sam_labels")

    return status


def generate_som_labels(
    clean_images_dir: str,
    som_output_dir: str,
    labels_output_dir: str,
    sam_checkpoint: str,
    granularity: float = 2.6,
    alpha: float = 0.1,
    label_mode: str = "Number",
    device: str = "cuda"
) -> None:
    """
    Generate SoM annotations and label coordinates using SAM.

    Based on legacy/code/SoM/batch_som.py

    Args:
        clean_images_dir: Directory with clean images
        som_output_dir: Output directory for SoM annotated images
        labels_output_dir: Output directory for label coordinate files
        sam_checkpoint: Path to SAM vit_h checkpoint
        granularity: Segmentation granularity (2.5-3.0 for SAM)
        alpha: Mask transparency
        label_mode: "Number" or "Alphabet"
        device: "cuda" or "cpu"
    """
    from segment_anything import sam_model_registry

    # Import the inference function from legacy code
    import sys
    legacy_som_path = Path(__file__).parent / "assets" / "som"
    sys.path.insert(0, str(legacy_som_path))

    try:
        from task_adapter.sam.tasks.inference_sam_m2m_auto import inference_sam_m2m_auto
    except ImportError:
        raise ImportError(
            f"Cannot import SoM dependencies. Please ensure the legacy SoM code is available at: {legacy_som_path}"
        )

    # Create output directories
    os.makedirs(som_output_dir, exist_ok=True)
    os.makedirs(labels_output_dir, exist_ok=True)

    # Initialize SAM model — use vit_b (12 blocks, stable on torch 2.9+ / Ampere GPUs)
    # vit_h (32 blocks) produces NaN due to numerical overflow in FP32
    print("Loading SAM model...")
    sam_type = "vit_b" if "vit_b" in sam_checkpoint else "vit_h"
    model_sam = sam_model_registry[sam_type](checkpoint=sam_checkpoint)
    model_sam.eval().to(device)
    print(f"SAM model ready ({sam_type}).")

    # Process each image
    clean_path = Path(clean_images_dir)
    image_files = sorted(list(clean_path.glob("*.jpg")) + list(clean_path.glob("*.png")))

    label_mode_char = 'a' if label_mode == 'Alphabet' else '1'
    text_size = 900
    anno_mode = ['Mask', 'Mark']

    print(f"Processing {len(image_files)} images...")
    for img_file in image_files:
        try:
            image = Image.open(img_file).convert('RGB')

            # Run SAM segmentation (no autocast — FP16 causes NaN on some GPUs)
            with torch.no_grad():
                output, anns = inference_sam_m2m_auto(
                    model_sam, image, text_size, label_mode_char, alpha, anno_mode
                )

            # Save SoM annotated image
            output_path = Path(som_output_dir) / img_file.name
            Image.fromarray(output).save(output_path)

            # Compute and save label coordinates (normalized)
            labels = []
            H, W = anns[0]["segmentation"].shape
            for ann in anns:
                cy, cx = center_of_mass(ann["segmentation"])
                x_norm = (2 * cx / W) - 1  # Normalize to [-1, 1]
                y_norm = (2 * cy / H) - 1
                labels.append((x_norm, y_norm))

            label_filename = img_file.stem + '.txt'
            label_path = Path(labels_output_dir) / label_filename
            with open(label_path, 'w') as f:
                for x_norm, y_norm in labels:
                    f.write(f"{x_norm:.4f}, {y_norm:.4f}\n")

            print(f"  ✓ {img_file.name} ({len(labels)} segments)")

        except Exception as e:
            print(f"  ✗ {img_file.name}: {e}")

    print(f"SoM generation complete. Output: {som_output_dir}")


GPT_PROMPT_GENERAL = """You will receive two input images:
The first image is the original image;
The second image is the semantic segmentation map (SoM image) of the original image, where each identifiable region has a unique numeric label displayed at the center of the mask.

Your task: select the single best region for placing an adversarial patch overlay.

Selection criteria (in order of priority):
1. The region should belong to the most prominent, visually dominant object in the image.
2. The region must have a clearly visible numeric label in the segmentation image.
3. Prefer large, flat, unoccluded surfaces — the patch should be clearly visible.
4. Prefer regions with a frontal or side-on view (not heavily foreshortened).
5. If one object spans multiple labeled regions, choose the region with the largest area.
6. Avoid background regions (sky, ground, distant scenery). Prefer foreground objects.

Only select the single best region.
Return format: {"label": <number>} — output only this JSON object, nothing else."""

GPT_PROMPT_DRIVING = """You will receive two input images:
The first image is the original scene;
The second image is the semantic segmentation map (SOM image) of the original scene, where each identifiable region has a unique numeric label displayed at the center of the mask.

Please analyze the images according to the following requirements and output the region ID that best meets the criteria:

- First, identify areas in the original image that are suitable for pasting or painting artistic advertisements.
- Then, locate the corresponding region in the SOM image and find its label number.
- The selected region must have a clearly visible numeric label in the segmentation image.
- It is allowed to select the surface of vehicles as the painting area, especially those parked near the edge.
- The selected region should be prominent in the image, flat, and easy to paint on.
- Prioritize objects with a frontal or side-on view.
- The object must not be occluded — the painting surface should be clearly visible and unobstructed.
- If an entity (such as a vehicle) consists of multiple labeled regions, choose the region with the largest area as the representative center label.
- The selected object should occupy a relatively large area in the current camera view, ensuring it is clearly visible and impactful for advertisement placement.
- Prefer selecting large-looking buildings, vehicles, or other structures that appear visually dominant from the current camera perspective.

Only select the single best region.
Return format: {"label": <center label>} — output only this JSON object."""


def generate_coordinates_with_gpt(
    clean_images_dir: str,
    som_dir: str,
    labels_dir: str,
    output_coords_file: str,
    api_key: str,
    api_base: Optional[str] = None,
    model: str = "gpt-5-mini-2025-08-07",
    prompt_style: str = "general",
) -> List[Tuple[str, float, float]]:
    """
    Use GPT-4V to select optimal patch placement coordinates.

    Based on legacy/code/som_gpt.py (updated for OpenAI API 1.0+)

    Args:
        clean_images_dir: Directory with original clean images
        som_dir: Directory with SoM annotated images
        labels_dir: Directory with label coordinate files
        output_coords_file: Output path for final coordinates
        api_key: OpenAI API key
        api_base: Optional custom API base URL
        model: Model name (default: gpt-5-mini-2025-08-07)
        prompt_style: "general" for arbitrary images, "driving" for autonomous driving scenes

    Returns:
        List of (stem, x, y) coordinate tuples
    """
    from openai import OpenAI

    # Initialize OpenAI client (new API 1.0+)
    client = OpenAI(api_key=api_key, base_url=api_base) if api_base else OpenAI(api_key=api_key)

    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    prompt = GPT_PROMPT_DRIVING if prompt_style == "driving" else GPT_PROMPT_GENERAL

    # Get sorted image lists
    clean_images = sorted([
        os.path.join(clean_images_dir, f)
        for f in os.listdir(clean_images_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    som_images = sorted([
        os.path.join(som_dir, f)
        for f in os.listdir(som_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    # coordinates stored as list of (image_stem, x, y) — includes stem so
    # skipped/failed entries don't cause misalignment downstream.
    coordinates = []  # [(stem, x, y), ...]
    print(f"Running GPT-4V coordinate selection on {len(clean_images)} images...")

    for i, (clean_path, som_path) in enumerate(zip(clean_images, som_images)):
        image_filename = os.path.basename(clean_path)
        image_stem = os.path.splitext(image_filename)[0]
        try:
            # Encode images
            base64_clean = encode_image(clean_path)
            base64_som = encode_image(som_path)

            # Query GPT-4V (new API 1.0+)
            response = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_clean}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_som}"}}
                    ]
                }],
                temperature=1,
                max_completion_tokens=512
            )

            raw_output = response.choices[0].message.content

            # Parse JSON response
            match = re.search(r'\{.*?\}', raw_output)
            if match:
                json_str = match.group(0)
                data = json.loads(json_str)
                label_index = data.get("label")

                if isinstance(label_index, int):
                    # Look up coordinate from label file
                    txt_filename = image_stem + ".txt"
                    txt_path = os.path.join(labels_dir, txt_filename)

                    if os.path.exists(txt_path):
                        with open(txt_path, 'r') as f:
                            lines = f.readlines()

                        if 1 <= label_index <= len(lines):
                            coord_line = lines[label_index - 1].strip()
                            x_str, y_str = coord_line.split(',')
                            x = float(x_str.strip())
                            y = float(y_str.strip())
                            coordinates.append((image_stem, x, y))
                            print(f"  ✓ {image_filename} → Label {label_index} → ({x:.4f}, {y:.4f})")
                        else:
                            print(f"  ✗ {image_filename}: Label {label_index} out of bounds (max {len(lines)})")
                    else:
                        print(f"  ✗ {image_filename}: Label file not found")
                else:
                    print(f"  ✗ {image_filename}: Non-integer label: {label_index}")
            else:
                print(f"  ✗ {image_filename}: No JSON in response: {raw_output[:100]}")

        except Exception as e:
            print(f"  ✗ {image_filename}: {e}")

    # Save coordinates to file — format: "image_stem, x, y" per line
    # This prevents misalignment when some images fail.
    os.makedirs(os.path.dirname(output_coords_file), exist_ok=True)
    with open(output_coords_file, 'w') as f:
        for stem, x, y in coordinates:
            f.write(f"{stem}, {x:.4f}, {y:.4f}\n")

    print(f"Saved {len(coordinates)} coordinates to {output_coords_file}")
    return coordinates


def ensure_coordinates(
    clean_images_dir: str,
    coords_file: str,
    som_dir: Optional[str] = None,
    sam_labels_dir: Optional[str] = None,
    sam_checkpoint: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    openai_api_base: Optional[str] = None,
    device: str = "cuda",
    prompt_style: str = "general",
) -> str:
    """
    Ensure coordinates exist, generating them if necessary.

    This is the main orchestration function that:
    1. Checks if coordinates exist → return path
    2. If not, checks if SoM/labels exist → run GPT-4V
    3. If not, generates SoM/labels → run GPT-4V

    Args:
        clean_images_dir: Directory with clean images (required)
        coords_file: Path to coordinates file
        som_dir: Directory for SoM images (optional, auto-inferred)
        sam_labels_dir: Directory for SAM labels (optional, auto-inferred)
        sam_checkpoint: Path to SAM checkpoint (optional, auto-inferred)
        openai_api_key: OpenAI API key (required for generation)
        openai_api_base: Optional custom API base
        device: "cuda" or "cpu"

    Returns:
        Path to coordinates file
    """
    # Auto-infer paths: som/ and sam_labels/ are siblings of the source dir
    dataset_root = Path(clean_images_dir).parent  # e.g., dataset/nips2017/ (parent of source/)

    if som_dir is None:
        som_dir = str(dataset_root / "som")
    if sam_labels_dir is None:
        sam_labels_dir = str(dataset_root / "sam_labels")
    if sam_checkpoint is None:
        sam_checkpoint = str(Path(__file__).parent / "assets" / "checkpoints" / "sam_vit_b_01ec64.pth")

    # Validate what exists
    status = validate_pipeline_data(
        clean_images_dir=clean_images_dir,
        coords_file=coords_file,
        som_dir=som_dir,
        sam_labels_dir=sam_labels_dir
    )

    print("=" * 70)
    print("PhysPatch Coordinate Pipeline Validation")
    print("=" * 70)
    print(f"Clean images:  {'✓' if status['clean_images_exist'] else '✗'} ({status['num_images']} images)")
    print(f"Coordinates:   {'✓' if status['coordinates_exist'] else '✗'}")
    print(f"SoM images:    {'✓' if status['som_exist'] else '✗'}")
    print(f"SAM labels:    {'✓' if status['labels_exist'] else '✗'}")
    print("=" * 70)

    # Case 1: Coordinates exist → use them
    if status["coordinates_exist"]:
        print("✓ Coordinates exist. Using existing file.")
        return coords_file

    # Case 2: Need to generate coordinates
    print("✗ Coordinates not found. Generating...")

    if not status["clean_images_exist"]:
        raise ValueError(f"Cannot generate coordinates: No clean images found in {clean_images_dir}")

    if openai_api_key is None:
        raise ValueError(
            "Cannot generate coordinates: OpenAI API key required. "
            "Please provide via --openai_api_key or set OPENAI_API_KEY environment variable."
        )

    # Case 2a: SoM/labels missing → generate them first
    if not status["som_exist"] or not status["labels_exist"]:
        print("\n[1/2] Generating SoM annotations and labels...")

        if not Path(sam_checkpoint).exists():
            raise ValueError(f"SAM checkpoint not found: {sam_checkpoint}")

        generate_som_labels(
            clean_images_dir=clean_images_dir,
            som_output_dir=som_dir,
            labels_output_dir=sam_labels_dir,
            sam_checkpoint=sam_checkpoint,
            device=device
        )
        print("✓ SoM generation complete.\n")

    # Case 2b: Run GPT-4V coordinate selection
    print("[2/2] Running GPT-4V coordinate selection...")
    generate_coordinates_with_gpt(
        clean_images_dir=clean_images_dir,
        som_dir=som_dir,
        labels_dir=sam_labels_dir,
        output_coords_file=coords_file,
        api_key=openai_api_key,
        api_base=openai_api_base,
        prompt_style=prompt_style,
    )
    print("✓ Coordinate generation complete.\n")

    return coords_file


if __name__ == "__main__":
    """
    Standalone usage for coordinate generation.

    Example:
        python coordinate_generator.py \
            --clean_images dataset/physpatch/images/clean \
            --coords_file dataset/physpatch/coordinates/full.txt \
            --openai_api_key YOUR_KEY
    """
    import argparse

    parser = argparse.ArgumentParser(description="PhysPatch coordinate generation pipeline")
    parser.add_argument('--clean_images', required=True, help='Clean images directory')
    parser.add_argument('--coords_file', required=True, help='Output coordinates file')
    parser.add_argument('--som_dir', help='SoM images directory (auto-inferred if not provided)')
    parser.add_argument('--labels_dir', help='SAM labels directory (auto-inferred if not provided)')
    parser.add_argument('--sam_checkpoint', help='SAM checkpoint path (auto-inferred if not provided)')
    parser.add_argument('--openai_api_key', help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--openai_api_base', help='Optional custom OpenAI API base URL')
    parser.add_argument('--prompt_style', default='general', choices=['general', 'driving'],
                        help='GPT prompt style: "general" for arbitrary images, "driving" for autonomous driving scenes')
    parser.add_argument('--device', default='cuda', help='Device: cuda or cpu')

    args = parser.parse_args()

    # Get API key from env if not provided
    api_key = args.openai_api_key or os.environ.get('OPENAI_API_KEY')

    ensure_coordinates(
        clean_images_dir=args.clean_images,
        coords_file=args.coords_file,
        som_dir=args.som_dir,
        sam_labels_dir=args.labels_dir,
        sam_checkpoint=args.sam_checkpoint,
        openai_api_key=api_key,
        openai_api_base=args.openai_api_base,
        device=args.device,
        prompt_style=args.prompt_style,
    )
