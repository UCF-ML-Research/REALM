#!/usr/bin/env python3
"""
Prepare target images for Chain of Attack by replicating the stop sign reference.
"""

import argparse
from pathlib import Path
from PIL import Image


def prepare_target_images(reference_path: str, output_dir: str, count: int = 100):
    """
    Replicate a reference image N times with sequential naming.

    Args:
        reference_path: Path to the reference image (e.g., stop_sign.png)
        output_dir: Directory to save replicated images
        count: Number of copies to create (default: 100)
    """
    reference_path = Path(reference_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load reference image
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference image not found: {reference_path}")

    print(f"Loading reference image: {reference_path}")
    reference_img = Image.open(reference_path)

    # Convert to RGB if needed (for PNG with alpha channel)
    if reference_img.mode in ("RGBA", "LA", "P"):
        reference_img = reference_img.convert("RGB")

    # Replicate
    print(f"Replicating to {count} images in {output_dir}")
    for i in range(1, count + 1):
        output_path = output_dir / f"{i:05d}.jpg"
        reference_img.save(output_path)
        if i % 10 == 0:
            print(f"  Created {i}/{count} images")

    print(f"✓ Successfully created {count} target images")


def main():
    parser = argparse.ArgumentParser(description="Prepare target images for CoA attack")
    parser.add_argument(
        "--reference",
        type=str,
        required=True,
        help="Path to reference image (e.g., stop_sign.png)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for replicated images",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of copies to create (default: 100)",
    )

    args = parser.parse_args()
    prepare_target_images(args.reference, args.output, args.count)


if __name__ == "__main__":
    main()
