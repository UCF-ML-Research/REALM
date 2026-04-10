#!/usr/bin/env python3
"""
Generate clean captions for PhysPatch images using Qwen2.5-VL-7B-Instruct.
"""

import argparse
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor


def generate_clean_captions(
    images_dir: str,
    output_path: str,
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
    device: str = "cuda",
    test_one: bool = False,
):
    """
    Generate semantic descriptions for images using Qwen2.5-VL.

    Args:
        images_dir: Directory containing clean images (e.g., PhysPatch clean images)
        output_path: Path to save captions file (one line per image)
        model_name: Qwen model name
        device: Device to run inference on
        test_one: If True, only process first image for testing
    """
    images_dir = Path(images_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get sorted image paths
    image_paths = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    if not image_paths:
        raise ValueError(f"No images found in {images_dir}")

    if test_one:
        image_paths = image_paths[:1]
        print(f"Test mode: Processing only first image: {image_paths[0].name}")
    else:
        print(f"Found {len(image_paths)} images in {images_dir}")

    # Load Qwen3-VL model
    print(f"Loading model: {model_name}")
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_name)

    # Generate captions
    captions = []
    print(f"Generating captions...")

    for idx, img_path in enumerate(image_paths, 1):
        # Prepare message (Qwen3-VL format)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": str(img_path),
                    },
                    {
                        "type": "text",
                        "text": "Describe this image in one sentence, focusing on the main objects and scene.",
                    },
                ],
            }
        ]

        # Prepare inputs (Qwen3-VL simplified API)
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)

        # Generate caption
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            caption = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

        captions.append(caption.strip())

        if idx % 10 == 0:
            print(f"  {idx}/{len(image_paths)}: {img_path.name} -> {caption[:60]}...")

    # Save captions
    print(f"Saving captions to {output_path}")
    with open(output_path, "w") as f:
        for caption in captions:
            f.write(caption + "\n")

    print(f"✓ Successfully generated {len(captions)} captions")

    if test_one:
        print(f"\nTest caption: {captions[0]}")


def main():
    parser = argparse.ArgumentParser(description="Generate clean captions using Qwen2.5-VL")
    parser.add_argument(
        "--images-dir",
        type=str,
        default="/home/yifei/data/Red_Teaming/dataset/physpatch/images/clean",
        help="Directory containing clean images",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="vlm_benchmark/attacks/coa/assets/captions/clean_captions_qwen.txt",
        help="Output path for captions file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Qwen model name",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (cuda/cpu)",
    )
    parser.add_argument(
        "--test-one",
        action="store_true",
        help="Test mode: only process first image",
    )

    args = parser.parse_args()
    generate_clean_captions(
        args.images_dir,
        args.output,
        args.model,
        args.device,
        args.test_one,
    )


if __name__ == "__main__":
    main()
