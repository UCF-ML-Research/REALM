"""
GPT-5-mini Vision Annotator for Background Region Selection

This module uses OpenAI's GPT-5-mini vision model to identify background regions
for semantic injection, simulating the "manual selection" process described in the paper:

"We manually select contiguous patches of size m×m from the image, typically located
in background regions or areas with fewer foreground objects."

Paper-faithful automation: GPT-5-mini acts as a consistent "human annotator".
"""

import base64
import io
import json
import re
import os
from typing import Tuple, Optional
from PIL import Image


def gpt_annotate_background_region(
    image_path: str,
    region_size: int = 100,
    api_key: Optional[str] = None,
    model: str = "gpt-5-mini-2025-08-07",
    image_size: int = 336
) -> Tuple[int, int, int, int]:
    """
    Use GPT-5-mini to identify background region for semantic injection.

    This simulates the paper's "manual selection" process in a scalable,
    reproducible manner.

    Args:
        image_path: Path to input image
        region_size: Desired region size (e.g., 100x100 pixels)
        api_key: OpenAI API key (if None, reads from OPENAI_API_KEY env var)
        model: Model to use (default: gpt-5-mini-2025-08-07)
        image_size: Target image size after preprocessing (default: 336)

    Returns:
        Bounding box (x, y, w, h) in pixel coordinates

    Raises:
        ImportError: If openai package not installed
        ValueError: If API call fails or response cannot be parsed
    """
    try:
        import openai
    except ImportError:
        raise ImportError(
            "OpenAI package not installed. Install with: pip install openai"
        )

    # Get API key
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError(
                "OpenAI API key not provided. Either pass api_key parameter or "
                "set OPENAI_API_KEY environment variable."
            )

    # Load and encode image
    print(f"Loading image for GPT annotation: {image_path}")
    with Image.open(image_path) as img:
        original_size = img.size
        print(f"  Original size: {original_size}")

        # Resize to target size (matching CLIP preprocessing)
        if img.size != (image_size, image_size):
            img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
            print(f"  Resized to: {img.size}")

        # Convert to base64
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

    # Create prompt (paper-faithful)
    prompt = f"""Analyze this image and identify a background region suitable for adversarial semantic injection.

**Task**: Select a contiguous rectangular region following these criteria (from the paper):
1. Size: Approximately {region_size}×{region_size} pixels
2. Location: Background regions or areas with FEWER FOREGROUND OBJECTS
3. Avoid: Main subjects, faces, salient objects, text, lane markings, or important details
4. Prefer (in order):
   - Sky regions (top of image)
   - Empty walls or uniform backgrounds
   - Blurred distant backgrounds
   - Ground/road ONLY if uniform with no markings

**Image info**:
- Dimensions: {image_size}×{image_size} pixels
- Coordinate system: Top-left is (0, 0)

**Output format** (JSON only, no markdown):
{{
  "x": <left coordinate, 0 to {image_size - region_size}>,
  "y": <top coordinate, 0 to {image_size - region_size}>,
  "width": {region_size},
  "height": {region_size},
  "reasoning": "<brief explanation of why this region is background/low-saliency>"
}}

Ensure the bbox (x, y, width, height) is fully within image bounds."""

    # Call OpenAI API
    print(f"Calling {model} for region annotation...")
    client = openai.OpenAI(api_key=api_key)

    try:
        # Use max_completion_tokens for newer models (gpt-5-mini, etc.)
        # Note: gpt-5-mini only supports default temperature=1
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_base64}",
                                "detail": "high"  # High detail for better region selection
                            }
                        }
                    ]
                }
            ],
            max_completion_tokens=1500  # High for reasoning models like gpt-5-mini
            # temperature is not set - gpt-5-mini only supports default value (1)
        )
    except Exception as e:
        raise ValueError(f"OpenAI API call failed: {str(e)}")

    # Parse response
    response_text = response.choices[0].message.content
    print(f"GPT response:\n{response_text}\n")

    # Extract JSON (handle potential markdown formatting)
    json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
    if not json_match:
        raise ValueError(f"Failed to extract JSON from GPT response: {response_text}")

    try:
        result = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}\nResponse: {response_text}")

    # Extract bbox
    x = int(result['x'])
    y = int(result['y'])
    w = int(result['width'])
    h = int(result['height'])
    reasoning = result.get('reasoning', 'No reasoning provided')

    # Validate bounds
    if x < 0 or y < 0 or x + w > image_size or y + h > image_size:
        print(f"⚠️  GPT returned out-of-bounds bbox, clipping to valid range")
        x = max(0, min(x, image_size - region_size))
        y = max(0, min(y, image_size - region_size))
        w = min(w, image_size - x)
        h = min(h, image_size - y)

    print(f"✓ GPT-annotated background region: ({x}, {y}, {w}, {h})")
    print(f"  Reasoning: {reasoning}")

    return (x, y, w, h)


def save_annotations(annotations: dict, output_path: str):
    """
    Save annotations to JSON file.

    Args:
        annotations: Dictionary mapping image_path -> (x, y, w, h)
        output_path: Path to save JSON file
    """
    # Convert tuples to lists for JSON serialization
    annotations_serializable = {
        k: list(v) if v is not None else None
        for k, v in annotations.items()
    }

    with open(output_path, 'w') as f:
        json.dump(annotations_serializable, f, indent=2)

    print(f"✓ Annotations saved to: {output_path}")


def load_annotations(input_path: str) -> dict:
    """
    Load annotations from JSON file.

    Args:
        input_path: Path to JSON file

    Returns:
        Dictionary mapping image_path -> (x, y, w, h)
    """
    with open(input_path, 'r') as f:
        annotations_loaded = json.load(f)

    # Convert lists back to tuples
    annotations = {
        k: tuple(v) if v is not None else None
        for k, v in annotations_loaded.items()
    }

    print(f"✓ Loaded {len(annotations)} annotations from: {input_path}")
    return annotations


def gpt_annotate_batch(
    image_paths: list,
    region_size: int = 100,
    api_key: Optional[str] = None,
    model: str = "gpt-5-mini-2025-08-07",
    save_path: Optional[str] = None
) -> dict:
    """
    Annotate multiple images in batch and optionally save to file.

    IMPORTANT: Saves annotations to avoid re-running GPT and wasting money!

    Args:
        image_paths: List of image paths
        region_size: Desired region size
        api_key: OpenAI API key
        model: Model to use (default: gpt-5-mini-2025-08-07)
        save_path: If provided, save annotations to this JSON file

    Returns:
        Dictionary mapping image_path -> (x, y, w, h)
    """
    annotations = {}

    print(f"Batch annotating {len(image_paths)} images with {model}...")
    for i, image_path in enumerate(image_paths, 1):
        print(f"\n[{i}/{len(image_paths)}] {image_path}")
        try:
            bbox = gpt_annotate_background_region(
                image_path, region_size, api_key, model
            )
            annotations[image_path] = bbox
        except Exception as e:
            print(f"❌ Failed to annotate {image_path}: {e}")
            annotations[image_path] = None

    # Save if path provided
    if save_path:
        save_annotations(annotations, save_path)

    return annotations


if __name__ == "__main__":
    # Test on sample image
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m utils.gpt_annotator <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    bbox = gpt_annotate_background_region(image_path, region_size=100)
    print(f"\nFinal bbox: {bbox}")
