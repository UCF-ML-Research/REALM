"""Visual description extraction utilities.

Extracts object visual descriptions from DriveLM key_object_infos
for better GPT evaluation context.
"""

import re
from typing import Dict, List, Optional, Any, Tuple


def extract_camera_from_question(question: str) -> List[str]:
    """Extract camera names from coordinate patterns in question.

    Patterns like <c1,CAM_FRONT,0.5,0.5> reference specific cameras.

    Args:
        question: Question text with potential coordinate references

    Returns:
        List of camera names found (e.g., ['CAM_FRONT', 'CAM_BACK'])
    """
    # Pattern matches <c1,CAM_FRONT,0.5073,0.5778>
    pattern = r'<[^,]+,([^,]+),[^>]+>'
    cameras = re.findall(pattern, question)
    return list(set(cameras))  # Deduplicate


def extract_object_id_from_question(question: str) -> Optional[str]:
    """Extract object identifier prefix from question.

    Args:
        question: Question text

    Returns:
        Object ID prefix like "c1,CAM_FRONT" or None
    """
    # Pattern matches <c1,CAM_FRONT,767.5,513.3>
    match = re.search(r'<([^>]+)>', question)
    if match:
        object_id = match.group(1)
        # Return prefix: c1,CAM_FRONT
        parts = object_id.split(',')
        if len(parts) >= 2:
            return ','.join(parts[:2])
    return None


def extract_visual_description(
    question: str,
    key_object_infos: Dict[str, Any],
) -> Optional[str]:
    """Extract visual description for object referenced in question.

    Based on DriveBench methodology for providing object context
    to GPT evaluation.

    Args:
        question: Question text with object reference
        key_object_infos: Dictionary of key objects with visual descriptions

    Returns:
        Visual description string or None
    """
    if not key_object_infos:
        return None

    # Special case: planning safety questions don't need object description
    if question == 'In this scenario, what are safe actions to take for the ego vehicle?':
        return "No visual description needed for this question."

    # Extract object ID prefix from question
    object_id_prefix = extract_object_id_from_question(question)
    if not object_id_prefix:
        return None

    # Find matching object in key_object_infos
    # Keys are like "<c1,CAM_FRONT,767.5,513.3>"
    for key, obj_info in key_object_infos.items():
        # Check if key starts with the object ID prefix
        if key.startswith(f"<{object_id_prefix}"):
            if isinstance(obj_info, dict) and 'Visual_description' in obj_info:
                return obj_info['Visual_description'].lower()
            elif isinstance(obj_info, dict) and 'visual_description' in obj_info:
                return obj_info['visual_description'].lower()

    return None


def format_object_context(
    key_object_infos: Dict[str, Any],
    cameras: Optional[List[str]] = None,
) -> str:
    """Format all object information for context.

    Args:
        key_object_infos: Dictionary of key objects
        cameras: Optional filter for specific cameras

    Returns:
        Formatted string with object context
    """
    if not key_object_infos:
        return ""

    lines = []
    for key, obj_info in key_object_infos.items():
        if not isinstance(obj_info, dict):
            continue

        # Extract camera from key
        match = re.search(r'<[^,]+,([^,]+),', key)
        obj_camera = match.group(1) if match else None

        # Filter by camera if specified
        if cameras and obj_camera and obj_camera not in cameras:
            continue

        # Build description
        category = obj_info.get('Category', obj_info.get('category', 'Unknown'))
        status = obj_info.get('Status', obj_info.get('status', ''))
        visual_desc = obj_info.get('Visual_description', obj_info.get('visual_description', ''))
        bbox = obj_info.get('2d_bbox', obj_info.get('bbox', ''))

        desc_parts = [f"- {category}"]
        if status:
            desc_parts.append(f"({status})")
        if visual_desc:
            desc_parts.append(f": {visual_desc}")
        if obj_camera:
            desc_parts.append(f"[{obj_camera}]")

        lines.append(' '.join(desc_parts))

    return '\n'.join(lines) if lines else ""


def get_object_by_id(
    object_id: str,
    key_object_infos: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Get object info by ID prefix.

    Args:
        object_id: Object ID like "c1" or "c1,CAM_FRONT"
        key_object_infos: Dictionary of key objects

    Returns:
        Object info dictionary or None
    """
    if not key_object_infos:
        return None

    for key, obj_info in key_object_infos.items():
        if key.startswith(f"<{object_id}"):
            return obj_info

    return None


def filter_images_by_question(
    image_paths: Dict[str, str],
    question: str,
) -> Dict[str, str]:
    """Filter image paths to only include cameras referenced in question.

    Based on DriveBench preprocessing - only load images for cameras
    that are actually referenced in the question.

    Args:
        image_paths: Dictionary of camera -> image path
        question: Question text

    Returns:
        Filtered dictionary with only relevant cameras
    """
    cameras = extract_camera_from_question(question)

    if not cameras:
        # No specific cameras referenced, return all
        return image_paths

    # Filter to only referenced cameras
    return {cam: path for cam, path in image_paths.items() if cam in cameras}


def parse_coordinate_reference(text: str) -> List[Tuple[str, str, float, float]]:
    """Parse all coordinate references from text.

    Args:
        text: Text with coordinate patterns like <c1,CAM_FRONT,0.5,0.5>

    Returns:
        List of (object_id, camera, x, y) tuples
    """
    pattern = r'<([^,]+),([^,]+),([^,]+),([^>]+)>'
    matches = re.findall(pattern, text)

    results = []
    for match in matches:
        obj_id, camera, x, y = match
        try:
            results.append((obj_id, camera, float(x), float(y)))
        except ValueError:
            continue

    return results


def denormalize_coordinates(
    x: float,
    y: float,
    image_width: int = 1600,
    image_height: int = 900,
) -> Tuple[int, int]:
    """Convert normalized coordinates to pixel coordinates.

    Args:
        x: Normalized x coordinate (0-1)
        y: Normalized y coordinate (0-1)
        image_width: Image width in pixels
        image_height: Image height in pixels

    Returns:
        (pixel_x, pixel_y) tuple
    """
    return (int(x * image_width), int(y * image_height))
