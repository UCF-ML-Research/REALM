"""
DriveBench dataset loader for robustness evaluation.

Loads DriveBench dataset with support for:
- Clean images (baseline)
- 15 pre-corrupted image types
- Text-only evaluation (visual grounding test)
- All 4 driving tasks (perception, prediction, planning, behavior)
"""

import os
import json
from typing import List, Optional, Dict, Any
from PIL import Image

from .base_dataset import BaseDataset, Sample, BehaviorLabel

# Image sizing configuration for DriveBench
# 448x448 provides optimal balance between:
# - Detail retention (4× better than 224×224)
# - Attention focus (7× fewer patches than 1600×900)
# - Patch alignment (448 ÷ 14 = 32, clean multiple)
# - VLM compatibility (InternVL standard size)
DRIVEBENCH_IMAGE_SIZE = (448, 448)


def letterbox_resize(img: Image.Image, target_size: tuple) -> Image.Image:
    """Resize image preserving aspect ratio with letterboxing.

    Instead of stretching 1600x900 to 448x448 (distortion), this function:
    - Scales image to fit within target_size while preserving aspect ratio
    - Adds gray padding (128, 128, 128) to fill remaining space
    - Centers the resized image within the target canvas

    Args:
        img: Input PIL Image
        target_size: Target (width, height) tuple

    Returns:
        PIL Image resized with letterboxing (no distortion)
    """
    target_w, target_h = target_size
    orig_w, orig_h = img.size

    # Calculate scale to fit within target while preserving aspect ratio
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)

    # Resize with high-quality LANCZOS filter
    resized = img.resize((new_w, new_h), Image.LANCZOS)

    # Create gray canvas and paste centered image
    result = Image.new('RGB', target_size, color=(128, 128, 128))
    paste_x, paste_y = (target_w - new_w) // 2, (target_h - new_h) // 2
    result.paste(resized, (paste_x, paste_y))

    return result


class DriveBenchDataset(BaseDataset):
    """Dataset loader for DriveBench robustness benchmark.

    Supports:
    - Clean evaluation (baseline performance)
    - Corruption evaluation (15 types of image degradation)
    - Text-only evaluation (black images for visual grounding test)

    Corruption types:
    - Weather: Rain, Fog, Snow
    - Lighting: Brightness, LowLight
    - Sensor: MotionBlur, ZoomBlur, LensObstacleCorruption
    - Compression: H256ABRCompression, ColorQuant, BitError
    - Occlusion: CameraCrash, FrameLost, WaterSplashCorruption, Saturate
    """

    CORRUPTION_TYPES = [
        "Rain", "Fog", "Snow",  # Weather
        "Brightness", "LowLight",  # Lighting
        "MotionBlur", "ZoomBlur", "LensObstacleCorruption",  # Sensor
        "H256ABRCompression", "ColorQuant", "BitError",  # Compression
        "CameraCrash", "FrameLost", "WaterSplashCorruption", "Saturate",  # Occlusion
    ]

    CAMERA_VIEWS = [
        "CAM_FRONT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT"
    ]

    def __init__(
        self,
        data_root: str,
        qa_json_path: str,
        corruption_type: Optional[str] = None,
        nuscenes_root: Optional[str] = None,
        split: str = "test",
        task_type: str = "all",
        camera_views: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
        text_only: bool = False,
    ):
        """Initialize DriveBench dataset.

        Args:
            data_root: Root directory for DriveBench data
            qa_json_path: Path to QA JSON file (drivebench-test.json or corruption-specific)
            corruption_type: Type of corruption (None for clean, or one of CORRUPTION_TYPES)
            nuscenes_root: Root directory for nuScenes images (for clean evaluation)
            split: Dataset split
            task_type: Task type filter ('perception', 'prediction', 'planning', 'behavior', 'all')
            camera_views: List of camera views to use
            max_samples: Maximum samples to load
            text_only: Use black images (visual grounding test)
        """
        super().__init__(data_root, split, task_type, max_samples, text_only=text_only)

        self.qa_json_path = qa_json_path
        self.corruption_type = corruption_type
        self.nuscenes_root = nuscenes_root or os.path.join(data_root, "nuscenes")
        self.camera_views = camera_views or list(self.CAMERA_VIEWS)

        # Corruption image directory
        if corruption_type and corruption_type != "clean":
            self.corruption_dir = os.path.join(data_root, corruption_type)
        else:
            self.corruption_dir = None

        # Track missing images
        self._missing_image_count = 0

        # Visual descriptions path
        # Try multiple possible locations
        possible_paths = [
            os.path.join(data_root, "visual_description.json"),  # In data dir
            os.path.join(os.path.dirname(data_root), "visual_description.json"),  # Up one level from data dir
            os.path.join(os.path.dirname(os.path.dirname(data_root)), "toolkit", "data", "visual_description.json"),  # Up two levels, then toolkit/data
            os.path.join(os.path.dirname(os.path.dirname(data_root)), "data", "visual_description.json"),  # Up two levels, then data
        ]
        self.visual_desc_path = None
        for path in possible_paths:
            if os.path.exists(path):
                self.visual_desc_path = path
                break
        self.visual_descriptions = {}

        # Load data
        self.raw_data: List[Dict] = []
        self.load_data()

    def load_data(self) -> None:
        """Load DriveBench dataset from JSON."""
        with open(self.qa_json_path, 'r') as f:
            self.raw_data = json.load(f)

        # Load visual descriptions if available
        if self.visual_desc_path and os.path.exists(self.visual_desc_path):
            with open(self.visual_desc_path, 'r') as f:
                self.visual_descriptions = json.load(f)
            print(f"✓ Loaded visual descriptions for {len(self.visual_descriptions)} scenes from {os.path.basename(self.visual_desc_path)}")
        else:
            print(f"⚠️  Visual descriptions not found - model will not have object context")

        # Process samples
        self._process_samples()

        # Apply max_samples limit
        if self.max_samples and len(self.samples) > self.max_samples:
            self.samples = self.samples[:self.max_samples]

    def _process_samples(self) -> None:
        """Process raw data into Sample objects."""
        for i, item in enumerate(self.raw_data):
            question_type = item.get('question_type', 'behavior')

            # Filter by task type
            if self.task_type != 'all' and question_type != self.task_type:
                continue

            sample_id = f"{item['scene_token']}_{item['frame_token']}_{i}"

            # Get tag
            tag = item.get('tag', [0])
            if isinstance(tag, int):
                tag = [tag]

            # Extract visual descriptions for this sample
            scene_token = item['scene_token']
            frame_token = item['frame_token']
            visual_desc = {}
            if scene_token in self.visual_descriptions:
                scene_data = self.visual_descriptions[scene_token]
                if 'key_frames' in scene_data and frame_token in scene_data['key_frames']:
                    key_object_infos = scene_data['key_frames'][frame_token].get('key_object_infos', {})
                    # Extract visual descriptions from key_object_infos
                    for obj_id, obj_data in key_object_infos.items():
                        if isinstance(obj_data, dict) and 'Visual_description' in obj_data:
                            visual_desc[obj_id] = obj_data['Visual_description']

            sample = Sample(
                id=sample_id,
                images=[],  # Loaded lazily
                question=item.get('question', ''),
                ground_truth=item.get('answer', ''),
                task_type=question_type,
                canbus_data=None,
                key_objects=None,
                tag=tag,
                metadata={
                    'scene_token': item['scene_token'],
                    'frame_token': item['frame_token'],
                    'image_paths': item.get('image_path', {}),
                    'corruption_type': self.corruption_type,
                    'text_only': self.text_only,
                    'visual_description': visual_desc,
                }
            )
            self.samples.append(sample)

    def get_sample(self, idx: int) -> Sample:
        """Get a single sample with images loaded.

        Args:
            idx: Sample index

        Returns:
            Sample with images loaded
        """
        sample = self.samples[idx]

        # Load images if not already loaded
        if not sample.images:
            image_paths = sample.metadata.get('image_paths', {})
            images = []
            loaded_views = []  # Track which views were actually loaded

            for view in self.camera_views:
                if view in image_paths:
                    img = self._load_image(image_paths[view], view)
                    images.append(img)
                    loaded_views.append(view)  # Record which view this image is from

            sample.images = images
            # Store loaded views in metadata for inference to use correct camera labels
            sample.metadata['loaded_camera_views'] = loaded_views

        return sample

    def _load_image(self, json_path: str, camera_view: str) -> Image.Image:
        """Load image with corruption support.

        Args:
            json_path: Original image path from JSON
            camera_view: Camera view name

        Returns:
            PIL Image (clean, corrupted, or black)
        """
        # Text-only mode: return black image at original nuScenes size
        if self.text_only:
            return Image.new('RGB', (1600, 900), color='black')

        # Extract filename from path
        filename = os.path.basename(json_path)

        # Try corruption directory first
        if self.corruption_dir:
            corrupt_path = os.path.join(self.corruption_dir, camera_view, filename)
            if os.path.exists(corrupt_path):
                img = Image.open(corrupt_path).convert('RGB')
                return img  # Keep original 1600x900 size for maximum detail

        # Try nuScenes samples directory
        nuscenes_path = os.path.join(self.nuscenes_root, "samples", camera_view, filename)
        if os.path.exists(nuscenes_path):
            img = Image.open(nuscenes_path).convert('RGB')
            return img  # Keep original 1600x900 size for maximum detail

        # Try resolving from json_path
        if 'samples' in json_path:
            parts = json_path.split('/')
            if 'samples' in parts:
                try:
                    idx = parts.index('samples')
                    rel_path = '/'.join(parts[idx:])
                    full_path = os.path.join(os.path.dirname(self.nuscenes_root), rel_path)
                    if os.path.exists(full_path):
                        img = Image.open(full_path).convert('RGB')
                        return img  # Keep original 1600x900 size for maximum detail
                except ValueError:
                    pass  # 'samples' not in list

        # Return placeholder for missing images at original nuScenes size
        self._missing_image_count += 1
        if self._missing_image_count <= 5:
            print(f"Warning: Missing image: {json_path} (view={camera_view})")
        elif self._missing_image_count == 6:
            print("Warning: Suppressing further missing image warnings...")
        return Image.new('RGB', (1600, 900), color='black')

    def get_corruption_types(self) -> List[str]:
        """Get list of available corruption types."""
        return self.CORRUPTION_TYPES

    def create_corruption_variant(self, corruption_type: str) -> 'DriveBenchDataset':
        """Create a variant of this dataset with different corruption.

        Args:
            corruption_type: Type of corruption to apply

        Returns:
            New DriveBenchDataset with the specified corruption
        """
        return DriveBenchDataset(
            data_root=self.data_root,
            qa_json_path=self.qa_json_path,
            corruption_type=corruption_type,
            nuscenes_root=self.nuscenes_root,
            split=self.split,
            task_type=self.task_type,
            camera_views=self.camera_views,
            max_samples=self.max_samples,
            text_only=False,
        )

    def create_text_only_variant(self) -> 'DriveBenchDataset':
        """Create a text-only variant for visual grounding test.

        Returns:
            New DriveBenchDataset with black images
        """
        return DriveBenchDataset(
            data_root=self.data_root,
            qa_json_path=self.qa_json_path,
            corruption_type=None,
            nuscenes_root=self.nuscenes_root,
            split=self.split,
            task_type=self.task_type,
            camera_views=self.camera_views,
            max_samples=self.max_samples,
            text_only=True,
        )

    def get_task_distribution(self) -> Dict[str, int]:
        """Get distribution of samples by task type.

        Returns:
            Dict with task type counts
        """
        distribution = {}
        for sample in self.samples:
            task = sample.task_type
            distribution[task] = distribution.get(task, 0) + 1
        return distribution

    @staticmethod
    def get_corruption_json_path(data_root: str, corruption_type: str) -> str:
        """Get the JSON path for a specific corruption type.

        Args:
            data_root: DriveBench data root
            corruption_type: Corruption type name

        Returns:
            Path to corruption-specific JSON file
        """
        # Map corruption type to JSON filename
        name_map = {
            "Rain": "rain.json",
            "Fog": "fog.json",
            "Snow": "snow.json",
            "Brightness": "bright.json",
            "LowLight": "lowlight.json",
            "MotionBlur": "motion.json",
            "ZoomBlur": "zoom.json",
            "LensObstacleCorruption": "lens.json",
            "H256ABRCompression": "h256.json",
            "ColorQuant": "colorquant.json",
            "BitError": "biterror.json",
            "CameraCrash": "camcrash.json",
            "FrameLost": "framelost.json",
            "WaterSplashCorruption": "water.json",
            "Saturate": "saturate.json",
        }

        json_name = name_map.get(corruption_type, "drivebench-test.json")
        return os.path.join(data_root, "data", json_name)
