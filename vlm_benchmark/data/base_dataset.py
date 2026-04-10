"""
Base dataset module for VLM benchmark.

Defines the unified Sample dataclass and BaseDataset abstract class.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Iterator
from PIL import Image
import json


@dataclass
class Sample:
    """Unified sample format for VLM benchmark.

    Attributes:
        id: Unique identifier for the sample
        images: List of PIL Images (single or multiple)
        question: The question/prompt for the model
        ground_truth: Expected answer
        task_type: Type of task (e.g. 'qa', 'attack')
        canbus_data: Optional sensor data (driving-specific)
        key_objects: Optional annotated key objects in scene (driving-specific)
        tag: Optional evaluation tag
        is_safety_critical: Whether this is a safety-critical scenario
        safety_category: Category of safety concern
        metadata: Additional sample metadata
    """
    id: str
    images: List[Image.Image]
    question: str
    ground_truth: str
    task_type: str
    canbus_data: Optional[Dict[str, Any]] = None
    key_objects: Optional[Dict[str, Any]] = None
    tag: Optional[List[int]] = None
    is_safety_critical: bool = False
    safety_category: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Automatically classify safety-critical status if not already set."""
        # Only auto-classify if not explicitly set
        if not self.is_safety_critical and self.safety_category is None:
            try:
                from ..metrics.safety_metrics import classify_safety_critical
                is_critical, category = classify_safety_critical(
                    question=self.question,
                    ground_truth=self.ground_truth,
                    task_type=self.task_type
                )
                self.is_safety_critical = is_critical
                self.safety_category = category
            except ImportError:
                # If safety metrics not available, keep defaults
                pass

    def to_dict(self) -> Dict[str, Any]:
        """Convert sample to dictionary (excluding images)."""
        return {
            "id": self.id,
            "question": self.question,
            "ground_truth": self.ground_truth,
            "task_type": self.task_type,
            "canbus_data": self.canbus_data,
            "key_objects": self.key_objects,
            "tag": self.tag,
            "is_safety_critical": self.is_safety_critical,
            "safety_category": self.safety_category,
            "metadata": self.metadata,
            "num_images": len(self.images),
        }

    def get_visual_description(self) -> Optional[str]:
        """Extract visual description for referenced object in question.

        Based on DriveBench methodology for GPT evaluation context.

        Returns:
            Visual description string or None
        """
        if not self.key_objects:
            return None

        try:
            from ..prompts.visual_description import extract_visual_description
            return extract_visual_description(self.question, self.key_objects)
        except ImportError:
            # Fallback: simple extraction
            import re
            match = re.search(r'<([^,]+),([^,]+),', self.question)
            if match:
                obj_prefix = f"<{match.group(1)},{match.group(2)}"
                for key, obj_info in self.key_objects.items():
                    if key.startswith(obj_prefix):
                        if isinstance(obj_info, dict):
                            return obj_info.get('Visual_description', obj_info.get('visual_description', ''))
            return None

    def get_object_context(self) -> str:
        """Get formatted context about all key objects.

        Returns:
            Formatted string with object descriptions
        """
        if not self.key_objects:
            return ""

        try:
            from ..prompts.visual_description import format_object_context
            return format_object_context(self.key_objects)
        except ImportError:
            # Fallback: simple formatting
            lines = []
            for key, obj_info in self.key_objects.items():
                if isinstance(obj_info, dict):
                    category = obj_info.get('Category', obj_info.get('category', 'Unknown'))
                    desc = obj_info.get('Visual_description', obj_info.get('visual_description', ''))
                    lines.append(f"- {category}: {desc}")
            return '\n'.join(lines)


# Backward-compatible alias
DrivingSample = Sample


@dataclass
class BehaviorLabel:
    """Parsed behavior label with direction and speed components.

    Direction classes (5):
        - going_straight
        - slightly_steering_left
        - slightly_steering_right
        - turning_left
        - turning_right

    Speed classes (5):
        - not_moving
        - slowly
        - normal
        - fast
        - very_fast
    """
    direction: str
    speed: str
    raw_text: str = ""

    DIRECTION_CLASSES = [
        "going_straight",
        "slightly_steering_left",
        "slightly_steering_right",
        "turning_left",
        "turning_right"
    ]

    SPEED_CLASSES = [
        "not_moving",
        "slowly",
        "normal",
        "fast",
        "very_fast"
    ]

    @classmethod
    def from_text(cls, text: str) -> "BehaviorLabel":
        """Parse behavior label from text."""
        text_lower = text.lower()

        # Parse direction
        if "turning left" in text_lower:
            direction = "turning_left"
        elif "turning right" in text_lower:
            direction = "turning_right"
        elif "steering to the left" in text_lower or "steering left" in text_lower:
            direction = "slightly_steering_left"
        elif "steering to the right" in text_lower or "steering right" in text_lower:
            direction = "slightly_steering_right"
        else:
            direction = "going_straight"

        # Parse speed
        if "not moving" in text_lower:
            speed = "not_moving"
        elif "very fast" in text_lower:
            speed = "very_fast"
        elif "fast" in text_lower:
            speed = "fast"
        elif "slowly" in text_lower:
            speed = "slowly"
        else:
            speed = "normal"

        return cls(direction=direction, speed=speed, raw_text=text)

    def to_text(self) -> str:
        """Convert to natural language text."""
        direction_map = {
            "going_straight": "going straight",
            "slightly_steering_left": "slightly steering to the left",
            "slightly_steering_right": "slightly steering to the right",
            "turning_left": "turning left",
            "turning_right": "turning right"
        }

        speed_map = {
            "not_moving": "not moving",
            "slowly": "driving slowly",
            "normal": "driving with normal speed",
            "fast": "driving fast",
            "very_fast": "driving very fast"
        }

        return f"The ego vehicle is {direction_map[self.direction]}. The ego vehicle is {speed_map[self.speed]}."


class BaseDataset(ABC):
    """Abstract base class for VLM benchmark datasets.

    All dataset implementations should inherit from this class and implement
    the required abstract methods.
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        task_type: str = "behavior",
        max_samples: Optional[int] = None,
        text_only: bool = False,
    ):
        """Initialize base dataset.

        Args:
            data_root: Root directory for dataset
            split: Dataset split ('train', 'val', 'test')
            task_type: Task type filter
            max_samples: Maximum number of samples to load (None for all)
            text_only: Use black images for visual grounding test (default: False)
        """
        self.data_root = data_root
        self.split = split
        self.task_type = task_type
        self.max_samples = max_samples
        self.text_only = text_only
        self.samples: List[Sample] = []

    @abstractmethod
    def load_data(self) -> None:
        """Load dataset from disk. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def get_sample(self, idx: int) -> Sample:
        """Get a single sample by index. Must be implemented by subclasses."""
        pass

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        """Get sample by index."""
        return self.get_sample(idx)

    def __iter__(self) -> Iterator[Sample]:
        """Iterate over all samples."""
        for i in range(len(self)):
            yield self.get_sample(i)

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        stats = {
            "total_samples": len(self.samples),
            "split": self.split,
            "task_type": self.task_type,
            "task_distribution": {},
            "safety_statistics": {
                "critical_count": 0,
                "critical_ratio": 0.0,
                "category_distribution": {},
            },
        }

        # Count task types and safety statistics
        critical_count = 0
        category_counts = {}

        for sample in self.samples:
            task = sample.task_type
            stats["task_distribution"][task] = stats["task_distribution"].get(task, 0) + 1

            # Count safety-critical samples
            if sample.is_safety_critical:
                critical_count += 1
                if sample.safety_category:
                    category_counts[sample.safety_category] = category_counts.get(sample.safety_category, 0) + 1

        # Update safety statistics
        total = len(self.samples)
        stats["safety_statistics"]["critical_count"] = critical_count
        stats["safety_statistics"]["critical_ratio"] = critical_count / total if total > 0 else 0.0
        stats["safety_statistics"]["category_distribution"] = category_counts

        return stats

    def filter_by_task(self, task_type: str) -> List[Sample]:
        """Filter samples by task type."""
        return [s for s in self.samples if s.task_type == task_type]

    def to_json(self, output_path: str) -> None:
        """Export dataset to JSON format."""
        data = [s.to_dict() for s in self.samples]
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

    def create_text_only_variant(self):
        """Create a text-only variant for visual grounding evaluation.

        Returns black images instead of real images to test if model
        truly uses visual information or just relies on text.

        Returns:
            New dataset instance with text_only=True

        Note:
            Subclasses should override to provide proper initialization.
            This base implementation raises NotImplementedError.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement create_text_only_variant()"
        )

    def _create_black_image(self, width: int = 1600, height: int = 900) -> Image:
        """Create a black placeholder image.

        Args:
            width: Image width (default: 1600)
            height: Image height (default: 900)

        Returns:
            Black PIL Image
        """
        return Image.new('RGB', (width, height), color='black')
