"""
Robo2VLM dataset loader for robotic manipulation evaluation.

Loads Robo2VLM dataset with support for:
- Robot manipulation tasks (depth perception, task completion, direction prediction, etc.)
- Multiple robot dataset sources (droid, austin, fractal, berkeley, viola)
- Multiple-choice question answering format
- Single-image or multi-view scenarios
"""

import os
import json
import ast
from typing import List, Optional, Dict, Any
from PIL import Image

from .base_dataset import BaseDataset, Sample


class Robo2VLMDataset(BaseDataset):
    """Dataset loader for Robo2VLM robotic manipulation benchmark.

    Features:
    - 10,000 samples from 5 robot datasets
    - Question categories: depth perception, task completion, direction prediction,
      cross-view matching, goal recognition, trajectory description
    - Pure MCQ format (typically 5 choices)
    - Single or multi-view images (640×360)

    Dataset sources:
    - austin: 3,635 samples (36.4%)
    - droid: 3,489 samples (34.9%)
    - fractal20220817: 2,267 samples (22.7%)
    - berkeley: 578 samples (5.8%)
    - viola: 31 samples (0.3%)
    """

    # Question categories based on analysis
    QUESTION_CATEGORIES = [
        "depth_perception",       # Which point is closest to camera?
        "task_completion",        # Has robot completed the task?
        "direction_prediction",   # Which arrow shows next move?
        "cross_view_matching",    # Match point across camera views
        "goal_recognition",       # Which config shows goal state?
        "trajectory_description", # Which instruction describes trajectory?
        "other"                   # Other manipulation questions
    ]

    # Data sources
    DATA_SOURCES = ["droid", "austin", "fractal20220817", "berkeley", "viola"]

    def __init__(
        self,
        data_root: str,
        json_path: str,
        split: str = "test",
        task_type: str = "all",
        max_samples: Optional[int] = None,
        filter_source: Optional[str] = None,
        filter_category: Optional[str] = None,
    ):
        """Initialize Robo2VLM dataset.

        Args:
            data_root: Root directory for Robo2VLM data (contains images/ folder)
            json_path: Path to robo2vlm_subset.json file
            split: Dataset split (default: 'test', single split available)
            task_type: Task type filter (for compatibility, maps to question categories)
            max_samples: Maximum samples to load (None for all)
            filter_source: Filter by data source (droid, austin, etc.)
            filter_category: Filter by question category
        """
        super().__init__(data_root, split, task_type, max_samples, text_only=False)

        self.json_path = json_path
        self.filter_source = filter_source
        self.filter_category = filter_category

        # Image directory
        self.image_dir = os.path.join(data_root, "images")
        if not os.path.exists(self.image_dir):
            raise ValueError(f"Image directory not found: {self.image_dir}")

        # Track missing images
        self._missing_image_count = 0

        # Load data
        self.raw_data: List[Dict] = []
        self.load_data()

    def load_data(self) -> None:
        """Load Robo2VLM dataset from JSON."""
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"JSON file not found: {self.json_path}")

        with open(self.json_path, 'r') as f:
            self.raw_data = json.load(f)

        print(f"✓ Loaded {len(self.raw_data)} samples from {os.path.basename(self.json_path)}")

        # Process samples
        self._process_samples()

        # Apply max_samples limit
        if self.max_samples and len(self.samples) > self.max_samples:
            self.samples = self.samples[:self.max_samples]

        print(f"✓ Processed {len(self.samples)} samples after filtering")

    def _categorize_question(self, question: str) -> str:
        """Categorize question based on content.

        Args:
            question: Question text

        Returns:
            Category name from QUESTION_CATEGORIES
        """
        q_lower = question.lower()

        if 'closest to the camera' in q_lower or 'depth' in q_lower:
            return 'depth_perception'
        elif 'completed the task' in q_lower or 'successfully' in q_lower:
            return 'task_completion'
        elif 'direction' in q_lower and 'arrow' in q_lower:
            return 'direction_prediction'
        elif 'corresponding to the same 3d location' in q_lower or 'closest point in' in q_lower:
            return 'cross_view_matching'
        elif 'goal state' in q_lower or 'configuration shows' in q_lower:
            return 'goal_recognition'
        elif 'trajectory' in q_lower and 'describes' in q_lower:
            return 'trajectory_description'
        else:
            return 'other'

    def _parse_choices(self, choices_str) -> List[str]:
        """Parse choice string to list.

        Args:
            choices_str: String representation of choices (e.g., "['A', 'B', 'C']")
                        or already parsed list

        Returns:
            List of choice strings
        """
        # If already a list, return as-is
        if isinstance(choices_str, list):
            return [str(c) for c in choices_str]

        try:
            # Use ast.literal_eval for safe parsing
            choices = ast.literal_eval(choices_str)
            if isinstance(choices, list):
                return [str(c) for c in choices]
        except (ValueError, SyntaxError):
            # Fallback: try manual parsing
            pass

        # If parsing fails, return as-is (will be handled in formatting)
        return [choices_str]

    def _extract_source(self, sample_id: str) -> str:
        """Extract data source from sample ID.

        Args:
            sample_id: Sample ID (e.g., 'droid_task_name_12345_q1')

        Returns:
            Source name (droid, austin, fractal20220817, etc.)
        """
        parts = sample_id.split('_')
        if parts:
            source = parts[0]
            # Handle fractal with date
            if source == 'fractal20220817':
                return 'fractal20220817'
            elif source in self.DATA_SOURCES:
                return source
        return 'unknown'

    def _process_samples(self) -> None:
        """Process raw data into Sample objects."""
        for item in self.raw_data:
            sample_id = item.get('id', '')
            question = item.get('question', '')
            choices_str = item.get('choices', '[]')
            correct_answer = item.get('correct_answer', 0)
            image_file = item.get('image_file', '')

            # Extract source and categorize
            source = self._extract_source(sample_id)
            category = self._categorize_question(question)

            # Apply filters
            if self.filter_source and source != self.filter_source:
                continue
            if self.filter_category and category != self.filter_category:
                continue
            if self.task_type != 'all' and category != self.task_type:
                continue

            # Parse choices
            choices = self._parse_choices(choices_str)

            # Create ground truth as letter (matching official Robo2VLM)
            # Official implementation uses letter-based answers (A/B/C/D/E)
            choice_letter_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H'}
            if 0 <= correct_answer < len(choices):
                ground_truth = choice_letter_map.get(correct_answer, f"Choice {correct_answer}")
                # Store choice text in metadata for debugging
                choice_text = choices[correct_answer]
            else:
                ground_truth = f"Choice {correct_answer}"
                choice_text = ground_truth

            # Create sample
            # Use category as task_type for consistency with benchmark
            sample = Sample(
                id=sample_id,
                images=[],  # Loaded lazily
                question=question,
                ground_truth=ground_truth,
                task_type=category,  # Use question category as task type
                canbus_data=None,
                key_objects=None,
                tag=[0],  # Tag 0 for MCQ accuracy evaluation
                metadata={
                    'choices': choices,
                    'correct_answer': correct_answer,
                    'correct_answer_text': choice_text,  # Store text for debugging
                    'image_file': image_file,
                    'source': source,
                    'category': category,
                    'num_choices': len(choices),
                }
            )
            self.samples.append(sample)

    def get_sample(self, idx: int) -> Sample:
        """Get a single sample with image loaded.

        Args:
            idx: Sample index

        Returns:
            Sample with image loaded
        """
        sample = self.samples[idx]

        # Load image if not already loaded
        if not sample.images:
            image_file = sample.metadata.get('image_file', '')
            img = self._load_image(image_file)
            sample.images = [img]  # Single image for robot manipulation

        return sample

    def _load_image(self, image_file: str) -> Image.Image:
        """Load image from file.

        Args:
            image_file: Relative image path (e.g., 'images/img_000000.png')

        Returns:
            PIL Image
        """
        # Handle both absolute and relative paths
        if image_file.startswith('images/'):
            # Relative path
            image_path = os.path.join(self.data_root, image_file)
        else:
            # Try as absolute path first
            if os.path.exists(image_file):
                image_path = image_file
            else:
                # Try constructing path
                image_path = os.path.join(self.image_dir, os.path.basename(image_file))

        # Load image
        if os.path.exists(image_path):
            try:
                img = Image.open(image_path).convert('RGB')
                return img
            except Exception as e:
                self._missing_image_count += 1
                if self._missing_image_count <= 5:
                    print(f"Warning: Error loading image {image_path}: {e}")
                elif self._missing_image_count == 6:
                    print("Warning: Suppressing further image loading error warnings...")
        else:
            self._missing_image_count += 1
            if self._missing_image_count <= 5:
                print(f"Warning: Missing image: {image_path}")
            elif self._missing_image_count == 6:
                print("Warning: Suppressing further missing image warnings...")

        # Return placeholder for missing images (640×360 to match dataset)
        return Image.new('RGB', (640, 360), color='black')

    def get_source_distribution(self) -> Dict[str, int]:
        """Get distribution of samples by data source.

        Returns:
            Dict with source counts
        """
        distribution = {}
        for sample in self.samples:
            source = sample.metadata.get('source', 'unknown')
            distribution[source] = distribution.get(source, 0) + 1
        return distribution

    def get_category_distribution(self) -> Dict[str, int]:
        """Get distribution of samples by question category.

        Returns:
            Dict with category counts
        """
        distribution = {}
        for sample in self.samples:
            category = sample.metadata.get('category', 'unknown')
            distribution[category] = distribution.get(category, 0) + 1
        return distribution

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive dataset statistics.

        Returns:
            Dict with dataset statistics
        """
        stats = super().get_statistics()

        # Add Robo2VLM-specific statistics
        stats['source_distribution'] = self.get_source_distribution()
        stats['category_distribution'] = self.get_category_distribution()

        # Choice statistics
        choice_counts = [sample.metadata.get('num_choices', 0) for sample in self.samples]
        if choice_counts:
            stats['choice_statistics'] = {
                'min_choices': min(choice_counts),
                'max_choices': max(choice_counts),
                'avg_choices': sum(choice_counts) / len(choice_counts),
            }

        return stats

    def format_mcq_prompt(self, sample: Sample) -> str:
        """Format sample as MCQ prompt with choices.

        Args:
            sample: Sample

        Returns:
            Formatted prompt string with question + choices
        """
        choices = sample.metadata.get('choices', [])
        question = sample.question

        # Format choices as A, B, C, D, E...
        choice_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        formatted_choices = []
        for i, choice in enumerate(choices):
            if i < len(choice_letters):
                formatted_choices.append(f"{choice_letters[i]}. {choice}")
            else:
                formatted_choices.append(f"{i+1}. {choice}")

        # Combine question and choices
        prompt = f"{question}\n\n" + "\n".join(formatted_choices)
        return prompt

    def create_text_only_variant(self) -> 'Robo2VLMDataset':
        """Create text-only variant (not applicable for Robo2VLM).

        Robo2VLM does not have corruption support, so text-only variant
        simply returns a new instance with black images.

        Returns:
            New Robo2VLMDataset instance
        """
        # Create new instance with same parameters
        dataset = Robo2VLMDataset(
            data_root=self.data_root,
            json_path=self.json_path,
            split=self.split,
            task_type=self.task_type,
            max_samples=self.max_samples,
            filter_source=self.filter_source,
            filter_category=self.filter_category,
        )
        # Override image loading to return black images
        dataset._create_black_images = True
        return dataset

    def __repr__(self) -> str:
        """String representation of dataset."""
        return (
            f"Robo2VLMDataset(\n"
            f"  samples={len(self.samples)},\n"
            f"  split='{self.split}',\n"
            f"  task_type='{self.task_type}',\n"
            f"  filter_source='{self.filter_source}',\n"
            f"  filter_category='{self.filter_category}'\n"
            f")"
        )
