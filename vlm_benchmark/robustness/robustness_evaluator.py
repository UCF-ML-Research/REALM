"""
Robustness evaluator for VLM benchmark.

Implements DriveBench-style evaluation:
1. Clean evaluation - baseline performance
2. Corrupted evaluation - sensitivity to degradation
3. Text-only evaluation - expose fake visual grounding

A robust VLM should:
- Perform well on clean inputs
- Degrade gracefully on corrupted inputs
- Fail on text-only inputs (not guess from text cues)
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from PIL import Image

from .image_corruption import ImageCorruptor, CorruptionType


@dataclass
class RobustnessResult:
    """Results from robustness evaluation."""
    clean_accuracy: float
    corrupted_accuracy: Dict[str, float]
    text_only_accuracy: float

    # Key metrics
    robustness_score: float      # How well it handles corruption
    visual_grounding_score: float  # clean - text_only (higher = better grounding)
    avg_corruption_drop: float   # Average accuracy drop under corruption

    corruption_details: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clean_accuracy": self.clean_accuracy,
            "corrupted_accuracy": self.corrupted_accuracy,
            "text_only_accuracy": self.text_only_accuracy,
            "robustness_score": self.robustness_score,
            "visual_grounding_score": self.visual_grounding_score,
            "avg_corruption_drop": self.avg_corruption_drop,
            "corruption_details": self.corruption_details,
        }

    def __str__(self) -> str:
        lines = [
            "Robustness Evaluation Results:",
            f"  Clean Accuracy:          {self.clean_accuracy*100:.1f}%",
            f"  Text-Only Accuracy:      {self.text_only_accuracy*100:.1f}%",
            f"  Visual Grounding Score:  {self.visual_grounding_score*100:.1f}%",
            f"  Avg Corruption Drop:     {self.avg_corruption_drop*100:.1f}%",
            f"  Robustness Score:        {self.robustness_score*100:.1f}%",
            "",
            "  Corrupted Accuracy by Type:",
        ]
        for ctype, acc in self.corrupted_accuracy.items():
            lines.append(f"    {ctype}: {acc*100:.1f}%")
        return "\n".join(lines)


class RobustnessEvaluator:
    """Evaluate VLM robustness following DriveBench methodology.

    Tests:
    1. Clean images - baseline
    2. Corrupted images - robustness
    3. Text-only (black images) - visual grounding

    A good VLM should:
    - High clean accuracy
    - Graceful degradation under corruption
    - Low text-only accuracy (not guessing from text)
    """

    def __init__(
        self,
        corruption_types: Optional[List[CorruptionType]] = None,
        severity: int = 3,
    ):
        """Initialize robustness evaluator.

        Args:
            corruption_types: Types of corruption to test
            severity: Corruption severity (1-5)
        """
        self.corruption_types = corruption_types or ImageCorruptor.get_driving_relevant_corruptions()
        self.severity = severity
        self.corruptor = ImageCorruptor(severity=severity)

        # Results storage
        self.clean_results: List[bool] = []
        self.corrupted_results: Dict[str, List[bool]] = {
            c.value: [] for c in self.corruption_types
        }
        self.text_only_results: List[bool] = []

    def reset(self) -> None:
        """Reset all results."""
        self.clean_results = []
        self.corrupted_results = {c.value: [] for c in self.corruption_types}
        self.text_only_results = []

    def evaluate_sample(
        self,
        model,
        image: Image.Image,
        question: str,
        ground_truth: str,
        task_type: str = "behavior",
    ) -> Dict[str, Any]:
        """Evaluate a single sample across all conditions.

        Args:
            model: VLM model wrapper
            image: Clean input image
            question: Question to ask
            ground_truth: Expected answer
            task_type: Task type for evaluation

        Returns:
            Dict with results for each condition
        """
        results = {}

        # 1. Clean evaluation
        clean_output = self._run_inference(model, image, question, task_type)
        clean_correct = self._check_correct(clean_output, ground_truth, task_type)
        results["clean"] = {
            "output": clean_output,
            "correct": clean_correct,
        }
        self.clean_results.append(clean_correct)

        # 2. Corrupted evaluations
        results["corrupted"] = {}
        for ctype in self.corruption_types:
            corrupted_image = self.corruptor.corrupt(image, ctype)
            corrupt_output = self._run_inference(model, corrupted_image, question, task_type)
            corrupt_correct = self._check_correct(corrupt_output, ground_truth, task_type)
            results["corrupted"][ctype.value] = {
                "output": corrupt_output,
                "correct": corrupt_correct,
            }
            self.corrupted_results[ctype.value].append(corrupt_correct)

        # 3. Text-only evaluation
        black_image = self.corruptor.corrupt(image, CorruptionType.TEXT_ONLY)
        text_only_output = self._run_inference(model, black_image, question, task_type)
        text_only_correct = self._check_correct(text_only_output, ground_truth, task_type)
        results["text_only"] = {
            "output": text_only_output,
            "correct": text_only_correct,
        }
        self.text_only_results.append(text_only_correct)

        return results

    def _run_inference(
        self,
        model,
        image: Image.Image,
        question: str,
        task_type: str,
    ) -> str:
        """Run model inference."""
        if task_type == "behavior":
            # Use behavior-specific inference
            if hasattr(model, 'inference_behavior'):
                output = model.inference_behavior(image, use_hybrid=False)
                return output.text
        # Generic inference
        output = model.inference([image], question)
        return output.text

    def _check_correct(
        self,
        prediction: str,
        ground_truth: str,
        task_type: str,
    ) -> bool:
        """Check if prediction is correct."""
        if task_type == "behavior":
            # Check direction and speed separately
            from ..metrics.accuracy_metrics import BehaviorAccuracy
            eval = BehaviorAccuracy()
            eval.add_prediction(prediction, ground_truth)
            result = eval.compute()
            # Consider correct if both direction and speed match
            return result.direction_accuracy > 0.5 and result.speed_accuracy > 0.5

        # Simple text matching for other tasks
        pred_clean = prediction.lower().strip().rstrip('.')
        gt_clean = ground_truth.lower().strip().rstrip('.')
        return pred_clean == gt_clean or pred_clean in gt_clean or gt_clean in pred_clean

    def compute(self) -> RobustnessResult:
        """Compute robustness metrics.

        Returns:
            RobustnessResult with all metrics
        """
        # Clean accuracy
        clean_acc = sum(self.clean_results) / len(self.clean_results) if self.clean_results else 0.0

        # Corrupted accuracy by type
        corrupted_acc = {}
        for ctype, results in self.corrupted_results.items():
            corrupted_acc[ctype] = sum(results) / len(results) if results else 0.0

        # Text-only accuracy
        text_only_acc = sum(self.text_only_results) / len(self.text_only_results) if self.text_only_results else 0.0

        # Visual grounding score: clean - text_only
        # Higher = model actually uses images, not just text cues
        visual_grounding = clean_acc - text_only_acc

        # Average corruption drop
        avg_corrupted = sum(corrupted_acc.values()) / len(corrupted_acc) if corrupted_acc else 0.0
        avg_corruption_drop = clean_acc - avg_corrupted

        # Robustness score: how well it maintains performance
        # 1.0 means no drop, 0.0 means complete failure
        robustness = avg_corrupted / clean_acc if clean_acc > 0 else 0.0

        return RobustnessResult(
            clean_accuracy=clean_acc,
            corrupted_accuracy=corrupted_acc,
            text_only_accuracy=text_only_acc,
            robustness_score=robustness,
            visual_grounding_score=visual_grounding,
            avg_corruption_drop=avg_corruption_drop,
            corruption_details={
                ctype: {
                    "accuracy": acc,
                    "drop": clean_acc - acc,
                }
                for ctype, acc in corrupted_acc.items()
            },
        )
