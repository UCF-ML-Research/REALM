"""Evaluation subpackage for VLM benchmark."""

# Adversarial attack evaluation
from .vlm_inference import VLMInference, VLMResponse
from .semantic_scorer import SemanticSimilarityScorer
from .metrics import calculate_asr, calculate_metrics, print_metrics, EvaluationMetrics
from .evaluator import AttackEvaluator

__all__ = [
    # Attack evaluation
    "VLMInference",
    "VLMResponse",
    "SemanticSimilarityScorer",
    "calculate_asr",
    "calculate_metrics",
    "print_metrics",
    "EvaluationMetrics",
    "AttackEvaluator",
]
