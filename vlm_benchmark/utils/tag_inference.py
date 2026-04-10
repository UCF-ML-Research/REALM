"""
Tag inference for DriveLM questions.

Automatically classifies questions into evaluation categories
following the DriveLM protocol when tags are not provided.

Tag System:
- Tag 0: Multi-choice / Yes-No questions -> Exact match accuracy
- Tag 1: Conversational questions -> GPT/semantic scoring
- Tag 2: Descriptive language questions -> BLEU/ROUGE/CIDEr
- Tag 3: Coordinate/Graph questions -> F1 matching + GPT
"""

import re
from typing import List, Optional, Tuple


class TagInference:
    """Infer evaluation tags from question/answer content."""

    # Patterns for Yes/No questions (Tag 0)
    YES_NO_PATTERNS = [
        r'^(is|are|does|do|can|will|would|should|has|have|was|were)\s',
        r'\?$',  # ends with question mark
    ]

    YES_NO_ANSWERS = ['yes', 'no', 'yes.', 'no.', 'true', 'false']

    # Patterns for coordinate/graph questions (Tag 3)
    COORDINATE_PATTERNS = [
        r'<c\d+,CAM_\w+,[\d.]+,[\d.]+>',  # DriveLM coordinate format
        r'\d+\.\d+,\s*\d+\.\d+',  # coordinate pairs
        r'bounding box',
        r'location',
        r'position',
        r'coordinates',
    ]

    # Patterns for behavior prediction (Tag 0 - classification)
    BEHAVIOR_PATTERNS = [
        r'predict the behavior',
        r'what is the ego vehicle doing',
        r'direction.*speed',
        r'going straight|turning|steering',
    ]

    # Patterns for descriptive/language questions (Tag 2)
    DESCRIPTIVE_PATTERNS = [
        r'^(what|describe|explain|how|why)',
        r'description',
        r'reason',
        r'because',
    ]

    @classmethod
    def infer_tag(
        cls,
        question: str,
        answer: str,
        task_type: Optional[str] = None
    ) -> List[int]:
        """Infer evaluation tag(s) from question and answer.

        DriveBench Tag System:
        - Tag 0: MCQ (Multi-choice / Yes-No questions) -> Accuracy
        - Tag 1: Planning VQA -> BLEU/ROUGE/CIDEr
        - Tag 2: Perception VQA -> BLEU/ROUGE/CIDEr
        - Tag 3: Prediction VQA -> BLEU/ROUGE/CIDEr

        Args:
            question: The question text
            answer: The ground truth answer
            task_type: Optional task type hint ('behavior', 'perception', 'prediction', 'planning')

        Returns:
            List of applicable tags (can have multiple)
        """
        question_lower = question.lower().strip()
        answer_lower = answer.lower().strip()
        tags = []

        # DriveBench-specific routing based on task_type
        if task_type:
            task_type_lower = task_type.lower()

            # Check if it's an MCQ (Tag 0)
            is_mcq = (cls._is_yes_no(question_lower, answer_lower) or
                     cls._is_multi_choice(answer_lower) or
                     'select the correct answer' in question_lower or
                     'please select' in question_lower or
                     re.search(r'(?:^|\n)\s*(?:\(?[A-D]\)?[\.\):])', question))

            if task_type_lower == 'behavior':
                # Behavior is always Tag 0 (MCQ accuracy)
                tags.append(0)
                return tags
            elif task_type_lower == 'perception':
                if is_mcq:
                    tags.append(0)  # Perception MCQ
                else:
                    tags.append(2)  # Perception VQA
                return tags
            elif task_type_lower == 'prediction':
                if is_mcq:
                    tags.append(0)  # Prediction MCQ
                else:
                    tags.append(3)  # Prediction VQA
                return tags
            elif task_type_lower == 'planning':
                # Planning is always Tag 1 (VQA)
                tags.append(1)
                return tags

        # Fallback logic for when task_type is not provided
        # Check for behavior prediction (classification task)
        if cls._matches_patterns(question_lower, cls.BEHAVIOR_PATTERNS):
            tags.append(0)  # Accuracy metric
            return tags

        # Check for coordinate/graph questions (Prediction VQA)
        if cls._has_coordinates(question) or cls._has_coordinates(answer):
            tags.append(3)  # Prediction VQA
            return tags

        # Check for Yes/No questions or multi-choice
        if cls._is_yes_no(question_lower, answer_lower) or cls._is_multi_choice(answer_lower):
            tags.append(0)  # MCQ accuracy
            return tags

        # Descriptive answers default to Perception VQA
        if cls._is_descriptive(question_lower, answer_lower):
            tags.append(2)  # Perception VQA
            return tags

        # Default: Planning VQA
        tags.append(1)
        return tags

    @classmethod
    def _matches_patterns(cls, text: str, patterns: List[str]) -> bool:
        """Check if text matches any pattern."""
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    @classmethod
    def _has_coordinates(cls, text: str) -> bool:
        """Check if text contains coordinate references."""
        for pattern in cls.COORDINATE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    @classmethod
    def _is_yes_no(cls, question: str, answer: str) -> bool:
        """Check if this is a Yes/No question."""
        # Check answer
        answer_clean = answer.strip().rstrip('.').lower()
        if answer_clean in ['yes', 'no', 'true', 'false']:
            return True

        # Check question patterns
        for pattern in cls.YES_NO_PATTERNS[:1]:  # Just check start patterns
            if re.match(pattern, question, re.IGNORECASE):
                return True

        return False

    @classmethod
    def _is_multi_choice(cls, answer: str) -> bool:
        """Check if answer is multi-choice style (short, definitive)."""
        words = answer.split()
        # Short answers (1-5 words) that aren't full sentences
        if len(words) <= 5 and not answer.endswith('.'):
            return True
        # Single letter/number answers
        if len(answer) <= 2:
            return True
        return False

    @classmethod
    def _is_descriptive(cls, question: str, answer: str) -> bool:
        """Check if this requires descriptive answer."""
        # Check question type
        if cls._matches_patterns(question, cls.DESCRIPTIVE_PATTERNS):
            return True
        # Long answers are descriptive
        if len(answer.split()) > 8:
            return True
        return False

    @classmethod
    def get_primary_tag(cls, tags: List[int]) -> int:
        """Get the primary (most important) tag from list."""
        # Priority: 0 (accuracy) > 3 (match) > 2 (language) > 1 (GPT)
        priority = [0, 3, 2, 1]
        for p in priority:
            if p in tags:
                return p
        return 1  # Default to GPT


def infer_question_tag(
    question: str,
    answer: str,
    task_type: Optional[str] = None
) -> Tuple[List[int], int]:
    """Convenience function to infer tags.

    Args:
        question: Question text
        answer: Ground truth answer
        task_type: Optional task type

    Returns:
        Tuple of (all_tags, primary_tag)
    """
    tags = TagInference.infer_tag(question, answer, task_type)
    primary = TagInference.get_primary_tag(tags)
    return tags, primary
