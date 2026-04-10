"""Prompt templates for VLM benchmark.

Contains:
- System prompts for VLM inference
- Task-specific GPT evaluation prompts (DriveBench style)
- Visual description extraction utilities
"""

from .system_prompts import (
    DRIVING_SYSTEM_PROMPT,
    GENERAL_DRIVING_SYSTEM_PROMPT,
    BDDX_FEW_SHOT_PREFIX,
)

from .gpt_eval_prompts import (
    PERCEPTION_MCQ_PROMPT,
    PERCEPTION_VQA_PROMPT,
    PREDICTION_VQA_PROMPT,
    PLANNING_VQA_PROMPT,
    BEHAVIOR_MCQ_PROMPT,
    get_eval_prompt_for_task,
)

from .visual_description import (
    extract_visual_description,
    extract_camera_from_question,
    format_object_context,
)

from .dataset_prompts import (
    MULTICHOICE_SUFFIX,
    SHORT_ANSWER_SUFFIX,
    BEHAVIOR_FALLBACK_PROMPT,
)

__all__ = [
    # System prompts
    "DRIVING_SYSTEM_PROMPT",
    "GENERAL_DRIVING_SYSTEM_PROMPT",
    "BDDX_FEW_SHOT_PREFIX",
    # GPT eval prompts
    "PERCEPTION_MCQ_PROMPT",
    "PERCEPTION_VQA_PROMPT",
    "PREDICTION_VQA_PROMPT",
    "PLANNING_VQA_PROMPT",
    "BEHAVIOR_MCQ_PROMPT",
    "get_eval_prompt_for_task",
    # Visual description
    "extract_visual_description",
    "extract_camera_from_question",
    "format_object_context",
    # Dataset-specific prompt constants
    "MULTICHOICE_SUFFIX",
    "SHORT_ANSWER_SUFFIX",
    "BEHAVIOR_FALLBACK_PROMPT",
]
