"""
Prompt library generation for LSE (Low Semantic Entropy) attacks.

Provides functions to generate semantically equivalent prompts with
low entropy to improve attack transferability.
"""

from typing import List, Optional
import random


def generate_lse_prompts(
    original_prompt: str,
    num_prompts: int = 3,
    task_type: str = "behavior",
    method: str = "template",
) -> List[str]:
    """Generate low-semantic-entropy paraphrases of prompt.

    These prompts should have the same semantic meaning but different
    surface forms to improve attack robustness.

    Args:
        original_prompt: Original question/prompt
        num_prompts: Number of paraphrases to generate
        task_type: Type of task (affects paraphrase style)
        method: Generation method ("template", "llm", "synonym")

    Returns:
        List of paraphrased prompts (including original)
    """
    if method == "template":
        return _template_based_paraphrases(original_prompt, num_prompts, task_type)

    elif method == "synonym":
        return _synonym_based_paraphrases(original_prompt, num_prompts)

    elif method == "llm":
        raise NotImplementedError("LLM-based paraphrasing not yet implemented")

    else:
        raise ValueError(f"Unknown method: {method}")


def _template_based_paraphrases(
    original: str,
    num_prompts: int,
    task_type: str,
) -> List[str]:
    """Generate paraphrases using templates.

    Args:
        original: Original prompt
        num_prompts: Number of paraphrases
        task_type: Task type

    Returns:
        List of paraphrases
    """
    paraphrases = [original]  # Include original

    if task_type == "behavior":
        # Behavior prediction paraphrases
        templates = [
            "What action is the vehicle taking?",
            "Describe the vehicle's current behavior.",
            "What is the ego vehicle doing right now?",
            "Please describe the driving action being performed.",
            "What maneuver is the autonomous vehicle executing?",
        ]

    elif task_type == "action":
        templates = [
            "What should the vehicle do?",
            "What is the recommended action?",
            "Describe the appropriate driving decision.",
            "What action should be taken in this scenario?",
            "Please suggest the next driving action.",
        ]

    elif task_type == "perception":
        templates = [
            "What do you see in the image?",
            "Describe the scene.",
            "What objects are present?",
            "Please describe the visual content.",
            "What is visible in this driving scene?",
        ]

    elif task_type == "planning":
        templates = [
            "What is the driving plan?",
            "Describe the planned trajectory.",
            "What should the vehicle plan to do?",
            "Please outline the driving strategy.",
            "What is the appropriate course of action?",
        ]

    else:
        # Generic paraphrases
        templates = [
            original,
            f"Please answer: {original}",
            f"Question: {original}",
            original.replace("?", " please?") if "?" in original else f"{original}?",
        ]

    # Shuffle and select
    random.shuffle(templates)
    paraphrases.extend(templates[:num_prompts - 1])

    return paraphrases[:num_prompts]


def _synonym_based_paraphrases(
    original: str,
    num_prompts: int,
) -> List[str]:
    """Generate paraphrases by substituting synonyms.

    Args:
        original: Original prompt
        num_prompts: Number of paraphrases

    Returns:
        List of paraphrases
    """
    # Simple synonym dictionary for driving domain
    synonyms = {
        "vehicle": ["car", "automobile", "ego vehicle"],
        "driving": ["moving", "traveling", "operating"],
        "action": ["maneuver", "behavior", "decision"],
        "scene": ["scenario", "situation", "environment"],
        "describe": ["explain", "detail", "characterize"],
        "what": ["which", "what kind of"],
    }

    paraphrases = [original]
    original_lower = original.lower()

    # Generate variations by substituting synonyms
    for _ in range(num_prompts - 1):
        paraphrase = original

        # Try to substitute one or two words
        for word, syns in synonyms.items():
            if word in original_lower:
                syn = random.choice(syns)
                # Case-preserving replacement
                if word.capitalize() in original:
                    paraphrase = paraphrase.replace(word.capitalize(), syn.capitalize())
                else:
                    paraphrase = paraphrase.replace(word, syn)

                # Only substitute one word per paraphrase for diversity
                if paraphrase != original:
                    break

        # If no substitution made, add prefix/suffix
        if paraphrase == original:
            prefixes = ["Please answer: ", "Question: ", ""]
            suffixes = [" Please be specific.", " Thanks.", ""]
            paraphrase = random.choice(prefixes) + original + random.choice(suffixes)

        paraphrases.append(paraphrase)

    return paraphrases


def create_viewpoint_robust_prompts(
    original_prompt: str,
    num_prompts: int = 5,
) -> List[str]:
    """Create prompts robust to different viewpoints (for ADvLM).

    Args:
        original_prompt: Original prompt
        num_prompts: Number of prompts to generate

    Returns:
        List of viewpoint-robust prompts
    """
    # Combine multiple paraphrasing strategies
    prompts = set([original_prompt])

    # Template-based
    prompts.update(_template_based_paraphrases(original_prompt, num_prompts // 2, "behavior"))

    # Synonym-based
    prompts.update(_synonym_based_paraphrases(original_prompt, num_prompts // 2))

    # Convert back to list
    prompts_list = list(prompts)

    # Ensure we have enough
    while len(prompts_list) < num_prompts:
        prompts_list.append(original_prompt)

    return prompts_list[:num_prompts]
