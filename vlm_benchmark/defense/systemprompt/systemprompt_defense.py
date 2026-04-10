"""SystemPrompt defense: hardens the VLM system prompt, image unchanged."""

from dataclasses import dataclass
from PIL import Image

from ..base_defense import BaseDefense, DefenseConfig, DefenseResult
from .config import SYSTEM_PROMPT


@dataclass
class SystemPromptDefenseConfig(DefenseConfig):
    pass


class SystemPromptDefense(BaseDefense):
    """Prompt-level defense: returns the image unchanged and provides a
    hardened system prompt that instructs the VLM to ignore embedded text,
    adversarial overlays, and visual perturbations."""

    def get_system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def clean(self, image_path: str, **kwargs) -> DefenseResult:
        """Image is returned unchanged — defense operates at the prompt level."""
        img = Image.open(image_path).convert("RGB")
        return DefenseResult(
            cleaned_sample=img,
            original_image_path=image_path,
            detection_confidence=0.0,
            regions_removed=0,
            metadata={"system_prompt": SYSTEM_PROMPT},
        )

    def requires_model(self) -> bool:
        return False
