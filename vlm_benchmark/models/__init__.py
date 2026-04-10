"""Model wrapper modules for VLM benchmark.

Unified backend architecture:
- TransformersWrapper: Handles all architectures (Qwen, LLaVA, InternVL, Dolphins) with transformers backend
- VLLMWrapper: Handles all architectures with vLLM backend
- OpenAIWrapper: GPT-4o via OpenAI API
- OpenRouterWrapper: Multiple models via OpenRouter API
"""

from .base_model import BaseVLMWrapper, ModelOutput, GenerationConfig
from .transformers_wrapper import TransformersWrapper
from .llm_wrapper import LLMWrapper, MockLLM
from .vllm_wrapper import VLLMWrapper
from .openai_wrapper import OpenAIWrapper
from .openrouter_wrapper import OpenRouterWrapper, OPENROUTER_MODELS, get_model_identifier
from .model_factory import (
    create_model,
    BACKEND_COMPARISON,
    print_backend_comparison,
    # Convenience functions
    create_qwen_transformers,
    create_qwen_vllm,
    create_llava_transformers,
    create_llava_vllm,
    create_openai_gpt4o,
    create_openrouter_claude,
)

__all__ = [
    # Base classes
    "BaseVLMWrapper",
    "ModelOutput",
    "GenerationConfig",
    # Unified backend wrappers
    "TransformersWrapper",
    "VLLMWrapper",
    "OpenAIWrapper",
    "OpenRouterWrapper",
    # LLM for attacks
    "LLMWrapper",
    "MockLLM",
    # OpenRouter utilities
    "OPENROUTER_MODELS",
    "get_model_identifier",
    # Factory (recommended)
    "create_model",
    "BACKEND_COMPARISON",
    "print_backend_comparison",
    # Convenience functions
    "create_qwen_transformers",
    "create_qwen_vllm",
    "create_llava_transformers",
    "create_llava_vllm",
    "create_openai_gpt4o",
    "create_openrouter_claude",
]
