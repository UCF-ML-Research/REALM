"""
Model factory for creating VLM wrappers with different backends.

Provides unified interface for initializing models across:
- Transformers (local, gradient support) - unified for all architectures
- vLLM (local, high-throughput) - unified for all architectures
- OpenAI (API, GPT-4o)
- OpenRouter (API, multiple models)
"""

from typing import Optional, Dict, Any
import torch

from .transformers_wrapper import TransformersWrapper
from .vllm_wrapper import VLLMWrapper
from .openai_wrapper import OpenAIWrapper
from .openrouter_wrapper import OpenRouterWrapper, get_model_identifier


# Default model names for common architectures
DEFAULT_MODELS = {
    "qwen": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen2.5-vl": "Qwen/Qwen2.5-VL-7B-Instruct",
    "llava": "liuhaotian/llava-v1.5-7b",
    "internvl": "OpenGVLab/InternVL2-8B",
    "dolphins": "gray311/Dolphins",
    "openai": "gpt-4o",
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
    "openrouter": "anthropic/claude-3.5-sonnet",
}


def create_model(
    model_type: str,
    backend: str = "auto",
    model_name: Optional[str] = None,
    device: str = "auto",
    dtype: torch.dtype = torch.bfloat16,
    **kwargs
) -> Any:
    """Create a VLM wrapper with specified backend.

    Unified factory that supports any model architecture with any backend.

    Args:
        model_type: Model architecture ('qwen', 'llava', 'internvl', 'dolphins') or API ('openai', 'openrouter')
        backend: Inference backend ('auto', 'transformers', 'vllm', 'openai', 'openrouter')
        model_name: Optional model name (defaults to model_type default)
        device: Device for local models ('auto', 'cuda', 'cpu')
        dtype: Data type for local models
        **kwargs: Additional model-specific arguments

    Returns:
        VLM wrapper instance

    Examples:
        >>> # Qwen with transformers (gradient support)
        >>> model = create_model("qwen", backend="transformers")

        >>> # Qwen with vLLM (high throughput)
        >>> model = create_model("qwen", backend="vllm", tensor_parallel_size=2)

        >>> # LLaVA with transformers
        >>> model = create_model("llava", backend="transformers")

        >>> # LLaVA with vLLM
        >>> model = create_model("llava", backend="vllm")

        >>> # OpenAI API
        >>> model = create_model("openai", model_name="gpt-4o", api_key="...")

        >>> # OpenRouter API
        >>> model = create_model("openrouter", model_name="anthropic/claude-3.5-sonnet", api_key="...")
    """
    # Resolve model name
    if model_name is None:
        model_name = DEFAULT_MODELS.get(model_type, model_type)

    # Auto-detect backend
    if backend == "auto":
        backend = _detect_backend(model_type, model_name, kwargs)

    # Create model based on backend
    if backend == "transformers":
        return _create_transformers_model(model_name, device, dtype, **kwargs)

    elif backend == "vllm":
        return _create_vllm_model(model_name, device, dtype, **kwargs)

    elif backend == "openai":
        return _create_openai_model(model_name, **kwargs)

    elif backend == "openrouter":
        return _create_openrouter_model(model_name, **kwargs)

    else:
        raise ValueError(f"Unknown backend: {backend}")


def _detect_backend(model_type: str, model_name: str, kwargs: Dict[str, Any]) -> str:
    """Auto-detect appropriate backend.

    Args:
        model_type: Model type
        model_name: Model name
        kwargs: Additional arguments

    Returns:
        Backend name
    """
    # Check for explicit API keys
    if "api_key" in kwargs or model_type in ["openai", "gpt-4o", "gpt-4o-mini"]:
        return "openai"

    if model_type == "openrouter" or "openrouter" in model_name.lower():
        return "openrouter"

    # Check for vLLM-specific arguments
    if any(k in kwargs for k in ["tensor_parallel_size", "gpu_memory_utilization"]):
        return "vllm"

    # Default to transformers (gradient support)
    return "transformers"


def _create_transformers_model(
    model_name: str,
    device: str,
    dtype: torch.dtype,
    **kwargs
) -> TransformersWrapper:
    """Create model with transformers backend.

    Unified wrapper that auto-detects architecture from model name.

    Args:
        model_name: Model name (architecture auto-detected)
        device: Device
        dtype: Data type
        **kwargs: Additional arguments

    Returns:
        TransformersWrapper instance
    """
    return TransformersWrapper(
        model_name=model_name,
        device=device,
        dtype=dtype,
        **kwargs
    )


def _create_vllm_model(
    model_name: str,
    device: str,
    dtype: torch.dtype,
    **kwargs
) -> VLLMWrapper:
    """Create model with vLLM backend.

    Works with any architecture - vLLM auto-detects from model name.

    Args:
        model_name: Model name (architecture auto-detected)
        device: Device (vLLM auto-manages)
        dtype: Data type
        **kwargs: Additional arguments

    Returns:
        VLLMWrapper instance
    """
    return VLLMWrapper(
        model_name=model_name,
        device=device,
        dtype=dtype,
        **kwargs
    )


def _create_openai_model(model_name: str, **kwargs) -> OpenAIWrapper:
    """Create model with OpenAI API backend.

    Args:
        model_name: OpenAI model name
        **kwargs: Additional arguments (api_key, etc.)

    Returns:
        OpenAIWrapper instance
    """
    return OpenAIWrapper(
        model_name=model_name,
        **kwargs
    )


def _create_openrouter_model(model_name: str, **kwargs) -> OpenRouterWrapper:
    """Create model with OpenRouter API backend.

    Args:
        model_name: OpenRouter model identifier
        **kwargs: Additional arguments (api_key, etc.)

    Returns:
        OpenRouterWrapper instance
    """
    # Resolve shortname if needed
    full_model_name = get_model_identifier(model_name)

    return OpenRouterWrapper(
        model_name=full_model_name,
        **kwargs
    )


# Convenience functions for common model+backend combinations
def create_qwen_transformers(**kwargs):
    """Create Qwen model with transformers backend (gradient support).

    Args:
        **kwargs: Additional arguments passed to create_model()

    Returns:
        TransformersWrapper for Qwen
    """
    return create_model("qwen", backend="transformers", **kwargs)


def create_qwen_vllm(**kwargs):
    """Create Qwen model with vLLM backend (high throughput).

    Args:
        **kwargs: Additional arguments passed to create_model()

    Returns:
        VLLMWrapper for Qwen
    """
    return create_model("qwen", backend="vllm", **kwargs)


def create_llava_transformers(**kwargs):
    """Create LLaVA model with transformers backend (gradient support).

    Args:
        **kwargs: Additional arguments passed to create_model()

    Returns:
        TransformersWrapper for LLaVA
    """
    return create_model("llava", backend="transformers", **kwargs)


def create_llava_vllm(**kwargs):
    """Create LLaVA model with vLLM backend (high throughput).

    Args:
        **kwargs: Additional arguments passed to create_model()

    Returns:
        VLLMWrapper for LLaVA
    """
    return create_model("llava", backend="vllm", **kwargs)


def create_openai_gpt4o(api_key: Optional[str] = None, **kwargs):
    """Create GPT-4o model via OpenAI API.

    Args:
        api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        **kwargs: Additional arguments passed to create_model()

    Returns:
        OpenAIWrapper for GPT-4o
    """
    return create_model("openai", model_name="gpt-4o", api_key=api_key, **kwargs)


def create_openrouter_claude(api_key: Optional[str] = None, **kwargs):
    """Create Claude 3.5 Sonnet model via OpenRouter API.

    Args:
        api_key: OpenRouter API key (or set OPENROUTER_API_KEY env var)
        **kwargs: Additional arguments passed to create_model()

    Returns:
        OpenRouterWrapper for Claude 3.5 Sonnet
    """
    return create_model("openrouter", model_name="anthropic/claude-3.5-sonnet", api_key=api_key, **kwargs)


# Backend comparison matrix
BACKEND_COMPARISON = {
    "transformers": {
        "supports_gradients": True,
        "location": "local",
        "speed": "medium",
        "memory": "high",
        "cost": "hardware",
        "use_case": "Visual attacks (ADvLM, Adversarial Patch), gradient utilities",
    },
    "vllm": {
        "supports_gradients": False,
        "location": "local",
        "speed": "very high",
        "memory": "medium",
        "cost": "hardware",
        "use_case": "High-throughput inference, large-scale evaluation",
    },
    "openai": {
        "supports_gradients": False,
        "location": "api",
        "speed": "medium",
        "memory": "none",
        "cost": "per-token",
        "use_case": "GPT-4o evaluation, no GPU available",
    },
    "openrouter": {
        "supports_gradients": False,
        "location": "api",
        "speed": "medium",
        "memory": "none",
        "cost": "per-token",
        "use_case": "Multiple model comparison, Claude/Gemini/etc",
    },
}


def print_backend_comparison():
    """Print comparison table of backends."""
    print("\n=== Backend Comparison ===\n")
    print(f"{'Backend':<15} {'Gradients':<12} {'Location':<10} {'Speed':<12} {'Memory':<10} {'Cost':<12}")
    print("-" * 80)

    for backend, info in BACKEND_COMPARISON.items():
        print(f"{backend:<15} {str(info['supports_gradients']):<12} {info['location']:<10} "
              f"{info['speed']:<12} {info['memory']:<10} {info['cost']:<12}")

    print("\n=== Use Cases ===\n")
    for backend, info in BACKEND_COMPARISON.items():
        print(f"{backend:>15}: {info['use_case']}")
    print()


if __name__ == "__main__":
    # Print comparison when run directly
    print_backend_comparison()

    print("\n=== Example Usage ===\n")
    print("# Transformers (gradient support for visual attacks)")
    print('model = create_model("qwen", backend="transformers")')
    print()
    print("# vLLM (high throughput)")
    print('model = create_model("qwen", backend="vllm", tensor_parallel_size=2)')
    print()
    print("# OpenAI API")
    print('model = create_model("openai", model_name="gpt-4o", api_key="...")')
    print()
    print("# OpenRouter API")
    print('model = create_model("openrouter", model_name="anthropic/claude-3.5-sonnet", api_key="...")')
