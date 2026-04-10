"""
OpenRouter API wrapper for VLM inference.

Supports multiple vision models via OpenRouter unified API including:
- Anthropic Claude 3.5/4 Sonnet with vision
- Google Gemini models
- Meta Llama 3.2 Vision
- Qwen2-VL
- And many more

Does not support gradients (API-based inference only).
"""

from typing import List, Dict, Any, Optional
from PIL import Image
import torch
import base64
import io
import os

from .base_model import ImageVLMWrapper, ModelOutput, GenerationConfig


class OpenRouterWrapper(ImageVLMWrapper):
    """Wrapper for multiple VLM models via OpenRouter API.

    Features:
    - Access to 100+ models through single API
    - No local GPU required
    - Automatic fallback and routing
    - Cost optimization
    - No gradient support

    Popular models:
    - anthropic/claude-3.5-sonnet (recommended)
    - anthropic/claude-4-sonnet
    - google/gemini-pro-vision
    - meta-llama/llama-3.2-90b-vision
    - qwen/qwen2-vl-72b
    - openai/gpt-4o (via OpenRouter)
    """

    DEFAULT_MODEL = "anthropic/claude-3.5-sonnet"
    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        site_url: Optional[str] = None,
        app_name: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 60,
        **kwargs
    ):
        """Initialize OpenRouter wrapper.

        Args:
            model_name: OpenRouter model identifier (e.g., "anthropic/claude-3.5-sonnet")
            api_key: OpenRouter API key (or set OPENROUTER_API_KEY env var)
            site_url: Optional site URL for rankings
            app_name: Optional app name for rankings
            max_retries: Maximum number of retries
            timeout: Request timeout in seconds
            **kwargs: Additional arguments
        """
        super().__init__(model_name, device="api", dtype=torch.float32, backend="openrouter", **kwargs)
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.site_url = site_url
        self.app_name = app_name or "VLM-Adversarial-Benchmark"
        self.max_retries = max_retries
        self.timeout = timeout
        self.supports_multi_image = True
        self.supports_gradients = False

        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not provided. Set OPENROUTER_API_KEY environment variable "
                "or pass api_key parameter."
            )

    def load_model(self) -> None:
        """Initialize OpenRouter client (OpenAI-compatible)."""
        try:
            import openai

            print(f"Initializing OpenRouter client ({self.model_name})...")

            # OpenRouter uses OpenAI-compatible API
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.BASE_URL,
                max_retries=self.max_retries,
                timeout=self.timeout,
            )

            self._loaded = True
            print(f"OpenRouter client initialized successfully")

        except ImportError:
            raise ImportError(
                "openai package not installed. Install with: pip install openai"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenRouter client: {e}")

    def _encode_image(self, image: Image.Image) -> str:
        """Encode PIL Image to base64 string.

        Args:
            image: PIL Image

        Returns:
            Base64 encoded image string
        """
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"

    def preprocess(
        self,
        images: List[Image.Image],
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Preprocess images and text for OpenRouter API.

        Args:
            images: List of PIL Images
            prompt: Text prompt
            system_prompt: Optional system prompt
            **kwargs: Additional arguments

        Returns:
            Dictionary with messages for OpenRouter API
        """
        if not self._loaded:
            self.load_model()

        # Build message content
        content = []

        # Add images (OpenRouter uses OpenAI format)
        for img in images:
            img_data = self._encode_image(img)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": img_data,
                }
            })

        # Add text
        content.append({
            "type": "text",
            "text": prompt
        })

        # Build messages
        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })

        messages.append({
            "role": "user",
            "content": content
        })

        return {"messages": messages}

    def generate(
        self,
        inputs: Dict[str, Any],
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> str:
        """Generate text using OpenRouter API.

        Args:
            inputs: Dict with 'messages'
            config: Generation configuration
            **kwargs: Additional generation arguments

        Returns:
            Generated text string
        """
        if config is None:
            config = GenerationConfig(
                max_new_tokens=256,
                do_sample=False,
            )

        try:
            # OpenRouter API parameters
            api_kwargs = {
                "model": self.model_name,
                "messages": inputs["messages"],
                "max_tokens": config.max_new_tokens,
            }

            # Add sampling parameters if do_sample=True
            if config.do_sample:
                api_kwargs["temperature"] = config.temperature
                api_kwargs["top_p"] = config.top_p
            else:
                api_kwargs["temperature"] = 0.0

            # OpenRouter-specific headers
            extra_headers = {}
            if self.site_url:
                extra_headers["HTTP-Referer"] = self.site_url
            if self.app_name:
                extra_headers["X-Title"] = self.app_name

            # Call API
            response = self.client.chat.completions.create(
                **api_kwargs,
                extra_headers=extra_headers if extra_headers else None
            )

            # Extract generated text
            generated = response.choices[0].message.content.strip()

            return generated

        except Exception as e:
            raise RuntimeError(f"OpenRouter API call failed: {e}")

    def batch_inference(
        self,
        batch_images: List[List[Image.Image]],
        batch_prompts: List[str],
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> List[ModelOutput]:
        """Run batch inference with OpenRouter API.

        Note: Sequential processing (no native batch API).

        Args:
            batch_images: List of image lists
            batch_prompts: List of prompts
            config: Generation configuration
            **kwargs: Additional arguments

        Returns:
            List of ModelOutputs
        """
        # Sequential inference
        return super().batch_inference(
            batch_images,
            batch_prompts,
            config,
            **kwargs
        )

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information including pricing."""
        base_info = super().get_model_info()
        base_info.update({
            "backend": "openrouter",
            "base_url": self.BASE_URL,
            "app_name": self.app_name,
        })
        return base_info


# Popular model presets
OPENROUTER_MODELS = {
    # Anthropic models
    "claude-3.5-sonnet": "anthropic/claude-3.5-sonnet",
    "claude-4-sonnet": "anthropic/claude-4-sonnet",
    "claude-3.5-opus": "anthropic/claude-3.5-opus",

    # Google models
    "gemini-pro-vision": "google/gemini-pro-vision",
    "gemini-1.5-pro": "google/gemini-1.5-pro",

    # Meta models
    "llama-3.2-90b-vision": "meta-llama/llama-3.2-90b-vision",
    "llama-3.2-11b-vision": "meta-llama/llama-3.2-11b-vision",

    # Qwen models
    "qwen2-vl-72b": "qwen/qwen2-vl-72b",
    "qwen2-vl-7b": "qwen/qwen2-vl-7b",

    # OpenAI via OpenRouter
    "gpt-4o": "openai/gpt-4o",
    "gpt-4o-mini": "openai/gpt-4o-mini",
}


def get_model_identifier(shortname: str) -> str:
    """Get full OpenRouter model identifier from shortname.

    Args:
        shortname: Short model name

    Returns:
        Full OpenRouter identifier

    Example:
        >>> get_model_identifier("claude-3.5-sonnet")
        'anthropic/claude-3.5-sonnet'
    """
    return OPENROUTER_MODELS.get(shortname, shortname)
