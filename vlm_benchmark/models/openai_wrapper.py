"""
OpenAI API wrapper for VLM inference.

Supports GPT-4o, GPT-4o-mini, and other OpenAI vision models.
Does not support gradients (API-based inference only).
"""

from typing import List, Dict, Any, Optional, Union
from PIL import Image
import torch
import base64
import io
import os

from .base_model import ImageVLMWrapper, ModelOutput, GenerationConfig


class OpenAIWrapper(ImageVLMWrapper):
    """Wrapper for OpenAI vision models via API.

    Features:
    - GPT-4o, GPT-4o-mini, GPT-4-turbo with vision
    - No local GPU required
    - High quality responses
    - No gradient support

    Supported models:
    - gpt-4o
    - gpt-4o-mini
    - gpt-4-turbo
    - gpt-4-vision-preview (legacy)
    """

    DEFAULT_MODEL = "gpt-4o"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 60,
        **kwargs
    ):
        """Initialize OpenAI wrapper.

        Args:
            model_name: OpenAI model name
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            base_url: Optional custom base URL
            max_retries: Maximum number of retries
            timeout: Request timeout in seconds
            **kwargs: Additional arguments
        """
        super().__init__(model_name, device="api", dtype=torch.float32, backend="openai", **kwargs)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url
        self.max_retries = max_retries
        self.timeout = timeout
        self.supports_multi_image = True
        self.supports_gradients = False

        # Detect vLLM server mode (localhost URLs indicate local vLLM server)
        self.is_vllm_server = base_url and ("localhost" in base_url or "127.0.0.1" in base_url)
        self.supports_concurrent_batch = self.is_vllm_server  # vLLM servers support concurrent batching

        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

    def load_model(self) -> None:
        """Initialize OpenAI client."""
        try:
            import openai

            print(f"Initializing OpenAI client ({self.model_name})...")

            client_kwargs = {
                "api_key": self.api_key,
                "max_retries": self.max_retries,
                "timeout": self.timeout,
            }

            if self.base_url:
                client_kwargs["base_url"] = self.base_url

            self.client = openai.OpenAI(**client_kwargs)

            self._loaded = True
            print(f"OpenAI client initialized successfully")

        except ImportError:
            raise ImportError(
                "openai package not installed. Install with: pip install openai"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}")

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
        detail: str = "auto",
        use_manual_prompt: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Preprocess images and text for OpenAI API.

        Args:
            images: List of PIL Images
            prompt: Text prompt
            system_prompt: Optional system prompt
            detail: Image detail level ('auto', 'low', 'high')
            use_manual_prompt: If True, raises error (not supported via HTTP API)
            **kwargs: Additional arguments

        Returns:
            Dictionary with messages for OpenAI API

        Raises:
            ValueError: If use_manual_prompt=True (not supported via HTTP)
        """
        if not self._loaded:
            self.load_model()

        # Note: use_manual_prompt is ignored for OpenAI API (always uses chat templates)
        # The HTTP API doesn't support manual prompt formatting

        # Build message content
        content = []

        # Add images with enhanced camera labels (position context)
        camera_labels = kwargs.get("camera_labels")
        total_images = len(images)
        for i, img in enumerate(images):
            if camera_labels and i < len(camera_labels):
                content.append({
                    "type": "text",
                    "text": f"[Image {i+1}/{total_images} - {camera_labels[i]}]"
                })
            img_data = self._encode_image(img)
            # Log image size for debugging
            if i == 0:  # Only log first image to avoid spam
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(f"[OpenAI] Encoding image {i+1}: size={img.size}, mode={img.mode}, base64_len={len(img_data)}")
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": img_data,
                    "detail": detail,
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

    def inference(
        self,
        images,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> ModelOutput:
        """Run inference with logprob capture.

        Overrides ImageVLMWrapper.inference to thread a logprob container
        through generate() so first-token logprobs appear in metadata.
        """
        if not self._loaded:
            self.load_model()

        # Normalize to list (ImageVLMWrapper normally does this via super)
        if isinstance(images, Image.Image):
            images = [images]

        if config is None:
            config = GenerationConfig(max_new_tokens=256, do_sample=False)

        inputs = self.preprocess(images, prompt, **kwargs)
        logprob_container = {}
        text = self.generate(inputs, config, _logprob_out=logprob_container, **kwargs)

        metadata = {
            "model_name": self.model_name,
            "num_images": len(images) if isinstance(images, list) else 1,
            "prompt": prompt,
        }
        if "logprobs" in logprob_container:
            metadata["logprobs"] = logprob_container["logprobs"]

        return ModelOutput(text=text, metadata=metadata)

    def generate(
        self,
        inputs: Dict[str, Any],
        config: Optional[GenerationConfig] = None,
        _logprob_out: Optional[Dict] = None,
        **kwargs
    ) -> str:
        """Generate text using OpenAI API.

        Args:
            inputs: Dict with 'messages'
            config: Generation configuration
            _logprob_out: Optional dict to receive first-token logprobs
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
            # OpenAI API parameters
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

            # Pass frequency_penalty for vLLM servers (maps from repetition_penalty)
            # vLLM's OpenAI-compatible API uses frequency_penalty instead of repetition_penalty
            # repetition_penalty 1.3 maps to frequency_penalty 0.3
            if self.is_vllm_server and config.repetition_penalty != 1.0:
                api_kwargs["frequency_penalty"] = config.repetition_penalty - 1.0

            # Add logprobs to API request if config requests them
            if config and config.logprobs:
                api_kwargs["logprobs"] = True
                api_kwargs["top_logprobs"] = config.top_logprobs

            # Call API
            response = self.client.chat.completions.create(**api_kwargs)

            # Extract generated text
            generated = response.choices[0].message.content.strip()

            # Extract first-token logprobs into container if provided
            if _logprob_out is not None and config and config.logprobs:
                try:
                    lp_content = response.choices[0].logprobs
                    if lp_content and lp_content.content and len(lp_content.content) > 0:
                        first_token = lp_content.content[0]
                        _logprob_out["logprobs"] = {
                            tlp.token.strip().upper(): tlp.logprob
                            for tlp in first_token.top_logprobs
                            if tlp.token.strip().upper() in {"A", "B", "C", "D"}
                        }
                except Exception:
                    pass  # Logprobs unavailable, text fallback handles it

            return generated

        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {e}")

    def batch_inference(
        self,
        batch_images: List[List[Image.Image]],
        batch_prompts: List[str],
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> List[ModelOutput]:
        """Run batch inference with OpenAI API.

        For vLLM servers (localhost), uses concurrent requests for better throughput.
        For OpenAI API, falls back to sequential processing.

        Args:
            batch_images: List of image lists
            batch_prompts: List of prompts
            config: Generation configuration
            **kwargs: Additional arguments passed through.

        Returns:
            List of ModelOutputs
        """
        if self.is_vllm_server:
            # vLLM server: use concurrent requests for better throughput
            from concurrent.futures import ThreadPoolExecutor, as_completed

            # Limit concurrent requests to avoid overwhelming the server
            max_workers = min(32, len(batch_images))

            def process_sample(idx):
                """Process a single sample and return (index, result)."""
                try:
                    result = self.inference(
                        batch_images[idx],
                        batch_prompts[idx],
                        config=config,
                        **kwargs
                    )
                    return (idx, result)
                except Exception as e:
                    print(f"Error processing sample {idx}: {e}")
                    return (idx, None)

            # Submit all tasks concurrently
            results = [None] * len(batch_images)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(process_sample, i): i for i in range(len(batch_images))}
                for future in as_completed(futures):
                    idx, result = future.result()
                    results[idx] = result

            return results
        else:
            # OpenAI API: sequential inference
            return super().batch_inference(
                batch_images,
                batch_prompts,
                config,
                **kwargs
            )
