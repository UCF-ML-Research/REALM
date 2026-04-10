"""
VLM Inference Module

Provides a unified interface for querying Vision-Language Models
with clean and adversarial images. Supports multiple VLM providers.
"""

import base64
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image


@dataclass
class VLMResponse:
    """VLM response container."""
    image_path: str
    model_name: str
    query: str
    response: str
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class VLMInference:
    """
    Unified VLM inference interface.

    Supports multiple VLM providers:
    - OpenAI (GPT-4V, GPT-4o, GPT-4o-mini)
    - Anthropic (Claude 3.5 Sonnet, etc.)
    - Google (Gemini)

    Example:
        vlm = VLMInference(model="gpt-4o-mini-2024-07-18", api_key="...")
        responses = vlm.query_images(
            image_paths=["img1.png", "img2.png"],
            query="Describe the main object in the scene."
        )
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini-2024-07-18",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        max_workers: int = 10
    ):
        """
        Initialize VLM inference.

        Args:
            model: Model name (e.g., "gpt-4o-mini-2024-07-18", "claude-3-5-sonnet-20241022")
            api_key: API key (or set via environment variable)
            api_base: Optional custom API base URL
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
        """
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_workers = max_workers

        # Detect provider from model name
        self.provider = self._detect_provider(model)

        # Initialize client
        self._init_client()

    def _detect_provider(self, model: str) -> str:
        """Detect VLM provider from model name."""
        model_lower = model.lower()

        if "gpt" in model_lower or "o1" in model_lower:
            return "openai"
        elif "claude" in model_lower:
            return "anthropic"
        elif "gemini" in model_lower:
            return "google"
        elif "qwen" in model_lower or "llava" in model_lower or "llama" in model_lower:
            return "local"
        else:
            raise ValueError(f"Unknown VLM provider for model: {model}")

    def _init_client(self):
        """Initialize API client based on provider."""
        if self.provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base
            ) if self.api_base else OpenAI(api_key=self.api_key)

        elif self.provider == "anthropic":
            from anthropic import Anthropic
            self.client = Anthropic(api_key=self.api_key)

        elif self.provider == "google":
            # TODO: Add Gemini support
            raise NotImplementedError("Gemini support coming soon")

        elif self.provider == "local":
            # Local model via vllm OpenAI-compatible server
            from openai import OpenAI
            api_base = self.api_base or "http://localhost:8001/v1"
            self.client = OpenAI(
                api_key=self.api_key or "EMPTY",
                base_url=api_base
            )

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def _query_openai(self, image_path: str, query: str) -> str:
        """Query OpenAI VLM (GPT-4V, GPT-4o, etc.)."""
        base64_image = self._encode_image(image_path)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }}
                ]
            }],
            max_completion_tokens=self.max_tokens
        )

        return response.choices[0].message.content

    def _query_anthropic(self, image_path: str, query: str) -> str:
        """Query Anthropic Claude."""
        # Read image
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')

        # Detect image type
        image_path_lower = image_path.lower()
        if image_path_lower.endswith('.png'):
            media_type = "image/png"
        elif image_path_lower.endswith(('.jpg', '.jpeg')):
            media_type = "image/jpeg"
        else:
            media_type = "image/jpeg"  # Default

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data
                        }
                    },
                    {"type": "text", "text": query}
                ]
            }]
        )

        return response.content[0].text

    def query_image(
        self,
        image_path: str,
        query: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> VLMResponse:
        """
        Query VLM with a single image.

        Args:
            image_path: Path to image file
            query: Text query/prompt
            metadata: Optional metadata to attach to response

        Returns:
            VLMResponse with model output
        """
        try:
            if self.provider in ("openai", "local"):
                response_text = self._query_openai(image_path, query)
            elif self.provider == "anthropic":
                response_text = self._query_anthropic(image_path, query)
            else:
                raise NotImplementedError(f"Provider {self.provider} not supported")

            return VLMResponse(
                image_path=image_path,
                model_name=self.model,
                query=query,
                response=response_text,
                metadata=metadata
            )

        except Exception as e:
            return VLMResponse(
                image_path=image_path,
                model_name=self.model,
                query=query,
                response="",
                metadata=metadata,
                error=str(e)
            )

    def query_images(
        self,
        image_paths: List[str],
        query: str,
        metadata_list: Optional[List[Dict[str, Any]]] = None,
        verbose: bool = True
    ) -> List[VLMResponse]:
        """
        Query VLM with multiple images.

        Args:
            image_paths: List of image file paths
            query: Text query/prompt (same for all images)
            metadata_list: Optional list of metadata dicts (one per image)
            verbose: Print progress

        Returns:
            List of VLMResponse objects
        """
        if metadata_list is None:
            metadata_list = [None] * len(image_paths)

        responses = [None] * len(image_paths)
        completed = [0]

        def _query(args):
            idx, img_path, metadata = args
            return idx, self.query_image(img_path, query, metadata)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(_query, (i, img_path, meta)): i
                for i, (img_path, meta) in enumerate(zip(image_paths, metadata_list))
            }
            for future in as_completed(futures):
                idx, response = future.result()
                responses[idx] = response
                completed[0] += 1
                if verbose:
                    status = f"  → {response.response[:80]}..." if not response.error else f"  ✗ {response.error[:60]}"
                    print(f"[{completed[0]}/{len(image_paths)}] {Path(response.image_path).name}\n{status}")

        return responses
