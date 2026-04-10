"""
Base model wrapper for VLM benchmark.

Defines the abstract interface for all VLM model wrappers with
standard methods for loading, preprocessing, and inference.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from PIL import Image
import torch


@dataclass
class ModelOutput:
    """Standard output format for model inference.

    Attributes:
        text: Generated text response
        logits: Optional raw logits
        hidden_states: Optional hidden states
        attention: Optional attention weights
        metadata: Additional output metadata
    """
    text: str
    logits: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None
    attention: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding tensors)."""
        return {
            "text": self.text,
            "metadata": self.metadata,
        }


@dataclass
class GenerationConfig:
    """Configuration for text generation.

    Attributes:
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        do_sample: Whether to use sampling
        num_beams: Number of beams for beam search
        no_repeat_ngram_size: Prevent repeating n-grams
        length_penalty: Length penalty for beam search
    """
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    do_sample: bool = False
    num_beams: int = 1
    no_repeat_ngram_size: int = 0
    length_penalty: float = 1.0
    repetition_penalty: float = 1.0
    logprobs: bool = False        # Request logprobs from model
    top_logprobs: int = 5         # Number of top logprob alternatives per token

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for model.generate()."""
        d = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "do_sample": self.do_sample,
            "num_beams": self.num_beams,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
            "length_penalty": self.length_penalty,
        }
        if self.repetition_penalty != 1.0:
            d["repetition_penalty"] = self.repetition_penalty
        return d


class BaseVLMWrapper(ABC):
    """Abstract base class for Vision-Language Model wrappers.

    All model implementations should inherit from this class and implement
    the required abstract methods for loading, preprocessing, and inference.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        dtype: torch.dtype = torch.bfloat16,
        backend: str = "transformers",
        **kwargs
    ):
        """Initialize base wrapper.

        Args:
            model_name: Model name or path
            device: Device to load model on ('auto', 'cuda', 'cpu')
            dtype: Model data type
            backend: Inference backend ('transformers', 'vllm', 'openai', 'openrouter')
            **kwargs: Additional model-specific arguments
        """
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.backend = backend
        self.model = None
        self.processor = None
        self.tokenizer = None
        self._loaded = False
        self.supports_gradients = backend == "transformers"

    @abstractmethod
    def load_model(self) -> None:
        """Load model and processor.

        Must be implemented by subclasses. Should set:
        - self.model
        - self.processor (if applicable)
        - self.tokenizer (if applicable)
        - self._loaded = True
        """
        pass

    @abstractmethod
    def preprocess(
        self,
        images: List[Image.Image],
        prompt: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Preprocess images and text for model input.

        Args:
            images: List of PIL Images
            prompt: Text prompt
            **kwargs: Additional preprocessing arguments

        Returns:
            Dictionary of model inputs
        """
        pass

    @abstractmethod
    def generate(
        self,
        inputs: Dict[str, Any],
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> str:
        """Generate text from preprocessed inputs.

        Args:
            inputs: Preprocessed model inputs
            config: Generation configuration
            **kwargs: Additional generation arguments

        Returns:
            Generated text string
        """
        pass

    def inference(
        self,
        images: List[Image.Image],
        prompt: str,
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> ModelOutput:
        """Run full inference pipeline.

        Convenience method that combines preprocess + generate.

        Args:
            images: List of PIL Images
            prompt: Text prompt
            config: Generation configuration
            **kwargs: Additional arguments

        Returns:
            ModelOutput with generated text
        """
        if not self._loaded:
            self.load_model()

        # Preprocess
        inputs = self.preprocess(images, prompt, **kwargs)

        # Generate
        text = self.generate(inputs, config, **kwargs)

        return ModelOutput(
            text=text,
            metadata={
                "model_name": self.model_name,
                "num_images": len(images),
                "prompt": prompt,
            }
        )

    def batch_inference(
        self,
        batch_images: List[List[Image.Image]],
        batch_prompts: List[str],
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> List[ModelOutput]:
        """Run inference on a batch of samples.

        Default implementation runs sequentially. Subclasses can override
        for true batch inference if supported.

        Args:
            batch_images: List of image lists
            batch_prompts: List of prompts
            config: Generation configuration
            **kwargs: Additional arguments

        Returns:
            List of ModelOutputs
        """
        outputs = []
        for images, prompt in zip(batch_images, batch_prompts):
            output = self.inference(images, prompt, config, **kwargs)
            outputs.append(output)
        return outputs

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "loaded": self._loaded,
        }

    def unload(self) -> None:
        """Unload model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        self._loaded = False

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class ImageVLMWrapper(BaseVLMWrapper):
    """Base class for image-based VLMs (single image input).

    Extends BaseVLMWrapper with image-specific utilities.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.supports_multi_image = False

    def inference(
        self,
        images: Union[Image.Image, List[Image.Image]],
        prompt: str,
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> ModelOutput:
        """Run inference with single or multiple images.

        Args:
            images: Single image or list of images
            prompt: Text prompt
            config: Generation configuration

        Returns:
            ModelOutput with generated text
        """
        # Normalize to list
        if isinstance(images, Image.Image):
            images = [images]

        return super().inference(images, prompt, config, **kwargs)


class VideoVLMWrapper(BaseVLMWrapper):
    """Base class for video-based VLMs (multiple frame input).

    Extends BaseVLMWrapper with video-specific utilities.
    """

    def __init__(self, *args, num_frames: int = 16, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_frames = num_frames

    def process_video(
        self,
        frames: List[Image.Image],
        target_frames: Optional[int] = None
    ) -> List[Image.Image]:
        """Process video frames to target count.

        Args:
            frames: List of video frames
            target_frames: Target number of frames (default: self.num_frames)

        Returns:
            Processed list of frames
        """
        target = target_frames or self.num_frames

        if len(frames) == target:
            return frames

        if len(frames) > target:
            # Uniform sample
            step = len(frames) / target
            indices = [int(i * step) for i in range(target)]
            return [frames[i] for i in indices]

        # Duplicate frames to reach target
        while len(frames) < target:
            frames.append(frames[-1].copy())
        return frames[:target]
