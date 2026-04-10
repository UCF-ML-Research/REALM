"""
vLLM backend wrapper for fast VLM inference.

Provides high-throughput, low-latency inference using vLLM's optimized engine.
Does not support gradients (use transformers backend for gradient-based attacks).
"""

from typing import List, Dict, Any, Optional
from PIL import Image
import torch
import os
import logging

from .base_model import ImageVLMWrapper, ModelOutput, GenerationConfig

logger = logging.getLogger(__name__)


class VLLMWrapper(ImageVLMWrapper):
    """Wrapper for VLM models using vLLM backend.

    Features:
    - High-throughput inference (2-5x faster than transformers)
    - PagedAttention for efficient memory management
    - Continuous batching
    - No gradient support (inference only)

    Supported models:
    - Qwen2.5-VL series
    - LLaVA-NeXT series
    - InternVL2 series
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        dtype: torch.dtype = torch.bfloat16,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        trust_remote_code: bool = True,
        **kwargs
    ):
        """Initialize vLLM wrapper.

        Args:
            model_name: HuggingFace model name or local path
            device: Device (vLLM auto-manages GPUs)
            dtype: Model data type
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: GPU memory fraction to use (0.0-1.0)
            max_model_len: Maximum sequence length
            trust_remote_code: Trust remote code from HuggingFace
            **kwargs: Additional vLLM arguments
        """
        super().__init__(model_name, device, dtype, backend="vllm", **kwargs)
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.trust_remote_code = trust_remote_code
        self.supports_multi_image = True
        self.supports_gradients = False

    def load_model(self) -> None:
        """Load model using vLLM engine."""
        try:
            from vllm import LLM

            print(f"Loading {self.model_name} with vLLM backend...")

            # Use TORCH_SDPA for ViT attention on Blackwell GPUs (sm_100)
            # where pre-built flash attention kernels are incompatible
            if torch.cuda.is_available():
                cap = torch.cuda.get_device_capability()
                if cap[0] >= 10:  # Blackwell (B200) or newer
                    os.environ.setdefault("VLLM_VIT_ATTENTION_BACKEND", "TORCH_SDPA")
                    print(f"Detected Blackwell GPU (sm_{cap[0]}{cap[1]}), "
                          f"using TORCH_SDPA for ViT attention")

            # vLLM initialization arguments
            vllm_kwargs = {
                "model": self.model_name,
                "tensor_parallel_size": self.tensor_parallel_size,
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "trust_remote_code": self.trust_remote_code,
            }

            # Add optional arguments
            if self.max_model_len:
                vllm_kwargs["max_model_len"] = self.max_model_len

            # Dtype mapping
            if self.dtype == torch.bfloat16:
                vllm_kwargs["dtype"] = "bfloat16"
            elif self.dtype == torch.float16:
                vllm_kwargs["dtype"] = "float16"
            elif self.dtype == torch.float32:
                vllm_kwargs["dtype"] = "float32"

            self.model = LLM(**vllm_kwargs)

            # Load processor for preprocessing
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code
            )

            self._loaded = True
            print(f"vLLM model loaded successfully")

        except ImportError:
            raise ImportError(
                "vLLM not installed. Install with: pip install vllm"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load vLLM model: {e}")

    def preprocess(
        self,
        images: List[Image.Image],
        prompt: str,
        system_prompt: Optional[str] = None,
        use_manual_prompt: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Preprocess images and text for vLLM.

        vLLM uses a different input format - returns dict with prompt and images.

        Args:
            images: List of PIL Images
            prompt: Text prompt
            system_prompt: Optional system prompt
            use_manual_prompt: If True, use DriveBench official manual format instead of chat template
            **kwargs: Additional arguments

        Returns:
            Dictionary with 'prompt' and 'multi_modal_data'
        """
        if not self._loaded:
            self.load_model()

        # Validate inputs
        if not images:
            from PIL import Image as PILImage
            images = [PILImage.new('RGB', (224, 224), color='black')]
            print("Warning: Empty image list, using black placeholder")

        # DriveBench official manual prompt format
        if use_manual_prompt:
            # Format: "USER: <image>\n<image>\n... [SYSTEM_PROMPT] [QUESTION]\nASSISTANT:"
            text_prompt = "USER: "
            for _ in images:
                text_prompt += "<image>\n"
            if system_prompt:
                text_prompt += system_prompt + " "
            text_prompt += prompt + "\nASSISTANT:"
        else:
            # Build message content (similar to transformers)
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
                content.append({"type": "image", "image": img})

            # Add text
            content.append({"type": "text", "text": prompt or "Describe this image."})

            # Build messages
            messages = []

            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            messages.append({"role": "user", "content": content})

            # Apply chat template to get text prompt
            text_prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

        # vLLM format: text prompt + image list
        return {
            "prompt": text_prompt,
            "multi_modal_data": {"image": images},
        }

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
            config = GenerationConfig(max_new_tokens=128, do_sample=False)

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
        """Generate text using vLLM.

        Args:
            inputs: Dict with 'prompt' and 'multi_modal_data'
            config: Generation configuration
            _logprob_out: Optional dict to receive first-token logprobs
            **kwargs: Additional generation arguments

        Returns:
            Generated text string
        """
        if config is None:
            config = GenerationConfig(
                max_new_tokens=128,
                do_sample=False,
            )

        # vLLM sampling parameters
        from vllm import SamplingParams

        sp_kwargs = dict(
            max_tokens=config.max_new_tokens,
            temperature=config.temperature if config.do_sample else 0.0,
            top_p=config.top_p if config.do_sample else 1.0,
            n=1,
        )
        if config.top_k > 0:
            sp_kwargs["top_k"] = config.top_k
        if config.repetition_penalty != 1.0:
            sp_kwargs["repetition_penalty"] = config.repetition_penalty
        # Note: vLLM >=0.12 removed best_of/use_beam_search from SamplingParams.
        # When num_beams>1 is requested, fall back to greedy decoding.
        if config.num_beams > 1:
            logger.warning(
                "num_beams=%d requested but vLLM does not support beam search. "
                "Falling back to greedy decoding. This may degrade quality for "
                "datasets that rely on beam search (e.g. BDD-X/Dolphins).",
                config.num_beams,
            )
        # Add logprobs if requested
        if config and config.logprobs:
            sp_kwargs["logprobs"] = config.top_logprobs
        sampling_params = SamplingParams(**sp_kwargs)

        # Generate with multimodal data
        mm_data = inputs.get("multi_modal_data", {})
        generate_input = {"prompt": inputs["prompt"]}
        if mm_data:
            generate_input["multi_modal_data"] = mm_data
        outputs = self.model.generate(
            [generate_input],
            sampling_params,
        )

        # Extract generated text with bounds checking
        if not outputs or not outputs[0].outputs:
            return ""
        generated = outputs[0].outputs[0].text.strip()

        # Extract first-token logprobs
        if _logprob_out is not None and config and config.logprobs:
            try:
                token_lps = outputs[0].outputs[0].logprobs
                if token_lps and len(token_lps) > 0:
                    first = token_lps[0]  # dict[int, Logprob]
                    _logprob_out["logprobs"] = {
                        lp.decoded_token.strip().upper(): lp.logprob
                        for _, lp in first.items()
                        if lp.decoded_token and lp.decoded_token.strip().upper() in {"A", "B", "C", "D"}
                    }
            except Exception:
                pass

        return generated

    def batch_inference(
        self,
        batch_images: List[List[Image.Image]],
        batch_prompts: List[str],
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> List[ModelOutput]:
        """Run batch inference using vLLM's continuous batching.

        vLLM automatically optimizes batching for maximum throughput.

        Args:
            batch_images: List of image lists
            batch_prompts: List of prompts
            config: Generation configuration
            **kwargs: Additional arguments.

        Returns:
            List of ModelOutputs
        """

        if not self._loaded:
            self.load_model()

        if len(batch_images) != len(batch_prompts):
            raise ValueError(
                f"batch_images ({len(batch_images)}) and batch_prompts "
                f"({len(batch_prompts)}) must have the same length"
            )

        # Preprocess all samples
        all_inputs = []
        for images, prompt in zip(batch_images, batch_prompts):
            inputs = self.preprocess(images, prompt, **kwargs)
            all_inputs.append(inputs)

        # vLLM sampling parameters
        if config is None:
            config = GenerationConfig(max_new_tokens=128, do_sample=False)

        from vllm import SamplingParams

        sp_kwargs = dict(
            max_tokens=config.max_new_tokens,
            temperature=config.temperature if config.do_sample else 0.0,
            top_p=config.top_p if config.do_sample else 1.0,
            n=1,
        )
        if config.top_k > 0:
            sp_kwargs["top_k"] = config.top_k
        if config.repetition_penalty != 1.0:
            sp_kwargs["repetition_penalty"] = config.repetition_penalty
        # Note: vLLM >=0.12 removed best_of/use_beam_search from SamplingParams.
        # When num_beams>1 is requested, fall back to greedy decoding.
        if config.num_beams > 1:
            logger.warning(
                "num_beams=%d requested but vLLM does not support beam search. "
                "Falling back to greedy decoding. This may degrade quality for "
                "datasets that rely on beam search (e.g. BDD-X/Dolphins).",
                config.num_beams,
            )
        # Add logprobs if requested
        if config and config.logprobs:
            sp_kwargs["logprobs"] = config.top_logprobs
        sampling_params = SamplingParams(**sp_kwargs)

        # Prepare batch inputs with multimodal data
        generate_inputs = []
        for inp in all_inputs:
            item = {"prompt": inp["prompt"]}
            mm_data = inp.get("multi_modal_data", {})
            if mm_data:
                item["multi_modal_data"] = mm_data
            generate_inputs.append(item)

        # Generate (vLLM automatically batches)
        outputs = self.model.generate(
            generate_inputs,
            sampling_params,
        )

        # Format outputs with bounds checking + logprob extraction
        results = []
        for i, output in enumerate(outputs):
            text = ""
            logprobs_dict = None
            if output.outputs:
                text = output.outputs[0].text.strip()
                if config and config.logprobs and output.outputs[0].logprobs:
                    try:
                        token_lps = output.outputs[0].logprobs
                        if token_lps and len(token_lps) > 0:
                            first = token_lps[0]
                            logprobs_dict = {
                                lp.decoded_token.strip().upper(): lp.logprob
                                for _, lp in first.items()
                                if lp.decoded_token and lp.decoded_token.strip().upper() in {"A", "B", "C", "D"}
                            }
                    except Exception:
                        pass

            meta = {
                "model_name": self.model_name,
                "num_images": len(batch_images[i]) if i < len(batch_images) else 0,
                "prompt": batch_prompts[i] if i < len(batch_prompts) else "",
                "backend": "vllm",
            }
            if logprobs_dict:
                meta["logprobs"] = logprobs_dict
            results.append(ModelOutput(text=text, metadata=meta))

        return results
