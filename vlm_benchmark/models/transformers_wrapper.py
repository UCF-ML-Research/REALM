"""
Unified transformers backend wrapper for VLM inference.

Supports multiple model architectures with automatic detection:
- Qwen2.5-VL series
- LLaVA series
- InternVL series
- Dolphins (OpenFlamingo)

All models share the same interface but use architecture-specific preprocessing.
"""

from typing import List, Dict, Any, Optional, Union
from PIL import Image
import torch

from .base_model import ImageVLMWrapper, GenerationConfig


class TransformersWrapper(ImageVLMWrapper):
    """Unified wrapper for VLM models using transformers backend.

    Automatically detects model architecture and applies appropriate preprocessing.

    Supported architectures:
    - Qwen2.5-VL: Qwen/Qwen2.5-VL-*
    - LLaVA: liuhaotian/llava-*
    - InternVL: OpenGVLab/InternVL*
    - Dolphins: gray311/Dolphins
    """

    # Architecture detection patterns
    ARCHITECTURES = {
        'qwen': ['qwen2.5-vl', 'qwen/qwen2.5-vl', 'qwen2-vl'],
        'llava': ['llava'],
        'internvl': ['internvl'],
        'dolphins': ['dolphins'],
    }

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = True,
        architecture: Optional[str] = None,
        **kwargs
    ):
        """Initialize unified transformers wrapper.

        Args:
            model_name: HuggingFace model name or local path
            device: Device to load model on
            dtype: Model data type
            trust_remote_code: Trust remote code from HuggingFace
            architecture: Explicitly specify architecture (auto-detected if None)
            **kwargs: Architecture-specific arguments
        """
        super().__init__(model_name, device, dtype, backend="transformers", **kwargs)
        self.trust_remote_code = trust_remote_code
        self.architecture = architecture or self._detect_architecture(model_name)
        self.kwargs = kwargs

        # Set capabilities based on architecture
        if self.architecture in ['qwen', 'internvl']:
            self.supports_multi_image = True
        elif self.architecture == 'dolphins':
            # Video model
            self.supports_multi_image = True  # Multiple frames
        else:
            self.supports_multi_image = False

        print(f"Detected architecture: {self.architecture}")

    def _detect_architecture(self, model_name: str) -> str:
        """Auto-detect model architecture from name.

        Args:
            model_name: Model name or path

        Returns:
            Architecture identifier
        """
        model_lower = model_name.lower()

        for arch, patterns in self.ARCHITECTURES.items():
            if any(pattern in model_lower for pattern in patterns):
                return arch

        # Default to qwen if can't detect
        print(f"Warning: Could not detect architecture from {model_name}, defaulting to qwen")
        return 'qwen'

    def load_model(self) -> None:
        """Load model and processor based on architecture."""
        if self.architecture == 'qwen':
            self._load_qwen()
        elif self.architecture == 'llava':
            self._load_llava()
        elif self.architecture == 'internvl':
            self._load_internvl()
        elif self.architecture == 'dolphins':
            self._load_dolphins()
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")

        self._loaded = True
        print(f"Model loaded successfully ({self.architecture})")

    def _load_qwen(self) -> None:
        """Load Qwen2.5-VL model."""
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        print(f"Loading Qwen2.5-VL: {self.model_name}")

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            device_map=self.device,
            trust_remote_code=self.trust_remote_code,
        )

        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code
        )

    def _load_llava(self) -> None:
        """Load LLaVA model."""
        try:
            # Try llava package first
            from llava.model.builder import load_pretrained_model
            from llava.mm_utils import get_model_name_from_path

            print(f"Loading LLaVA with llava package: {self.model_name}")

            model_name = get_model_name_from_path(self.model_name)
            load_8bit = self.kwargs.get('load_in_8bit', False)
            load_4bit = self.kwargs.get('load_in_4bit', False)

            self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
                self.model_name,
                None,  # model_base
                model_name,
                load_8bit=load_8bit,
                load_4bit=load_4bit,
                device_map=self.device,
            )
            self.processor = self.image_processor

        except ImportError:
            # Fallback to transformers
            from transformers import AutoProcessor, LlavaForConditionalGeneration

            print(f"Loading LLaVA with transformers: {self.model_name}")

            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                device_map=self.device,
            )
            self.tokenizer = self.processor.tokenizer

    def _load_internvl(self) -> None:
        """Load InternVL model."""
        from transformers import AutoModel, AutoTokenizer

        print(f"Loading InternVL: {self.model_name}")

        self.model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            device_map=self.device,
            trust_remote_code=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

    def _load_dolphins(self) -> None:
        """Load Dolphins (OpenFlamingo) model."""
        from mllm.src.factory import create_model_and_transforms
        from peft import LoraConfig
        from huggingface_hub import hf_hub_download

        print(f"Loading Dolphins: {self.model_name}")

        # LoRA configuration
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
        )

        clip_model = self.kwargs.get('clip_model', 'ViT-L-14-336')
        lang_model = self.kwargs.get('lang_model', 'anas-awadalla/mpt-7b')

        # Create model
        self.model, self.image_processor, self.tokenizer = create_model_and_transforms(
            clip_vision_encoder_path=clip_model,
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path=lang_model,
            tokenizer_path=lang_model,
            cross_attn_every_n_layers=4,
            use_peft=True,
            peft_config=lora_config,
        )

        # Load checkpoint
        checkpoint_path = hf_hub_download(self.model_name, "checkpoint.pt")
        self.model.load_state_dict(torch.load(checkpoint_path), strict=False)

        # Move to device
        if self.device == "auto":
            self.model.half().cuda()
        else:
            self.model.half().to(self.device)

    def preprocess(
        self,
        images: List[Image.Image],
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Preprocess images and text (architecture-specific).

        Args:
            images: List of PIL Images
            prompt: Text prompt
            system_prompt: Optional system prompt
            **kwargs: Additional arguments

        Returns:
            Dictionary with model inputs
        """
        if not self._loaded:
            self.load_model()

        if self.architecture == 'qwen':
            return self._preprocess_qwen(images, prompt, system_prompt, **kwargs)
        elif self.architecture == 'llava':
            return self._preprocess_llava(images, prompt, **kwargs)
        elif self.architecture == 'internvl':
            return self._preprocess_internvl(images, prompt, **kwargs)
        elif self.architecture == 'dolphins':
            return self._preprocess_dolphins(images, prompt, **kwargs)
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")

    def _preprocess_qwen(
        self,
        images: List[Image.Image],
        prompt: str,
        system_prompt: Optional[str],
        **kwargs
    ) -> Dict[str, Any]:
        """Qwen-specific preprocessing following official example."""
        from qwen_vl_utils import process_vision_info

        # Build message content
        content = []
        for img in images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt})

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Process vision info (official approach)
        image_inputs, video_inputs = process_vision_info(messages)

        # Process inputs
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)

        return inputs

    def _preprocess_llava(
        self,
        images: List[Image.Image],
        prompt: str,
        **kwargs
    ) -> Dict[str, Any]:
        """LLaVA-specific preprocessing."""
        image = images[0] if images else Image.new('RGB', (224, 224))

        # Check if using llava package or transformers
        if hasattr(self, 'image_processor') and self.image_processor is not None:
            # Using llava package
            from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
            from llava.conversation import conv_templates
            from llava.mm_utils import tokenizer_image_token, process_images

            conv = conv_templates["llava_v1"].copy()
            prompt_with_image = DEFAULT_IMAGE_TOKEN + '\n' + prompt
            conv.append_message(conv.roles[0], prompt_with_image)
            conv.append_message(conv.roles[1], None)
            full_prompt = conv.get_prompt()

            image_tensor = process_images([image], self.image_processor, self.model.config)[0]
            input_ids = tokenizer_image_token(full_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

            return {
                "input_ids": input_ids.unsqueeze(0),
                "images": image_tensor.unsqueeze(0).half(),
            }
        else:
            # Using transformers
            full_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
            inputs = self.processor(text=full_prompt, images=image, return_tensors="pt").to(self.model.device)
            return inputs

    def _preprocess_internvl(
        self,
        images: List[Image.Image],
        prompt: str,
        **kwargs
    ) -> Dict[str, Any]:
        """InternVL-specific preprocessing."""
        # InternVL uses custom format
        return {
            "images": images,
            "prompt": prompt,
        }

    def _preprocess_dolphins(
        self,
        images: List[Image.Image],
        prompt: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Dolphins-specific preprocessing."""
        num_frames = self.kwargs.get('num_frames', 16)

        # Process video frames
        frames = self._process_video_frames(images, num_frames)

        # Stack processed frames
        vision_x = torch.stack([
            self.image_processor(frame) for frame in frames
        ], dim=0).unsqueeze(0).unsqueeze(0)

        # Format prompt
        formatted_prompt = f"USER: <image> is a driving video. {prompt} GPT:<answer>"

        # Tokenize
        inputs = self.tokenizer([formatted_prompt], return_tensors="pt")

        return {
            "vision_x": vision_x,
            "lang_x": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }

    def _process_video_frames(self, frames: List[Image.Image], target_frames: int) -> List[Image.Image]:
        """Process video frames to target count."""
        # Critical Fix #1: Check for empty frames list
        if not frames:
            raise ValueError("frames list cannot be empty")

        if len(frames) == target_frames:
            return frames

        if len(frames) > target_frames:
            # Uniform sample
            step = len(frames) / target_frames
            indices = [int(i * step) for i in range(target_frames)]
            return [frames[i] for i in indices]

        # Duplicate frames to reach target
        while len(frames) < target_frames:
            frames.append(frames[-1].copy())
        return frames[:target_frames]

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
        if config is None:
            config = GenerationConfig(
                max_new_tokens=256,
                do_sample=False,
            )

        if self.architecture == 'qwen':
            return self._generate_qwen(inputs, config, **kwargs)
        elif self.architecture == 'llava':
            return self._generate_llava(inputs, config, **kwargs)
        elif self.architecture == 'internvl':
            return self._generate_internvl(inputs, config, **kwargs)
        elif self.architecture == 'dolphins':
            return self._generate_dolphins(inputs, config, **kwargs)
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")

    def _generate_qwen(self, inputs: Dict[str, Any], config: GenerationConfig, **kwargs) -> str:
        """Qwen-specific generation."""
        gen_kwargs = config.to_dict()
        gen_kwargs['pad_token_id'] = self.processor.tokenizer.pad_token_id

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        # Decode only new tokens
        generated = self.processor.batch_decode(
            [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)],
            skip_special_tokens=True
        )[0].strip()

        return generated

    def _generate_llava(self, inputs: Dict[str, Any], config: GenerationConfig, **kwargs) -> str:
        """LLaVA-specific generation."""
        device = next(self.model.parameters()).device
        gen_kwargs = config.to_dict()

        with torch.no_grad():
            if "images" in inputs:
                # Using llava package format
                output_ids = self.model.generate(
                    inputs["input_ids"].to(device),
                    images=inputs["images"].to(device, dtype=self.dtype),
                    **gen_kwargs
                )
                generated = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                if "ASSISTANT:" in generated:
                    generated = generated.split("ASSISTANT:")[-1].strip()
            else:
                # Using transformers format
                output_ids = self.model.generate(**inputs, **gen_kwargs)
                generated = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
                if "ASSISTANT:" in generated:
                    generated = generated.split("ASSISTANT:")[-1].strip()

        return generated

    def _generate_internvl(self, inputs: Dict[str, Any], config: GenerationConfig, **kwargs) -> str:
        """InternVL-specific generation."""
        generation_config = {
            'max_new_tokens': config.max_new_tokens,
            'do_sample': config.do_sample,
        }
        if config.do_sample:
            generation_config['temperature'] = config.temperature

        # InternVL uses custom .chat() method
        with torch.no_grad():
            # Load images
            if len(inputs["images"]) == 1:
                pixel_values = self._load_internvl_image(inputs["images"][0]).to(
                    self.model.device, dtype=self.dtype
                )
                question = f'<image>\n{inputs["prompt"]}'
            else:
                pixel_values = torch.cat([
                    self._load_internvl_image(img) for img in inputs["images"]
                ], dim=0).to(self.model.device, dtype=self.dtype)
                question = ''.join([f'Image-{i+1}: <image>\n' for i in range(len(inputs["images"]))])
                question += inputs["prompt"]

            response = self.model.chat(self.tokenizer, pixel_values, question, generation_config)

        return response

    def _load_internvl_image(self, image: Image.Image) -> torch.Tensor:
        """Load and preprocess image for InternVL."""
        try:
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
            return processor(images=image, return_tensors="pt")["pixel_values"]
        except Exception:
            # Fallback to basic processing
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((448, 448)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            return transform(image).unsqueeze(0)

    def _generate_dolphins(self, inputs: Dict[str, Any], config: GenerationConfig, **kwargs) -> str:
        """Dolphins-specific generation."""
        gen_kwargs = {
            'max_new_tokens': config.max_new_tokens,
            'temperature': config.temperature,
            'top_k': config.top_k,
            'top_p': config.top_p,
            'no_repeat_ngram_size': config.no_repeat_ngram_size,
            'length_penalty': config.length_penalty,
            'do_sample': config.do_sample,
            'early_stopping': True,
        }

        device = next(self.model.parameters()).device

        with torch.no_grad():
            generated_tokens = self.model.generate(
                vision_x=inputs["vision_x"].half().to(device),
                lang_x=inputs["lang_x"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
                num_beams=config.num_beams,
                **gen_kwargs,
            )

        # Decode
        generated_tokens = generated_tokens.cpu().numpy()
        if isinstance(generated_tokens, tuple):
            generated_tokens = generated_tokens[0]

        generated_text = self.tokenizer.batch_decode(generated_tokens)[0]

        # Extract answer after <answer> token
        if "<answer>" in generated_text:
            generated_text = generated_text.split("<answer>")[-1]

        # Clean up
        generated_text = generated_text.replace("<|endofchunk|>", "").strip()

        return generated_text

    def forward_with_gradients(
        self,
        images: Union[List[Image.Image], List[torch.Tensor]],
        prompt: str,
        ground_truth: str,
        return_loss: bool = True,
        pil_images: Optional[List[Image.Image]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with gradient computation enabled.

        This method is designed for gradient-based attacks (FGSM, PGD, CAD).
        Unlike inference(), it:
        1. Does NOT use torch.no_grad()
        2. Does NOT use model.generate()
        3. Uses direct model.forward() with labels
        4. Returns differentiable loss tensor

        Args:
            images: List of PIL Images OR torch.Tensors [C,H,W]
            prompt: Text prompt/question
            ground_truth: Expected answer for loss computation
            return_loss: If True, compute and return loss
            pil_images: Original PIL images (required when images are tensors, for sizing info)

        Returns:
            Dict with keys:
                - 'loss': Cross-entropy loss (if return_loss=True)
                - 'logits': Model output logits
                - 'pixel_values': Preprocessed image tensor (for debugging)
        """
        if not self._loaded:
            self.load_model()

        # Step 1: Format prompt with chat template (includes image tokens)
        # This is critical for Qwen models which need proper image token placement
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        text = self.processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Step 2: Handle tensor vs PIL input for image preprocessing
        if isinstance(images[0], torch.Tensor):
            # Images are already tensors (from CAD optimization)
            # Use differentiable preprocessing
            if pil_images is None:
                raise ValueError("pil_images must be provided when images are tensors (needed for sizing info)")

            from ..attacks.gradient_attacks.diff_preprocess import create_diff_preprocessor

            # First, process with PIL images to get correct tokenization
            # This ensures image tokens match the number of features
            inputs = self.processor(
                text=[text],
                images=pil_images,  # Use PIL images for tokenization
                return_tensors="pt",
                padding=True
            )
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            grid_thw = inputs.get('image_grid_thw', None)

            # Now use differentiable preprocessing for pixel_values (preserves gradients)
            diff_preprocess = create_diff_preprocessor(
                self.model_name,
                self.processor,
                pil_images,  # PIL images for sizing info
                device=self.device
            )

            # Stack tensors and preprocess (this gives us differentiable pixel_values)
            image_tensors = torch.stack(images) if len(images) > 1 else images[0].unsqueeze(0)
            pixel_values, grid_thw = diff_preprocess(image_tensors)

        else:
            # Images are PIL (normal case)
            # Use standard preprocessing (not differentiable, but fine for non-attack usage)
            inputs = self.processor(
                text=[text],
                images=images,
                return_tensors="pt"
            )
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            pixel_values = inputs['pixel_values'].to(self.device)
            grid_thw = inputs.get('image_grid_thw', None)

        # Step 3: Create labels for loss computation
        # For autoregressive models, labels should be the same as input_ids
        # The model will automatically shift them for next-token prediction
        if return_loss:
            # Append ground truth to the prompt for teacher forcing
            # This follows FGSM/PGD approach
            full_text_with_gt = text + ground_truth

            # Re-tokenize with ground truth appended (if using tensors)
            if isinstance(images[0], torch.Tensor):
                inputs_with_gt = self.processor(
                    text=[full_text_with_gt],
                    images=pil_images,
                    return_tensors="pt",
                    padding=True
                )
                input_ids = inputs_with_gt['input_ids'].to(self.device)
                attention_mask = inputs_with_gt['attention_mask'].to(self.device)

            else:
                # For PIL images, retokenize with ground truth
                inputs_with_gt = self.processor(
                    text=[full_text_with_gt],
                    images=images,
                    return_tensors="pt"
                )
                input_ids = inputs_with_gt['input_ids'].to(self.device)
                attention_mask = inputs_with_gt['attention_mask'].to(self.device)

            # Labels are the same as input_ids for autoregressive loss
            labels = input_ids.clone()
        else:
            labels = None

        # Step 4: Forward pass (NO torch.no_grad()!)
        # This is the key difference from inference()
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=grid_thw,  # Qwen-specific
            labels=labels,  # Enables loss computation
            return_dict=True,
        )

        result = {
            'logits': outputs.logits,
            'pixel_values': pixel_values,
        }

        if return_loss and labels is not None:
            result['loss'] = outputs.loss

        return result
