"""Direct access to LLaVA's vision tower for gray-box patch extraction (ADVEDM-R)."""

import os
# Disable flash-attn to avoid compatibility issues.
os.environ["DISABLE_FLASH_ATTN"] = "1"

import torch
import torch.nn as nn
from typing import Tuple, Optional
from PIL import Image
import warnings

warnings.filterwarnings("ignore")


class LLaVAVisionEncoder:
    """Wrapper for the LLaVA vision tower with patch embedding extraction."""

    def __init__(
        self,
        model_path: str = "liuhaotian/llava-v1.5-7b",
        device: str = "cuda",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ):
        """Initialize the LLaVA vision encoder."""
        self.device = device
        self.model_path = model_path

        print(f"Loading LLaVA model from {model_path}...")

        try:
            from llava.model.builder import load_pretrained_model
            from llava.mm_utils import get_model_name_from_path

            model_name = get_model_name_from_path(model_path)
            (
                self.tokenizer,
                self.model,
                self.image_processor,
                self.context_len,
            ) = load_pretrained_model(
                model_path=model_path,
                model_base=None,
                model_name=model_name,
                load_8bit=load_in_8bit,
                load_4bit=load_in_4bit,
                device=device,
            )
            print("✓ Loaded LLaVA model using official loader")

        except ImportError as e:
            print(f"Warning: LLaVA package not found ({e})")
            print("Falling back to HuggingFace transformers...")
            from transformers import AutoTokenizer, AutoModelForCausalLM, CLIPImageProcessor

            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto" if device == "cuda" else None,
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
            )

            try:
                self.image_processor = CLIPImageProcessor.from_pretrained(model_path)
            except:
                from transformers import AutoProcessor
                self.image_processor = AutoProcessor.from_pretrained(model_path)

            self.context_len = 2048

        self.model.eval()

        self.vision_tower = self._extract_vision_tower()
        self.vision_tower.eval()

        print(f"✓ Vision tower extracted: {type(self.vision_tower)}")

        self.image_size = self._get_image_size()
        print(f"✓ Image size: {self.image_size}x{self.image_size}")

    def _extract_vision_tower(self) -> nn.Module:
        """Extract the vision tower module from the loaded LLaVA model."""
        possible_paths = [
            "model.vision_tower",
            "vision_tower",
            "model.model.vision_tower",
            "model.vision_model",
        ]

        vision_tower = None
        for path in possible_paths:
            try:
                parts = path.split(".")
                module = self.model
                for part in parts:
                    module = getattr(module, part)
                vision_tower = module
                print(f"  Found vision tower at: {path}")
                break
            except AttributeError:
                continue

        if vision_tower is None:
            raise ValueError("Could not find vision tower in LLaVA model")

        if hasattr(vision_tower, "vision_model"):
            return vision_tower.vision_model
        elif hasattr(vision_tower, "vision_tower"):
            return vision_tower.vision_tower

        return vision_tower

    def _get_image_size(self) -> int:
        """Get the expected input image size."""
        if hasattr(self.image_processor, "crop_size"):
            if isinstance(self.image_processor.crop_size, dict):
                return self.image_processor.crop_size.get("height", 336)
            else:
                return self.image_processor.crop_size
        elif hasattr(self.image_processor, "size"):
            if isinstance(self.image_processor.size, dict):
                return self.image_processor.size.get("height", 336)
            else:
                return self.image_processor.size

        # Default for LLaVA-1.5.
        return 336

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess a PIL image to a tensor [1, C, H, W]."""
        inputs = self.image_processor(images=image, return_tensors="pt")

        if "pixel_values" in inputs:
            pixel_values = inputs["pixel_values"]
        elif "image" in inputs:
            pixel_values = inputs["image"]
        else:
            raise ValueError(f"Unknown image processor output keys: {inputs.keys()}")

        return pixel_values.to(self.device)

    @torch.no_grad()
    def get_patch_embeddings(
        self, image: torch.Tensor, return_cls: bool = False, normalize: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Extract patch embeddings (and optionally CLS) from the vision tower."""
        if normalize:
            mean = torch.tensor([0.48145466, 0.45782750, 0.40821073]).view(1, 3, 1, 1).to(image.device)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(image.device)
            image = (image - mean) / std

        outputs = self.vision_tower(image)

        if isinstance(outputs, tuple):
            outputs = outputs[0]

        if hasattr(outputs, "last_hidden_state"):
            hidden_states = outputs.last_hidden_state
        elif isinstance(outputs, torch.Tensor):
            hidden_states = outputs
        else:
            raise ValueError(f"Unexpected vision tower output type: {type(outputs)}")

        cls_embed = hidden_states[:, 0, :]
        patch_embeds = hidden_states[:, 1:, :]

        if return_cls:
            return patch_embeds, cls_embed
        else:
            return patch_embeds, None

    def get_num_patches(self) -> int:
        """Return the number of patches based on image size and patch size."""
        if hasattr(self.vision_tower, "config"):
            config = self.vision_tower.config
            if hasattr(config, "patch_size"):
                patch_size = config.patch_size
                num_patches = (self.image_size // patch_size) ** 2
                return num_patches

        # Default for LLaVA-1.5 (ViT-L/14 @ 336x336).
        return 576


def load_clip_text_encoder(
    model_name: str = "ViT-B/32", device: str = "cuda"
) -> Tuple:
    """Load a CLIP text encoder for target text embedding."""
    try:
        import clip

        print(f"Loading CLIP text encoder: {model_name}")
        model, preprocess = clip.load(model_name, device=device)
        model.eval()
        print(f"✓ CLIP text encoder loaded")
        return model, preprocess

    except ImportError:
        print("OpenAI CLIP not found, using HuggingFace CLIP...")
        from transformers import CLIPModel, CLIPProcessor

        hf_model_name = f"openai/clip-vit-base-patch{model_name.split('/')[-1].lower()}"
        model = CLIPModel.from_pretrained(hf_model_name).to(device)
        processor = CLIPProcessor.from_pretrained(hf_model_name)
        model.eval()
        print(f"✓ CLIP text encoder loaded from HuggingFace")
        return model, processor


if __name__ == "__main__":
    print("Testing LLaVA Vision Encoder...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 60)
    print("Test 1: Loading LLaVA vision encoder")
    print("=" * 60)

    try:
        encoder = LLaVAVisionEncoder(
            model_path="liuhaotian/llava-v1.5-7b", device=device
        )
        print("✓ Vision encoder loaded successfully\n")

        print("=" * 60)
        print("Test 2: Extracting patch embeddings")
        print("=" * 60)

        dummy_image = torch.randn(1, 3, 336, 336).to(device)
        print(f"Input shape: {dummy_image.shape}")

        patch_embeds, cls_embed = encoder.get_patch_embeddings(
            dummy_image, return_cls=True
        )

        print(f"✓ Patch embeddings shape: {patch_embeds.shape}")
        print(f"✓ CLS embedding shape: {cls_embed.shape}")

        expected_patches = encoder.get_num_patches()
        assert (
            patch_embeds.shape[1] == expected_patches
        ), f"Expected {expected_patches} patches, got {patch_embeds.shape[1]}"
        print(f"✓ Number of patches matches: {expected_patches}\n")

        print("=" * 60)
        print("Test 3: Loading CLIP text encoder")
        print("=" * 60)

        clip_model, _ = load_clip_text_encoder("ViT-B/32", device=device)
        print("✓ CLIP text encoder loaded\n")

        print("=" * 60)
        print("Test 4: Encoding target text")
        print("=" * 60)

        import clip as clip_pkg

        text = clip_pkg.tokenize(["a pedestrian"]).to(device)
        text_features = clip_model.encode_text(text)
        print(f"✓ Text features shape: {text_features.shape}\n")

        print("=" * 60)
        print("All tests passed!")
        print("=" * 60)

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()
