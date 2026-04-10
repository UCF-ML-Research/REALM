"""AdvDiffVLM attack wrapper for VLM benchmark framework."""

import sys
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path
import torch
import torchvision.transforms as transforms
import torchvision.datasets as tv_datasets
from PIL import Image
import numpy as np
import csv

from ..base_attack import BaseAttack, AttackConfig, AttackResult
from ...data import Sample


@dataclass
class AdvDiffVLMConfig(AttackConfig):
    """Configuration for AdvDiffVLM attack."""

    # Diffusion parameters
    ddim_steps: int = 200
    ddim_eta: float = 0.0
    guidance_scale: float = 5.0

    # AEGE parameters
    gradient_scale: float = 35
    gradient_clip: float = 0.0025
    refinement_iterations: int = 10

    # CLIP ensemble
    clip_models: List[str] = field(default_factory=lambda: ["RN50", "RN101", "ViT-B/16", "ViT-B/32"])

    # GradCAM
    gradcam_resolution: int = 64
    gradcam_layer: str = "layer4"
    auto_generate_masks: bool = True
    gradcam_masks_dir: Optional[str] = None

    # Target
    target_strategy: str = "stop_sign"
    target_images_dir: Optional[str] = None
    class_label: Optional[int] = None

    # Image resolution (fixed for cin256-v2)
    image_resolution: int = 256


class AdvDiffVLMAttack(BaseAttack):
    """
    AdvDiffVLM attack wrapper integrating with VLM benchmark.

    Implements diffusion-based adversarial examples using Latent Diffusion
    with AEGE (Adaptive Ensemble Gradient Estimation) for transferability.
    """

    def __init__(self, config: AdvDiffVLMConfig):
        super().__init__(config)
        self.config: AdvDiffVLMConfig = config

        # Lazy initialization (models are heavy ~4GB total)
        self._ldm_model = None
        self._ddim_sampler = None
        self._clip_models = []
        self._clip_preprocess = None
        self._gradcam_generator = None
        self._target_paths_cache = None
        self._label_map_cache = None

    def _initialize_models(self):
        """Lazy load Latent Diffusion + CLIP ensemble."""
        if self._ldm_model is not None:
            return  # Already initialized

        print("Loading AdvDiffVLM models (LDM + CLIP ensemble)...")

        # Make ldm/ and taming/ importable (both live in advdiffvlm/)
        _attack_dir = str(Path(__file__).parent)
        if _attack_dir not in sys.path:
            sys.path.insert(0, _attack_dir)

        # Import from legacy code - DO NOT RECREATE
        from omegaconf import OmegaConf
        from ldm.util import instantiate_from_config
        from ldm.models.diffusion.ddim_main import DDIMSampler
        import clip

        # Get paths from config
        from .config import get_checkpoint_path, get_config_yaml_path, get_clip_cache_dir

        config_yaml = get_config_yaml_path()
        checkpoint = get_checkpoint_path()
        clip_cache = get_clip_cache_dir()

        # Load Latent Diffusion Model (EXACT LEGACY CODE)
        print(f"Loading LDM checkpoint: {checkpoint}")
        config = OmegaConf.load(config_yaml)
        pl_sd = torch.load(checkpoint, map_location=self.config.device)
        sd = pl_sd["state_dict"]
        # Pad cond_stage embedding from 1000→1001 (null class for classifier-free guidance)
        emb_key = "cond_stage_model.embedding.weight"
        if emb_key in sd and sd[emb_key].shape[0] == 1000:
            sd[emb_key] = torch.cat([sd[emb_key], torch.zeros(1, sd[emb_key].shape[1], device=sd[emb_key].device)], dim=0)
        model = instantiate_from_config(config.model)
        model.load_state_dict(sd, strict=False)
        model.to(self.config.device)
        model.eval()
        model.requires_grad_(False)
        self._ldm_model = model

        # Load CLIP ensemble (EXACT LEGACY CODE)
        print(f"Loading CLIP ensemble: {self.config.clip_models}")
        for clip_name in self.config.clip_models:
            clip_model, _ = clip.load(
                clip_name,
                device=self.config.device,
                download_root=clip_cache
            )
            clip_model.eval()
            clip_model.requires_grad_(False)
            self._clip_models.append(clip_model)

        # CLIP preprocessing (EXACT LEGACY from main.py lines 99-105)
        # CRITICAL: Must include Resize + CenterCrop + Normalize!
        # The DDIM sampler outputs 256x256 images that need to be resized to 224x224 for CLIP
        clip_input_res = self._clip_models[0].visual.input_resolution  # Should be 224
        self._clip_preprocess = transforms.Compose([
            transforms.Resize(clip_input_res, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(clip_input_res),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])

        # Load DDIM Sampler with AEGE (EXACT LEGACY CODE from main.py line 160)
        # CRITICAL: Pass models and preprocess to sampler!
        self._ddim_sampler = DDIMSampler(model, models=self._clip_models, preprocess=self._clip_preprocess)

        print("AdvDiffVLM models loaded successfully")

    def _initialize_gradcam(self):
        """Lazy load GradCAM generator."""
        if self._gradcam_generator is not None:
            return

        from .gradcam import GradCAMGenerator
        self._gradcam_generator = GradCAMGenerator(
            layer_name=self.config.gradcam_layer,
            resolution=self.config.gradcam_resolution,
            device=self.config.device
        )

    def generate(self, model, sample: Sample, **kwargs) -> AttackResult:
        """Generate adversarial example using AdvDiffVLM."""
        # 1. Initialize models
        self._initialize_models()

        # 2. Prepare images
        clean_image = self._prepare_image(sample.images[0])

        # 3. Load target image (legacy pairing by ImageFolder order)
        sample_idx = kwargs.get("sample_idx", None)
        target_image = self._load_target_image(sample, sample_idx)

        # 4. Auto-detect class_label from source if not set (legacy: source's ImageNet class)
        if self.config.class_label is None:
            self.config.class_label = self._infer_class_label(sample.images[0])

        # 5. Load/generate GradCAM mask
        gradcam_mask = self._load_or_generate_gradcam_mask(sample)
        # Ensure mask is on CPU for legacy DDIM sampler (it does .numpy() internally)
        gradcam_mask = gradcam_mask.cpu()

        # 6. Extract target CLIP features
        target_features = self._extract_clip_features(target_image)

        # 7. Encode clean image to latent
        with torch.no_grad():
            with self._ldm_model.ema_scope():
                # clean_image is already in [-1, 1] from _prepare_image(), no conversion needed
                encoder_posterior = self._ldm_model.encode_first_stage(clean_image)
                z = self._ldm_model.get_first_stage_encoding(encoder_posterior).detach()

        # 8. Prepare conditioning
        class_label = self._get_class_label(sample)
        xc = torch.tensor([class_label], device=self.config.device)

        with self._ldm_model.ema_scope():
            c = self._ldm_model.get_learned_conditioning({self._ldm_model.cond_stage_key: xc})
            uc = self._ldm_model.get_learned_conditioning(
                {self._ldm_model.cond_stage_key: torch.tensor([1000]).to(self.config.device)}
            )

        # 9. Run refinement loop (EXACT LEGACY CODE)
        print(f"Running AdvDiffVLM with {self.config.refinement_iterations} refinement iterations...")
        latent = z
        for refinement_iter in range(self.config.refinement_iterations):
            # Detach latent to prevent gradient graph accumulation across
            # refinement iterations (AEGE grads are per-timestep, not cross-iter)
            latent = latent.detach()

            # Call DDIMSampler.sample() - EXACT LEGACY INTERFACE
            latent, _ = self._ddim_sampler.sample(
                S=self.config.ddim_steps,
                conditioning=c,
                x_T=latent,
                batch_size=1,
                shape=[3, 64, 64],
                verbose=False,
                unconditional_guidance_scale=self.config.guidance_scale,
                unconditional_conditioning=uc,
                eta=self.config.ddim_eta,
                label=xc,
                tgt_image_features_list=target_features,
                org_image_features_list=None,
                cam=gradcam_mask,
                K=1,
                s=self.config.gradient_scale,
                a=5
            )
            if (refinement_iter + 1) % 5 == 0:
                print(f"  Refinement {refinement_iter + 1}/{self.config.refinement_iterations}")

        # 10. Decode latent to image
        with torch.no_grad():
            adv_image = self._ldm_model.decode_first_stage(latent.detach())
            adv_image = torch.clamp((adv_image + 1.0) / 2.0, 0, 1)
            # Ensure tensor is on CPU for PIL conversion
            adv_image = adv_image.cpu()
        # Free GPU memory after generation
        del latent
        torch.cuda.empty_cache()

        # 11. Convert to PIL
        adv_pil = self._tensor_to_pil(adv_image)

        # 12. Return result (success evaluated by framework)
        return AttackResult(
            success=False,  # Evaluated by framework
            adversarial_sample=adv_pil,
            original_output="",
            adversarial_output="",
            perturbation_norm=0.0,  # Diffusion is unbounded
            queries=1,
            metadata={
                "ddim_steps": self.config.ddim_steps,
                "refinement_iterations": self.config.refinement_iterations,
                "gradient_scale": self.config.gradient_scale,
                "clip_models": self.config.clip_models,
                "class_label": class_label,
                "target_strategy": self.config.target_strategy,
            }
        )

    def _prepare_image(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL image to tensor [1, 3, 256, 256] in [-1, 1] (legacy)."""
        transform = transforms.Compose([
            transforms.Resize(self.config.image_resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(self.config.image_resolution),
            transforms.ToTensor(),  # [0, 1]
            transforms.Lambda(lambda img: (img * 2.0 - 1.0)),  # [-1, 1]
        ])
        tensor = transform(image).unsqueeze(0).to(self.config.device)
        return tensor

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert tensor [1, 3, H, W] to PIL image."""
        # Remove batch dimension and move to CPU
        img_np = tensor.squeeze(0).cpu().numpy()

        # Convert [C, H, W] -> [H, W, C]
        img_np = np.transpose(img_np, (1, 2, 0))

        # Scale to [0, 255]
        img_np = (img_np * 255).astype(np.uint8)

        return Image.fromarray(img_np)

    def _load_target_image(self, sample: Sample, sample_idx: Optional[int]) -> Image.Image:
        """Load target image by ImageFolder order (legacy pairing)."""
        if self.config.target_images_dir is None:
            raise ValueError("target_images_dir must be specified")
        if sample_idx is None:
            raise ValueError("sample_idx is required to match legacy target pairing")

        if self._target_paths_cache is None:
            target_dir = Path(self.config.target_images_dir)
            if not target_dir.exists():
                raise FileNotFoundError(f"Target images dir not found: {target_dir}")
            dataset = tv_datasets.ImageFolder(str(target_dir))
            self._target_paths_cache = [p for (p, _) in dataset.samples]

        if sample_idx < 0 or sample_idx >= len(self._target_paths_cache):
            raise IndexError(
                f"sample_idx {sample_idx} out of range for target dataset "
                f"(size={len(self._target_paths_cache)})"
            )

        target_path = self._target_paths_cache[sample_idx]
        return Image.open(target_path).convert("RGB")

    def _load_or_generate_gradcam_mask(self, sample: Sample) -> torch.Tensor:
        """Load or auto-generate GradCAM mask [64, 64] using legacy PNG pipeline."""
        if self.config.gradcam_masks_dir is None:
            raise ValueError("gradcam_masks_dir must be specified")

        masks_dir = Path(self.config.gradcam_masks_dir)
        masks_dir.mkdir(parents=True, exist_ok=True)

        # Determine mask filename
        if self.config.class_label is not None:
            # Direct mode: class_label provided externally, no CSV needed
            mask_id = str(sample.id)
        else:
            # Legacy mode: use images_subset.csv mapping
            if self._label_map_cache is None:
                self._label_map_cache = self._load_images_subset_mapping()
            name_key, _ = self._label_map_cache

            label_id = None
            if hasattr(sample, "metadata") and sample.metadata.get("image_file"):
                label_id = Path(sample.metadata["image_file"]).stem
            elif sample.id:
                label_id = str(sample.id)
            if label_id is None:
                raise ValueError("Cannot determine label_id for GradCAM mask naming")
            if label_id not in name_key:
                raise KeyError(f"Label id '{label_id}' not found in images_subset.csv index map")
            mask_id = name_key[label_id]

        mask_file = masks_dir / f"{mask_id}.png"

        # Try to load cached mask
        if mask_file.exists():
            import cv2
            cam = cv2.imread(str(mask_file), 0)
            if cam is None:
                raise FileNotFoundError(f"Failed to read mask image: {mask_file}")
            cam = cam / 255.0
            cam = cv2.resize(cam, (64, 64))
            return torch.tensor(cam, dtype=torch.float32, device=self.config.device)

        # Auto-generate if enabled
        if self.config.auto_generate_masks:
            print(f"Auto-generating GradCAM mask for {mask_id}...")
            self._initialize_gradcam()

            class_label = self._get_class_label(sample)
            mask = self._gradcam_generator.generate(sample.images[0], class_label)

            # Legacy-aligned cache: save as PNG then reload via cv2 pipeline
            cam_np = (mask.clamp(0, 1).cpu().numpy() * 255).astype("uint8")
            Image.fromarray(cam_np).save(mask_file)
            print(f"  Cached mask to {mask_file}")

            import cv2
            cam = cv2.imread(str(mask_file), 0)
            cam = cam / 255.0
            cam = cv2.resize(cam, (64, 64))
            return torch.tensor(cam, dtype=torch.float32, device=self.config.device)

        raise FileNotFoundError(f"GradCAM mask not found and auto_generate_masks=False: {mask_file}")

    def _infer_class_label(self, source_image: Image.Image) -> int:
        """Infer ImageNet class label from source image using GradCAM's ResNet50.

        Aligned with legacy: class_label is the source image's ImageNet class,
        used for both GradCAM spatial attention and LDM conditioning.
        """
        self._initialize_gradcam()
        with torch.no_grad():
            img_tensor = self._gradcam_generator.preprocess(source_image).unsqueeze(0).to(self.config.device)
            logits = self._gradcam_generator.model(img_tensor)
            label = int(logits.argmax(dim=1).item())
        print(f"Auto-detected class_label={label} from source image")
        return label

    def _get_class_label(self, sample: Sample) -> int:
        """Get ImageNet class label from config or legacy CSV mapping."""
        if self.config.class_label is not None:
            return self.config.class_label

        if self._label_map_cache is None:
            self._label_map_cache = self._load_images_subset_mapping()

        # Legacy uses clean image stem to map to ImageId via index->ImageId mapping
        label_id = None
        if hasattr(sample, "metadata") and sample.metadata.get("image_file"):
            label_id = Path(sample.metadata["image_file"]).stem
        elif sample.id:
            label_id = str(sample.id)

        if label_id is None:
            raise ValueError("Cannot determine label_id from sample for legacy mapping")

        name_key, labels = self._label_map_cache
        if label_id not in name_key:
            raise KeyError(f"Label id '{label_id}' not found in images_subset.csv index map")
        image_id = name_key[label_id]
        if image_id not in labels:
            raise KeyError(f"ImageId '{image_id}' not found in labels map from images_subset.csv")
        return labels[image_id]

    def _load_images_subset_mapping(self):
        """Load legacy ImageId/TrueLabel mapping from images_subset.csv."""
        csv_path = Path(__file__).parent / "data" / "images_subset.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"images_subset.csv not found: {csv_path}")

        name_key = {}
        labels = {}
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        for i, row in enumerate(rows):
            key = f"{i:05d}"
            image_id = row["ImageId"]
            name_key[key] = image_id
            labels[image_id] = int(row["TrueLabel"])
        return name_key, labels

    def _extract_clip_features(self, image: Image.Image) -> List[torch.Tensor]:
        """Extract CLIP features from all ensemble models (EXACT LEGACY from main.py lines 204-209)."""
        # CRITICAL: Use each CLIP model's input_resolution (legacy main.py line 101)
        # RN models use 224, ViT models may use different sizes
        input_res = self._clip_models[0].visual.input_resolution  # All models should use same res

        # Legacy target transform: resize+centercrop+ToTensor+clamp(0-1)
        transform = transforms.Compose([
            transforms.Resize(input_res, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(input_res),
            transforms.ToTensor(),
            transforms.Lambda(lambda img: torch.clamp(img, 0.0, 1.0)),
        ])
        image_tgt = transform(image).unsqueeze(0).to(self.config.device)  # [1, 3, H, W]

        # EXACT LEGACY CODE (main.py lines 204-209)
        with torch.no_grad():
            tgt_image_features_list = []
            image_tgt = self._clip_preprocess(image_tgt)  # Apply normalization
            for clip_model in self._clip_models:
                tgt_image_features = clip_model.encode_image(image_tgt)  # [bs, 512]
                tgt_image_features = tgt_image_features / tgt_image_features.norm(dim=1, keepdim=True)
                tgt_image_features_list.append(tgt_image_features)

        return tgt_image_features_list

    def is_gradient_based(self) -> bool:
        """Return whether this attack requires gradients."""
        return True
