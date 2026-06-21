"""PhysPatch physical adversarial patch attack for the VLM benchmark."""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path
import torch
import torchvision.transforms as transforms
from PIL import Image

from ..base_attack import BaseAttack, AttackConfig, AttackResult
from ...data import Sample


@dataclass
class PhysPatchConfig(AttackConfig):
    """Configuration for PhysPatch attack."""

    epsilon: float = 16.0
    max_iterations: int = 300
    alpha: float = 1.0

    attack_method: str = "pgd"  # "pgd" or "mifgsm"
    decay: float = 1.0  # Momentum decay for MI-FGSM

    backbone: List[str] = field(default_factory=lambda: ["B16", "B32", "Laion"])
    K: int = 8  # SVD components for ensemble

    coords_file: Optional[str] = None
    source_images_dir: Optional[str] = None

    target_strategy: str = "stop_sign"
    target_images_dir: Optional[str] = None

    crop_scale_min: float = 0.5
    crop_scale_max: float = 0.9



class PhysPatchAttack(BaseAttack):
    """PhysPatch attack wrapper integrating physically realizable patches with the VLM benchmark."""

    def __init__(self, config: PhysPatchConfig):
        super().__init__(config)
        self.config: PhysPatchConfig = config

        # Lazy initialization (models are heavy ~2GB)
        self._ensemble_extractor = None
        self._ensemble_loss = None
        self._coords = None
        self._target_image_paths = None
        self._source_size = None  # (H, W) set per-sample in generate()

    def _initialize_models(self):
        """Lazy load CLIP surrogate models."""
        if self._ensemble_extractor is not None:
            return

        print("Loading CLIP ensemble models...")

        from vlm_benchmark.attacks.physpatch.surrogates import (
            ClipB16FeatureExtractor,
            ClipB32FeatureExtractor,
            ClipLaionFeatureExtractor,
            EnsembleFeatureExtractor_ours,
            EnsembleFeatureLoss_ours_auto,
        )

        models = []
        for backbone in self.config.backbone:
            if backbone == "B16":
                model = ClipB16FeatureExtractor().eval().to(self.config.device)
                model.requires_grad_(False)
                models.append(model)
            elif backbone == "B32":
                model = ClipB32FeatureExtractor().eval().to(self.config.device)
                model.requires_grad_(False)
                models.append(model)
            elif backbone == "Laion":
                model = ClipLaionFeatureExtractor().eval().to(self.config.device)
                model.requires_grad_(False)
                models.append(model)
            print(f"  ✓ Loaded {backbone}")

        self._ensemble_extractor = EnsembleFeatureExtractor_ours(models, k=self.config.K)
        self._ensemble_loss = EnsembleFeatureLoss_ours_auto(models, k=self.config.K)

        print(f"✓ Ensemble ready with K={self.config.K} SVD components\n")

    def _load_coordinates(self) -> torch.Tensor:
        """Load patch placement coordinates from file, generating if needed."""
        if self._coords is not None:
            return self._coords

        from pathlib import Path
        import os

        if self.config.coords_file is None:
            if not self.config.source_images_dir:
                raise ValueError(
                    "PhysPatch requires either coords_file or source_images_dir "
                    "for coordinate auto-generation."
                )
            default_path = Path(self.config.source_images_dir).parent / "coordinates" / "auto.txt"
            self.config.coords_file = str(default_path)

        coords_path = Path(self.config.coords_file)

        if not coords_path.exists():
            print(f"Coordinates not found: {self.config.coords_file}")
            print("Auto-generating via SoM pipeline...")

            clean_images_dir = self.config.source_images_dir
            if not clean_images_dir:
                # Infer from coords file path (legacy layout)
                dataset_root = coords_path.parent.parent
                clean_images_dir = str(dataset_root / "images" / "clean")

            try:
                from .coordinate_generator import ensure_coordinates

                api_key = os.environ.get('OPENAI_API_KEY')

                self.config.coords_file = ensure_coordinates(
                    clean_images_dir=clean_images_dir,
                    coords_file=str(coords_path),
                    openai_api_key=api_key,
                    device=self.config.device
                )
                print()

            except Exception as e:
                raise RuntimeError(
                    f"Coordinate generation failed: {e}\n"
                    f"Please ensure:\n"
                    f"  1. OPENAI_API_KEY environment variable is set\n"
                    f"  2. SoM dependencies are installed\n"
                    f"  3. SAM checkpoint exists at assets/checkpoints/\n"
                    f"Or provide a valid coords_file path."
                )

        # Supports keyed "stem, x, y" lines and legacy positional "x, y" lines.
        coords_dict = {}
        coords_list = []
        with open(self.config.coords_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) == 3:
                    stem = parts[0].strip()
                    x, y = float(parts[1]), float(parts[2])
                    coords_dict[stem] = [x, y]
                elif len(parts) == 2:
                    x, y = float(parts[0]), float(parts[1])
                    coords_list.append([x, y])

        if coords_dict:
            self._coords = coords_dict
            print(f"Loaded {len(coords_dict)} keyed coordinates from {self.config.coords_file}")
        else:
            self._coords = torch.tensor(coords_list, device=self.config.device)
            print(f"Loaded {len(coords_list)} positional coordinates from {self.config.coords_file}")
        return self._coords

    def _prepare_image(self, image: Image.Image, target_size: tuple = None) -> torch.Tensor:
        """Convert a PIL image to a tensor in [0, 255] range (not normalized)."""
        import numpy as np

        image = image.convert("RGB")
        if target_size is not None:
            image = transforms.Resize(
                target_size,
                interpolation=transforms.InterpolationMode.BICUBIC
            )(image)

        mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
        img_array = np.array(image, mode_to_nptype.get(image.mode, np.uint8), copy=True)
        img_tensor = torch.from_numpy(img_array)
        img_tensor = img_tensor.view(image.size[1], image.size[0], len(image.getbands()))
        img_tensor = img_tensor.permute(2, 0, 1).contiguous().float()

        return img_tensor.unsqueeze(0).to(self.config.device)

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert a [0, 1] tensor back to a PIL image."""
        import numpy as np

        tensor = torch.clamp(tensor.squeeze(0), 0, 1)

        tensor = (tensor * 255).cpu().byte()

        img_array = tensor.permute(1, 2, 0).numpy()
        return Image.fromarray(img_array, mode='RGB')

    def _load_target_image(self, sample_id: str, sample_idx: Optional[int]) -> Optional[torch.Tensor]:
        """Load target image for the attack (legacy-compatible, strict matching)."""
        if not self.config.target_images_dir:
            return None

        target_dir = Path(self.config.target_images_dir)
        if not target_dir.exists():
            raise FileNotFoundError(f"Target images dir not found: {target_dir}")

        if self._target_image_paths is None:
            exts = (".png", ".jpg", ".jpeg")
            paths = sorted([p for p in target_dir.iterdir() if p.suffix.lower() in exts])
            if not paths:
                raise FileNotFoundError(f"No target images found in {target_dir}")
            self._target_image_paths = paths

        stem_map = {p.stem: p for p in self._target_image_paths}
        if sample_id not in stem_map:
            raise ValueError(
                f"Missing target image for sample_id={sample_id}. "
                f"Expected file named {sample_id}.* in {target_dir}"
            )
        target_path = stem_map[sample_id]

        target_image = Image.open(target_path).convert("RGB")
        return self._prepare_image(target_image, target_size=self._source_size)

    def _create_crops(self, coords: torch.Tensor, image_size: tuple):
        """Create source and target crop functions for the given coordinate and image size."""
        from vlm_benchmark.attacks.physpatch.core.utils import RandomPointConstrainedCrop

        size = list(image_size)

        source_crop = RandomPointConstrainedCrop(
            size=size,
            scale=[self.config.crop_scale_min, self.config.crop_scale_max],
            norm_coord=coords[0]
        )

        target_crop = transforms.RandomResizedCrop(
            size=size,
            scale=[self.config.crop_scale_min, self.config.crop_scale_max]
        )

        return source_crop, target_crop

    def generate(
        self,
        model,
        sample: Sample,
        **kwargs
    ) -> AttackResult:
        """Generate a PhysPatch adversarial example for the given sample."""
        self._initialize_models()

        coords = self._load_coordinates()
        if isinstance(coords, dict):
            # Keyed format: look up by sample.id (stem)
            sample_stem = str(sample.id)
            if sample_stem not in coords:
                raise KeyError(
                    f"No coordinate for sample '{sample_stem}'. "
                    f"Available: {list(coords.keys())[:5]}..."
                )
            xy = coords[sample_stem]
            coord = torch.tensor([xy], device=self.config.device)
        else:
            # Legacy positional format
            sample_idx = kwargs.get("sample_idx")
            if sample_idx is None:
                try:
                    sample_idx = int(sample.id)
                except Exception as e:
                    raise ValueError(f"sample_idx required for non-numeric sample.id={sample.id}") from e
            if sample_idx < 0 or sample_idx >= len(coords):
                raise IndexError(
                    f"Coordinate index out of range: sample_idx={sample_idx}, coords_len={len(coords)}"
                )
            coord = coords[sample_idx:sample_idx+1]

        source_pil = sample.images[0]
        self._source_size = (source_pil.height, source_pil.width)  # (H, W)
        clean_image = self._prepare_image(source_pil)
        target_tensor = self._load_target_image(sample.id, kwargs.get("sample_idx"))
        if target_tensor is None:
            target_tensor = clean_image

        source_crop, target_crop = self._create_crops(coord, self._source_size)

        from vlm_benchmark.attacks.physpatch.core.attacks import pgd, mifgsm

        attack_fn = pgd if self.config.attack_method == "pgd" else mifgsm

        print(f"Running {self.config.attack_method.upper()} attack on sample {sample.id}...")

        adv_image, adv_patch = attack_fn(
            image_tensor=clean_image,
            tgt_tensor=target_tensor,
            ensemble_extractor=self._ensemble_extractor,
            ensemble_loss=self._ensemble_loss,
            source_crop=source_crop,
            target_crop=target_crop,
            center=coord,
            num_iters=self.config.max_iterations,
            epsilon=self.config.epsilon,  # in [0, 255] range; attack converts at end
            alpha=self.config.alpha,
            decay=self.config.decay,
            device=self.config.device
        )

        adv_pil = self._tensor_to_pil(adv_image)

        original_output = ""
        adversarial_output = ""
        success = False

        if model is not None:
            original_output = model.inference([sample.images[0]], sample.question).text
            adversarial_output = model.inference([adv_pil], sample.question).text

            target_keywords = {
                "stop_sign": ["stop", "sign"],
                "speed_limit": ["speed", "limit"],
                "pedestrian_crossing": ["pedestrian", "crossing", "crosswalk"]
            }

            keywords = target_keywords.get(self.config.target_strategy, ["stop"])
            success = any(kw in adversarial_output.lower() for kw in keywords)

        return AttackResult(
            success=success,
            adversarial_sample=adv_pil,
            original_output=original_output,
            adversarial_output=adversarial_output,
            perturbation_norm=self.config.epsilon / 255.0,
            queries=1,
            metadata={
                "attack_method": self.config.attack_method,
                "coord": coord.cpu().tolist(),
                "backbone": self.config.backbone,
                "K": self.config.K,
                "target_strategy": self.config.target_strategy,
            }
        )
