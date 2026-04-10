"""PAD Defense Wrapper."""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import cv2
from PIL import Image

from ..base_defense import BaseDefense, DefenseConfig, DefenseResult


@dataclass
class PADDefenseConfig(DefenseConfig):
    """PAD-specific configuration."""
    iou_threshold: float = 0.5
    ratio_mi: float = 0.5
    kernel_param: int = 80
    thresh_param: int = 80
    sam_model_type: str = "vit_l"
    sam_checkpoint: Optional[str] = None
    device: str = "cuda:0"


class PADDefense(BaseDefense):
    """PAD defense wrapper."""

    def __init__(self, config: PADDefenseConfig):
        super().__init__(config)
        self.config: PADDefenseConfig = config
        self._sam_mask_generator = None
        self._pad_modules = None

    def _initialize_models(self):
        """Lazy load SAM and PAD helper modules."""
        if self._sam_mask_generator is not None:
            return

        # Import runtime PAD helpers (ported from legacy PAD code)
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        from .core import fuse_filter as fuse_filter_module

        self._pad_modules = {
            'fuse_filter_module': fuse_filter_module,
            'fuse_heatmap': fuse_filter_module.fuse_heatmap,
            'heatmap_filter': fuse_filter_module.heatmap_filter,
        }

        # Load SAM
        if self.config.sam_checkpoint is None:
            from .config import get_sam_checkpoint_path
            self.config.sam_checkpoint = get_sam_checkpoint_path()

        print(f"Loading SAM model from {self.config.sam_checkpoint}")
        sam = sam_model_registry[self.config.sam_model_type](
            checkpoint=self.config.sam_checkpoint
        )
        sam.to(device=self.config.device)
        self._sam_mask_generator = SamAutomaticMaskGenerator(sam)
        print("SAM model loaded successfully")

    def clean(self, image_path: str, **_kwargs) -> DefenseResult:
        """
        Clean adversarial image using PAD pipeline.

        Pipeline (ported from legacy/run-PAD.py lines 56-103):
        1. Generate fused heatmap (MI + CD)
        2. Filter heatmap with morphological operations
        3. Generate SAM segmentation masks
        4. Remove masks with high IoU to heatmap
        5. Return cleaned image

        Args:
            image_path: Path to adversarial image
            **_kwargs: Additional parameters (unused)

        Returns:
            DefenseResult with cleaned image and metadata
        """
        # 1. Initialize (lazy)
        self._initialize_models()

        # 2. Load image
        ori_img = Image.open(image_path).convert('RGB')
        ori_width, ori_height = ori_img.size

        # 3. Generate fused heatmap (MI + CD) - ported from legacy helper modules
        # Strict alignment: propagate config into helper module globals
        fuse_filter_module = self._pad_modules['fuse_filter_module']
        fuse_filter_module.ratio_mi = self.config.ratio_mi
        fuse_filter_module.kernel_param = self.config.kernel_param
        fuse_filter_module.thresh_param = self.config.thresh_param

        fuse_heatmap = self._pad_modules['fuse_heatmap']
        _mi_img, _cd_img, fuse_img = fuse_heatmap(image_path, ori_height, ori_width)

        # 4. Filter heatmap
        heatmap_filter = self._pad_modules['heatmap_filter']
        threshold = np.percentile(fuse_img, self.config.thresh_param)
        _h_t, _h_t_o, _h_t_o_c, h_t_o_c_o = heatmap_filter(
            fuse_img, threshold, ori_height, ori_width
        )

        # 5. Binary mask
        gray = np.where(h_t_o_c_o > 0, 1, 0)

        # 6. SAM segmentation
        rgb_color = cv2.imread(image_path)
        image = cv2.cvtColor(rgb_color, cv2.COLOR_BGR2RGB)
        masks = self._sam_mask_generator.generate(image.astype(np.uint8))

        print(f"Generated {len(masks)} SAM masks")

        # 7. Remove masks with high IoU - SAME AS LEGACY run-PAD.py lines 75-101
        result_mask = np.zeros(image.shape[:2])
        regions_removed = 0

        for k in range(len(masks)):
            mask_k = masks[k].get('segmentation')

            # IoU with heatmap
            intersection = mask_k & gray
            mask_area = np.sum(mask_k)
            if mask_area == 0:
                continue
            iou = np.sum(intersection) / mask_area

            # IoU with previous masks
            intersection_prev = mask_k & result_mask.astype(np.uint8)
            iou_prev = np.sum(intersection_prev) / mask_area

            # Remove if high IoU and low overlap with previous removals
            if iou > self.config.iou_threshold and iou_prev < 0.1:
                mask_k_3d = np.expand_dims(mask_k, axis=2)
                mask_k_3d = np.tile(mask_k_3d, 3)
                rgb_color = rgb_color * (~mask_k_3d)
                result_mask = result_mask.astype(np.uint8) | mask_k
                regions_removed += 1

        # 8. Convert to PIL
        cleaned_pil = Image.fromarray(cv2.cvtColor(rgb_color, cv2.COLOR_BGR2RGB))

        # 9. Return result
        detection_confidence = float(np.sum(gray > 0)) / (ori_height * ori_width)

        return DefenseResult(
            cleaned_sample=cleaned_pil,
            original_image_path=image_path,
            detection_confidence=detection_confidence,
            regions_removed=regions_removed,
            metadata={
                "iou_threshold": self.config.iou_threshold,
                "ratio_mi": self.config.ratio_mi,
                "total_sam_masks": len(masks),
                "image_size": f"{ori_width}x{ori_height}",
            }
        )

    def requires_model(self) -> bool:
        """PAD uses SAM, not a VLM."""
        return False
