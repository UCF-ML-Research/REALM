"""Dataset loader for Chain of Attack (clean/target images and captions)."""

from pathlib import Path
from typing import Dict, List
from PIL import Image


class CoADataset:
    """Dataset loader for Chain of Attack."""

    def __init__(
        self,
        clean_images_dir: str,
        clean_captions_path: str,
        target_images_dir: str,
        target_captions_path: str,
    ):
        """Initialize CoA dataset from clean/target image dirs and caption files."""
        self.clean_images_dir = Path(clean_images_dir)
        self.target_images_dir = Path(target_images_dir)

        self.clean_captions = self._load_captions(clean_captions_path)
        self.target_captions = self._load_captions(target_captions_path)

        self.clean_image_paths = sorted(
            list(self.clean_images_dir.glob("*.jpg"))
            + list(self.clean_images_dir.glob("*.png"))
            + list(self.clean_images_dir.glob("*.jpeg"))
        )
        self.target_image_paths = sorted(
            list(self.target_images_dir.glob("*.jpg"))
            + list(self.target_images_dir.glob("*.png"))
            + list(self.target_images_dir.glob("*.jpeg"))
        )

        self._validate_dataset()

    def _load_captions(self, captions_path: str) -> List[str]:
        """Load captions from text file (one per line). Returns empty list if file doesn't exist."""
        if captions_path is None:
            return []
        captions_path = Path(captions_path)
        if not captions_path.exists():
            print(f"  Warning: Captions file not found: {captions_path}")
            print(f"  Captions will be auto-generated during attack")
            return []

        with open(captions_path, "r") as f:
            captions = [line.strip() for line in f if line.strip()]
        return captions

    def _validate_dataset(self):
        """Validate that all components have matching sizes."""
        n_clean_imgs = len(self.clean_image_paths)
        n_clean_caps = len(self.clean_captions)
        n_target_imgs = len(self.target_image_paths)
        n_target_caps = len(self.target_captions)

        if n_clean_imgs == 0:
            raise ValueError(f"No clean images found in {self.clean_images_dir}")

        if n_target_imgs == 0:
            raise ValueError(f"No target images found in {self.target_images_dir}")

        # Missing/partial clean captions are allowed (auto-generated on demand)
        if n_clean_caps > n_clean_imgs:
            raise ValueError(
                f"Too many captions: {n_clean_caps} captions but only {n_clean_imgs} images"
            )

        if n_target_caps > 0 and n_target_imgs != n_target_caps:
            raise ValueError(
                f"Mismatch: {n_target_imgs} target images but {n_target_caps} target captions"
            )

        if n_clean_imgs != n_target_imgs:
            raise ValueError(
                f"Mismatch: {n_clean_imgs} clean images but {n_target_imgs} target images"
            )

        def _cap_status(n_caps, n_imgs):
            if n_caps == 0:
                return "auto-generate on demand"
            elif n_caps == n_imgs:
                return "all pre-loaded"
            return f"{n_caps} pre-loaded, {n_imgs - n_caps} auto-generate"

        clean_status = _cap_status(n_clean_caps, n_clean_imgs)
        target_status = _cap_status(n_target_caps, n_target_imgs)
        print(f"✓ CoA dataset validated: {n_clean_imgs} samples")
        print(f"  Clean captions:  {clean_status}")
        print(f"  Target captions: {target_status}")

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.clean_image_paths)

    def __getitem__(self, idx: int) -> Dict:
        """Get the sample dict (images, captions, paths) at the given index."""
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")

        clean_img_path = self.clean_image_paths[idx]
        target_img_path = self.target_image_paths[idx]

        clean_caption = self.clean_captions[idx] if idx < len(self.clean_captions) else ""

        target_caption = self.target_captions[idx] if idx < len(self.target_captions) else ""

        return {
            "clean_image": Image.open(clean_img_path).convert("RGB"),
            "clean_caption": clean_caption,
            "target_image": Image.open(target_img_path).convert("RGB"),
            "target_caption": target_caption,
            "target_image_path": str(target_img_path),
            "image_path": str(clean_img_path),
            "image_name": clean_img_path.name,
        }
