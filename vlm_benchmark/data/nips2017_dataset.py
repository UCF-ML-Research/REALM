"""NIPS 2017 adversarial dataset of ImageNet source/target pairs."""

import json
from pathlib import Path
from typing import Optional

from PIL import Image

from .base_dataset import Sample, BaseDataset

DEFAULT_QUESTION = "What is the main object in this image?"


class Nips2017Dataset(BaseDataset):
    """Loads the bundled NIPS-2017 source images, target images, and labels."""

    def __init__(
        self,
        data_root: str = "dataset/nips2017",
        question: str = DEFAULT_QUESTION,
        max_samples: Optional[int] = None,
    ):
        self.question = question
        self._labels = {}
        super().__init__(data_root, max_samples)

    def load_data(self) -> None:
        root = Path(self.data_root)
        labels_path = root / "labels.json"
        self._labels = json.loads(labels_path.read_text()) if labels_path.exists() else {}
        source_dir = root / "source"
        self.entries = sorted(
            list(source_dir.glob("*.png"))
            + list(source_dir.glob("*.jpg"))
            + list(source_dir.glob("*.jpeg"))
        )

    def get_sample(self, idx: int) -> Sample:
        src = self.entries[idx]
        sid = src.stem
        lab = self._labels.get(sid, {})
        source_text = lab.get("source", "")
        target_text = lab.get("target", "")
        target_img = Path(self.data_root) / "target" / f"{sid}.jpg"
        return Sample(
            id=sid,
            images=[Image.open(src).convert("RGB")],
            question=self.question,
            ground_truth=source_text,
            task_type="attack",
            metadata={
                "source_text": source_text,
                "target_text": target_text,
                "attack_source_text": source_text,
                "attack_target_text": target_text,
                "source_label": source_text,
                "target_label": target_text,
                "image_file": str(src),
                "target_image": str(target_img) if target_img.exists() else None,
            },
        )
