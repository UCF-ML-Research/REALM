"""Sample dataclass and the abstract BaseDataset for the VLM benchmark."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Iterator, Optional
from PIL import Image


@dataclass
class Sample:
    """Unified sample format for the VLM benchmark."""
    id: str
    images: List[Image.Image]
    question: str
    ground_truth: str
    task_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseDataset(ABC):
    """Abstract base class for benchmark datasets."""

    def __init__(self, data_root: str, max_samples: Optional[int] = None):
        self.data_root = data_root
        self.max_samples = max_samples
        self.entries: List[Any] = []
        self.load_data()
        if self.max_samples is not None:
            self.entries = self.entries[: self.max_samples]

    @abstractmethod
    def load_data(self) -> None:
        """Populate ``self.entries`` with one record per sample."""

    @abstractmethod
    def get_sample(self, idx: int) -> Sample:
        """Build the :class:`Sample` for ``idx`` (images loaded here)."""

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Sample:
        return self.get_sample(idx)

    def __iter__(self) -> Iterator[Sample]:
        for i in range(len(self)):
            yield self.get_sample(i)
