"""FOA surrogate models (imported from local copy)."""

from ..surrogates.FeatureExtractors.ClipB16 import ClipB16FeatureExtractor
from ..surrogates.FeatureExtractors.ClipB32 import ClipB32FeatureExtractor
from ..surrogates.FeatureExtractors.ClipL336 import ClipL336FeatureExtractor
from ..surrogates.FeatureExtractors.ClipLaion import ClipLaionFeatureExtractor
from ..surrogates.FeatureExtractors.Base import (
    EnsembleFeatureExtractor_ot,
    EnsembleFeatureLoss_OT_foa_attack
)

__all__ = [
    "ClipB16FeatureExtractor",
    "ClipB32FeatureExtractor",
    "ClipL336FeatureExtractor",
    "ClipLaionFeatureExtractor",
    "EnsembleFeatureExtractor_ot",
    "EnsembleFeatureLoss_OT_foa_attack",
]
