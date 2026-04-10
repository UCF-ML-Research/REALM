"""ADVEDM core components."""

from .advedm_attack import (
    ADVEDMSemanticAdditionPaperExact,
    ADVEDMSemanticRemovalPaperExact,
    adam_attack_advedm_a_paper_exact,
    adam_attack_advedm_r_paper_exact,
)
from .clip_contrastive_encoder import CLIPContrastiveEncoder
from .vision_backends import ClipVisionBackend, TargetVLMVisionBackend
from .gpt_annotator import gpt_annotate_background_region
from .ensemble_encoder import EnsembleEncoder, SurrogateModel, CLIPModelSpec, PAPER_ENSEMBLE
from .ssa_cwa import ssa_cwa_attack, ssa_gradient, dct_2d, idct_2d

__all__ = [
    "ADVEDMSemanticAdditionPaperExact",
    "ADVEDMSemanticRemovalPaperExact",
    "adam_attack_advedm_a_paper_exact",
    "adam_attack_advedm_r_paper_exact",
    "CLIPContrastiveEncoder",
    "ClipVisionBackend",
    "TargetVLMVisionBackend",
    "gpt_annotate_background_region",
    "EnsembleEncoder",
    "SurrogateModel",
    "CLIPModelSpec",
    "PAPER_ENSEMBLE",
    "ssa_cwa_attack",
    "ssa_gradient",
    "dct_2d",
    "idct_2d",
]
