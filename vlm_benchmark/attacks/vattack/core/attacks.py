"""PGD attack extracted from V-Attack.py with explicit parameters."""

import torch
import torchvision.transforms as transforms
from torch import nn
from typing import Optional
from tqdm import tqdm


def _dict_to_list(features_dict):
    return [features_dict[i] for i in range(len(features_dict))]


def pgd_attack(
    image_org: torch.Tensor,
    ensemble_extractor: nn.Module,
    ensemble_loss: nn.Module,
    source_crop: Optional[transforms.RandomResizedCrop],
    source_text: list,
    target_text: list,
    steps: int = 300,
    epsilon: float = 16.0,
    alpha: float = 0.75,
    vattack: bool = True,
    enhance: bool = True,
    both: bool = False,
    vision_attack: bool = False,
    target_text_flag: bool = True,
    device: str = "cuda",
) -> torch.Tensor:
    """Run PGD attack on a single image using CLIP ensemble.

    Args:
        image_org: Input image tensor in [0, 255] range, shape [1, C, H, W].
        ensemble_extractor: EnsembleFeatureExtractor with loaded CLIP models.
        ensemble_loss: EnsembleFeatureLoss with ground truth set.
        source_crop: RandomResizedCrop transform (or Identity).
        source_text: Encoded source text features (list per model).
        target_text: Encoded target text features (list per model).
        steps: Number of PGD iterations.
        epsilon: L_inf perturbation bound (pixel space, [0,255]).
        alpha: Adam learning rate.
        vattack: If True, use V-features; if False, use X-features.
        enhance: If True, use V@V attention enhancement.
        both: If True, use both V and X features.
        vision_attack: Whether to use vision-based attack loss.
        target_text_flag: Whether to use targeted text loss.
        device: Device string.

    Returns:
        Adversarial image tensor in [0, 1] range, shape [1, C, H, W].
    """
    delta = torch.zeros_like(image_org, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=alpha)

    pbar = tqdm(range(steps), desc="V-Attack PGD")

    with torch.no_grad():
        ensemble_loss.set_enhance(enhance)
        ensemble_loss.set_ground_truth(source_crop(image_org).to(device), source_text, vattack)
        ensemble_loss.set_target_text(target_text)
        ensemble_loss.set_mask()
        ensemble_loss.set_mask_index()

    for epoch in pbar:
        adv_image = image_org + delta
        local_cropped = source_crop(adv_image)

        if vattack:
            if not both:
                local_features = ensemble_extractor.vforward(local_cropped, enhance=enhance)
                local_features = _dict_to_list(local_features)
                loss = -ensemble_loss(local_features, Vision_A=vision_attack, Target_A=target_text_flag)
            else:
                local_features, x_features = ensemble_extractor.vforward(local_cropped, enhance=enhance, both=both)
                local_features = _dict_to_list(local_features)
                loss = -ensemble_loss(local_features, x_features, vision_attack, target_text_flag)
        else:
            local_features = ensemble_extractor.xforward(local_cropped)
            loss = -ensemble_loss(local_features, Vision_A=vision_attack, Target_A=target_text_flag)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        delta.data = torch.clamp(
            delta,
            min=-epsilon,
            max=epsilon,
        )

    adv_image = image_org + delta
    adv_image = torch.clamp(adv_image / 255.0, 0.0, 1.0)

    return adv_image
