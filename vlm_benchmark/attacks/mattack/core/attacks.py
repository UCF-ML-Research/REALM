"""M-Attack functions (adapted from original M-Attack)."""

import torch
from torch import nn
from typing import Optional
import torchvision.transforms as transforms
from tqdm import tqdm


def fgsm_attack(
    image_tensor: torch.Tensor,
    tgt_tensor: torch.Tensor,
    ensemble_extractor: nn.Module,
    ensemble_loss: nn.Module,
    source_crop: Optional[transforms.RandomResizedCrop],
    target_crop: Optional[transforms.RandomResizedCrop],
    img_index: int,
    num_iters: int,
    epsilon: float,
    alpha: float,
    device: str,
    use_source_crop: bool = True,
    use_target_crop: bool = True,
) -> torch.Tensor:
    """Run an FGSM attack to generate an adversarial example."""
    delta = torch.zeros_like(image_tensor, requires_grad=True)

    pbar = tqdm(range(num_iters), desc=f"M-Attack FGSM #{img_index}")

    for epoch in pbar:
        with torch.no_grad():
            ensemble_loss.set_ground_truth(target_crop(tgt_tensor))

        adv_image = image_tensor + delta
        adv_features = ensemble_extractor(adv_image)

        global_sim = ensemble_loss(adv_features)

        if use_source_crop:
            local_cropped = source_crop(adv_image)
            local_features = ensemble_extractor(local_cropped)
            local_sim = ensemble_loss(local_features)
            loss = local_sim
        else:
            loss = global_sim

        grad = torch.autograd.grad(loss, delta, create_graph=False)[0]

        delta.data = torch.clamp(
            delta + alpha * torch.sign(grad),
            min=-epsilon,
            max=epsilon,
        )

        pbar.set_postfix({
            "max_delta": f"{torch.max(torch.abs(delta)).item():.3f}",
            "global_sim": f"{global_sim.item():.3f}",
        })

    adv_image = image_tensor + delta
    adv_image = torch.clamp(adv_image / 255.0, 0.0, 1.0)

    return adv_image


def mifgsm_attack(
    image_tensor: torch.Tensor,
    tgt_tensor: torch.Tensor,
    ensemble_extractor: nn.Module,
    ensemble_loss: nn.Module,
    source_crop: Optional[transforms.RandomResizedCrop],
    target_crop: Optional[transforms.RandomResizedCrop],
    img_index: int,
    num_iters: int,
    epsilon: float,
    alpha: float,
    device: str,
    use_source_crop: bool = True,
    use_target_crop: bool = True,
) -> torch.Tensor:
    """Run an MI-FGSM attack to generate an adversarial example."""
    delta = torch.zeros_like(image_tensor, requires_grad=True)
    momentum = torch.zeros_like(image_tensor, requires_grad=False)

    pbar = tqdm(range(num_iters), desc=f"M-Attack MI-FGSM #{img_index}")

    for epoch in pbar:
        with torch.no_grad():
            ensemble_loss.set_ground_truth(target_crop(tgt_tensor))

        adv_image = image_tensor + delta
        adv_features = ensemble_extractor(adv_image)

        global_sim = ensemble_loss(adv_features)

        if use_source_crop:
            local_cropped = source_crop(adv_image)
            local_features = ensemble_extractor(local_cropped)
            local_sim = ensemble_loss(local_features)
            loss = local_sim
        else:
            loss = global_sim

        grad = torch.autograd.grad(loss, delta, create_graph=False)[0]

        momentum = momentum * 0.9 + grad
        delta.data = torch.clamp(
            delta + alpha * torch.sign(momentum),
            min=-epsilon,
            max=epsilon,
        )

        pbar.set_postfix({
            "max_delta": f"{torch.max(torch.abs(delta)).item():.3f}",
            "global_sim": f"{global_sim.item():.3f}",
        })

    adv_image = image_tensor + delta
    adv_image = torch.clamp(adv_image / 255.0, 0.0, 1.0)

    return adv_image


def pgd_attack(
    image_tensor: torch.Tensor,
    tgt_tensor: torch.Tensor,
    ensemble_extractor: nn.Module,
    ensemble_loss: nn.Module,
    source_crop: Optional[transforms.RandomResizedCrop],
    target_crop: Optional[transforms.RandomResizedCrop],
    img_index: int,
    num_iters: int,
    epsilon: float,
    alpha: float,
    device: str,
    use_source_crop: bool = True,
    use_target_crop: bool = True,
) -> torch.Tensor:
    """Run a PGD attack to generate an adversarial example."""
    delta = torch.zeros_like(image_tensor, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=alpha)

    pbar = tqdm(range(num_iters), desc=f"M-Attack PGD #{img_index}")

    for epoch in pbar:
        with torch.no_grad():
            ensemble_loss.set_ground_truth(target_crop(tgt_tensor))

        adv_image = image_tensor + delta
        adv_features = ensemble_extractor(adv_image)

        global_sim = ensemble_loss(adv_features)

        if use_source_crop:
            local_cropped = source_crop(adv_image)
            local_features = ensemble_extractor(local_cropped)
            local_sim = ensemble_loss(local_features)
            loss = -local_sim  # negated to maximize similarity
        else:
            loss = -global_sim

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        delta.data = torch.clamp(
            delta,
            min=-epsilon,
            max=epsilon,
        )

        pbar.set_postfix({
            "max_delta": f"{torch.max(torch.abs(delta)).item():.3f}",
            "global_sim": f"{global_sim.item():.3f}",
        })

    adv_image = image_tensor + delta
    adv_image = torch.clamp(adv_image / 255.0, 0.0, 1.0)

    return adv_image
