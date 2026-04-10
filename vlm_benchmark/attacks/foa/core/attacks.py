"""FOA attack functions (copied EXACTLY from original FOA-Attack)."""

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
    """
    FGSM attack (copied EXACTLY from original FOA-Attack line 426).

    Perform FGSM attack on the image to generate adversarial examples.
    """
    # Initialize perturbation
    delta = torch.zeros_like(image_tensor, requires_grad=True)

    # Progress bar for optimization
    pbar = tqdm(range(num_iters), desc=f"Attack progress")

    # Main optimization loop
    for epoch in pbar:

        with torch.no_grad():
            ensemble_loss.set_ground_truth(target_crop(tgt_tensor))

        # Forward pass
        adv_image = image_tensor + delta

        adv_out = ensemble_extractor(adv_image)
        if isinstance(adv_out, tuple):
            adv_features, adv_features_local = adv_out
        else:
            adv_features, adv_features_local = adv_out, None

        # Calculate metrics
        metrics = {
            "max_delta": torch.max(torch.abs(delta)).item(),
            "mean_delta": torch.mean(torch.abs(delta)).item(),
        }

        # Calculate loss based on configuration
        global_sim = ensemble_loss(adv_features, adv_features_local)
        metrics["global_similarity"] = global_sim.item()

        if use_source_crop:
            # If using source crop, calculate additional local similarity
            local_cropped = source_crop(adv_image)
            local_out = ensemble_extractor(local_cropped)
            if isinstance(local_out, tuple):
                local_features, local_features_local = local_out
            else:
                local_features, local_features_local = local_out, None
            local_sim = ensemble_loss(local_features, local_features_local)
            loss = local_sim
            metrics["local_similarity"] = local_sim.item()
        else:
            # Otherwise use global similarity as loss
            loss = global_sim

        # Update progress bar
        pbar_metrics = {
            k: f"{v:.5f}" if "sim" in k else f"{v:.3f}" for k, v in metrics.items()
        }
        pbar.set_postfix(pbar_metrics)

        grad = torch.autograd.grad(loss, delta, create_graph=False)[0]

        # Update delta using FGSM
        delta.data = torch.clamp(
            delta + alpha * torch.sign(grad),
            min=-epsilon,
            max=epsilon,
        )

    # Create final adversarial image
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
    """
    MI-FGSM attack (copied EXACTLY from original FOA-Attack line 520).

    Perform MI-FGSM attack on the image to generate adversarial examples.
    """
    # Initialize perturbation and momentum
    delta = torch.zeros_like(image_tensor, requires_grad=True)
    momentum = torch.zeros_like(image_tensor, requires_grad=False)

    # Progress bar for optimization
    pbar = tqdm(range(num_iters), desc=f"Attack progress")

    # Main optimization loop
    for epoch in pbar:

        with torch.no_grad():
            ensemble_loss.set_ground_truth(target_crop(tgt_tensor))

        # Forward pass
        adv_image = image_tensor + delta
        adv_out = ensemble_extractor(adv_image)
        if isinstance(adv_out, tuple):
            adv_features, adv_features_local = adv_out
        else:
            adv_features, adv_features_local = adv_out, None

        # Calculate metrics
        metrics = {
            "max_delta": torch.max(torch.abs(delta)).item(),
            "mean_delta": torch.mean(torch.abs(delta)).item(),
        }

        # Calculate loss based on configuration
        global_sim = ensemble_loss(adv_features, adv_features_local)
        metrics["global_similarity"] = global_sim.item()

        if use_source_crop:
            # If using source crop, calculate additional local similarity
            local_cropped = source_crop(adv_image)
            local_out = ensemble_extractor(local_cropped)
            if isinstance(local_out, tuple):
                local_features, local_features_local = local_out
            else:
                local_features, local_features_local = local_out, None
            local_sim = ensemble_loss(local_features, local_features_local)
            loss = local_sim
            metrics["local_similarity"] = local_sim.item()
        else:
            # Otherwise use global similarity as loss
            loss = global_sim

        # Update progress bar
        pbar_metrics = {
            k: f"{v:.5f}" if "sim" in k else f"{v:.3f}" for k, v in metrics.items()
        }
        pbar.set_postfix(pbar_metrics)

        grad = torch.autograd.grad(loss, delta, create_graph=False)[0]

        # MI-FGSM update
        momentum = momentum * 0.9 + grad
        delta.data = torch.clamp(
            delta + alpha * torch.sign(momentum),
            min=-epsilon,
            max=epsilon,
        )

    # Create final adversarial image
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
    """
    PGD attack (copied EXACTLY from original FOA-Attack line 610).

    Perform PGD attack on the image to generate adversarial examples.
    """
    # Initialize perturbation and momentum
    delta = torch.zeros_like(image_tensor, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=alpha)

    # Progress bar for optimization
    pbar = tqdm(range(num_iters), desc=f"Attack progress")

    # Main optimization loop
    for epoch in pbar:

        with torch.no_grad():
            ensemble_loss.set_ground_truth(target_crop(tgt_tensor))

        # Forward pass
        adv_image = image_tensor + delta
        adv_out = ensemble_extractor(adv_image)
        if isinstance(adv_out, tuple):
            adv_features, adv_features_local = adv_out
        else:
            adv_features, adv_features_local = adv_out, None

        # Calculate metrics
        metrics = {
            "max_delta": torch.max(torch.abs(delta)).item(),
            "mean_delta": torch.mean(torch.abs(delta)).item(),
        }

        # Calculate loss based on configuration
        global_sim = ensemble_loss(adv_features, adv_features_local)
        metrics["global_similarity"] = global_sim.item()

        if use_source_crop:
            # If using source crop, calculate additional local similarity
            local_cropped = source_crop(adv_image)
            local_out = ensemble_extractor(local_cropped)
            if isinstance(local_out, tuple):
                local_features, local_features_local = local_out
            else:
                local_features, local_features_local = local_out, None
            local_sim = ensemble_loss(local_features, local_features_local)
            loss = -local_sim  # since we want to maximize the loss
            metrics["local_similarity"] = local_sim.item()
        else:
            # Otherwise use global similarity as loss
            loss = -global_sim

        # Update progress bar
        pbar_metrics = {
            k: f"{v:.5f}" if "sim" in k else f"{v:.3f}" for k, v in metrics.items()
        }
        pbar.set_postfix(pbar_metrics)

        optimizer.zero_grad()
        loss.backward()

        # PGD update
        optimizer.step()
        delta.data = torch.clamp(
            delta,
            min=-epsilon,
            max=epsilon,
        )

    # Create final adversarial image
    adv_image = image_tensor + delta
    adv_image = torch.clamp(adv_image / 255.0, 0.0, 1.0)

    return adv_image


__all__ = ["fgsm_attack", "mifgsm_attack", "pgd_attack"]
