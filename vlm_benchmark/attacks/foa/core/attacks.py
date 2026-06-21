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
    """FGSM attack generating adversarial examples (from original FOA-Attack)."""
    delta = torch.zeros_like(image_tensor, requires_grad=True)

    pbar = tqdm(range(num_iters), desc=f"Attack progress")

    for epoch in pbar:

        with torch.no_grad():
            ensemble_loss.set_ground_truth(target_crop(tgt_tensor))

        adv_image = image_tensor + delta

        adv_out = ensemble_extractor(adv_image)
        if isinstance(adv_out, tuple):
            adv_features, adv_features_local = adv_out
        else:
            adv_features, adv_features_local = adv_out, None

        metrics = {
            "max_delta": torch.max(torch.abs(delta)).item(),
            "mean_delta": torch.mean(torch.abs(delta)).item(),
        }

        global_sim = ensemble_loss(adv_features, adv_features_local)
        metrics["global_similarity"] = global_sim.item()

        if use_source_crop:
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
            loss = global_sim

        pbar_metrics = {
            k: f"{v:.5f}" if "sim" in k else f"{v:.3f}" for k, v in metrics.items()
        }
        pbar.set_postfix(pbar_metrics)

        grad = torch.autograd.grad(loss, delta, create_graph=False)[0]

        delta.data = torch.clamp(
            delta + alpha * torch.sign(grad),
            min=-epsilon,
            max=epsilon,
        )

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
    """MI-FGSM attack generating adversarial examples (from original FOA-Attack)."""
    delta = torch.zeros_like(image_tensor, requires_grad=True)
    momentum = torch.zeros_like(image_tensor, requires_grad=False)

    pbar = tqdm(range(num_iters), desc=f"Attack progress")

    for epoch in pbar:

        with torch.no_grad():
            ensemble_loss.set_ground_truth(target_crop(tgt_tensor))

        adv_image = image_tensor + delta
        adv_out = ensemble_extractor(adv_image)
        if isinstance(adv_out, tuple):
            adv_features, adv_features_local = adv_out
        else:
            adv_features, adv_features_local = adv_out, None

        metrics = {
            "max_delta": torch.max(torch.abs(delta)).item(),
            "mean_delta": torch.mean(torch.abs(delta)).item(),
        }

        global_sim = ensemble_loss(adv_features, adv_features_local)
        metrics["global_similarity"] = global_sim.item()

        if use_source_crop:
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
            loss = global_sim

        pbar_metrics = {
            k: f"{v:.5f}" if "sim" in k else f"{v:.3f}" for k, v in metrics.items()
        }
        pbar.set_postfix(pbar_metrics)

        grad = torch.autograd.grad(loss, delta, create_graph=False)[0]

        momentum = momentum * 0.9 + grad
        delta.data = torch.clamp(
            delta + alpha * torch.sign(momentum),
            min=-epsilon,
            max=epsilon,
        )

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
    """PGD attack generating adversarial examples (from original FOA-Attack)."""
    delta = torch.zeros_like(image_tensor, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=alpha)

    pbar = tqdm(range(num_iters), desc=f"Attack progress")

    for epoch in pbar:

        with torch.no_grad():
            ensemble_loss.set_ground_truth(target_crop(tgt_tensor))

        adv_image = image_tensor + delta
        adv_out = ensemble_extractor(adv_image)
        if isinstance(adv_out, tuple):
            adv_features, adv_features_local = adv_out
        else:
            adv_features, adv_features_local = adv_out, None

        metrics = {
            "max_delta": torch.max(torch.abs(delta)).item(),
            "mean_delta": torch.mean(torch.abs(delta)).item(),
        }

        global_sim = ensemble_loss(adv_features, adv_features_local)
        metrics["global_similarity"] = global_sim.item()

        if use_source_crop:
            local_cropped = source_crop(adv_image)
            local_out = ensemble_extractor(local_cropped)
            if isinstance(local_out, tuple):
                local_features, local_features_local = local_out
            else:
                local_features, local_features_local = local_out, None
            local_sim = ensemble_loss(local_features, local_features_local)
            loss = -local_sim  # negate to maximize similarity
            metrics["local_similarity"] = local_sim.item()
        else:
            loss = -global_sim

        pbar_metrics = {
            k: f"{v:.5f}" if "sim" in k else f"{v:.3f}" for k, v in metrics.items()
        }
        pbar.set_postfix(pbar_metrics)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        delta.data = torch.clamp(
            delta,
            min=-epsilon,
            max=epsilon,
        )

    adv_image = image_tensor + delta
    adv_image = torch.clamp(adv_image / 255.0, 0.0, 1.0)

    return adv_image


__all__ = ["fgsm_attack", "mifgsm_attack", "pgd_attack"]
