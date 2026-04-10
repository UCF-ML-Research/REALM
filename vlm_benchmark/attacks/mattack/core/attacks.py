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
    """
    Perform FGSM attack on the image to generate adversarial examples.

    Adapted from M-Attack generate_adversarial_samples.py line 260.

    Args:
        image_tensor: Original source image tensor [0, 255]
        tgt_tensor: Target image tensor to match features with [0, 255]
        ensemble_extractor: Ensemble feature extractor model
        ensemble_loss: Ensemble loss function
        source_crop: Optional transform for cropping source images
        target_crop: Optional transform for cropping target images
        img_index: Index of the image (for logging)
        num_iters: Number of optimization steps
        epsilon: Maximum perturbation magnitude
        alpha: Step size
        device: Device to run on
        use_source_crop: Whether to use source cropping
        use_target_crop: Whether to use target cropping

    Returns:
        Generated adversarial image tensor [0, 1]
    """
    # Initialize perturbation
    delta = torch.zeros_like(image_tensor, requires_grad=True)

    # Progress bar for optimization
    pbar = tqdm(range(num_iters), desc=f"M-Attack FGSM #{img_index}")

    # Main optimization loop
    for epoch in pbar:
        with torch.no_grad():
            ensemble_loss.set_ground_truth(target_crop(tgt_tensor))

        # Forward pass
        adv_image = image_tensor + delta
        adv_features = ensemble_extractor(adv_image)

        # Calculate loss based on configuration
        global_sim = ensemble_loss(adv_features)

        if use_source_crop:
            # If using source crop, calculate additional local similarity
            local_cropped = source_crop(adv_image)
            local_features = ensemble_extractor(local_cropped)
            local_sim = ensemble_loss(local_features)
            loss = local_sim
        else:
            # Otherwise use global similarity as loss
            loss = global_sim

        grad = torch.autograd.grad(loss, delta, create_graph=False)[0]

        # Update delta using FGSM
        delta.data = torch.clamp(
            delta + alpha * torch.sign(grad),
            min=-epsilon,
            max=epsilon,
        )

        # Update progress bar
        pbar.set_postfix({
            "max_delta": f"{torch.max(torch.abs(delta)).item():.3f}",
            "global_sim": f"{global_sim.item():.3f}",
        })

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
    Perform MI-FGSM attack on the image to generate adversarial examples.

    Adapted from M-Attack generate_adversarial_samples.py line 349.

    Args:
        image_tensor: Original source image tensor [0, 255]
        tgt_tensor: Target image tensor to match features with [0, 255]
        ensemble_extractor: Ensemble feature extractor model
        ensemble_loss: Ensemble loss function
        source_crop: Optional transform for cropping source images
        target_crop: Optional transform for cropping target images
        img_index: Index of the image (for logging)
        num_iters: Number of optimization steps
        epsilon: Maximum perturbation magnitude
        alpha: Step size
        device: Device to run on
        use_source_crop: Whether to use source cropping
        use_target_crop: Whether to use target cropping

    Returns:
        Generated adversarial image tensor [0, 1]
    """
    # Initialize perturbation and momentum
    delta = torch.zeros_like(image_tensor, requires_grad=True)
    momentum = torch.zeros_like(image_tensor, requires_grad=False)

    # Progress bar for optimization
    pbar = tqdm(range(num_iters), desc=f"M-Attack MI-FGSM #{img_index}")

    # Main optimization loop
    for epoch in pbar:
        with torch.no_grad():
            ensemble_loss.set_ground_truth(target_crop(tgt_tensor))

        # Forward pass
        adv_image = image_tensor + delta
        adv_features = ensemble_extractor(adv_image)

        # Calculate loss based on configuration
        global_sim = ensemble_loss(adv_features)

        if use_source_crop:
            # If using source crop, calculate additional local similarity
            local_cropped = source_crop(adv_image)
            local_features = ensemble_extractor(local_cropped)
            local_sim = ensemble_loss(local_features)
            loss = local_sim
        else:
            # Otherwise use global similarity as loss
            loss = global_sim

        grad = torch.autograd.grad(loss, delta, create_graph=False)[0]

        # MI-FGSM update
        momentum = momentum * 0.9 + grad
        delta.data = torch.clamp(
            delta + alpha * torch.sign(momentum),
            min=-epsilon,
            max=epsilon,
        )

        # Update progress bar
        pbar.set_postfix({
            "max_delta": f"{torch.max(torch.abs(delta)).item():.3f}",
            "global_sim": f"{global_sim.item():.3f}",
        })

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
    Perform PGD attack on the image to generate adversarial examples.

    Adapted from M-Attack generate_adversarial_samples.py line 439.

    Args:
        image_tensor: Original source image tensor [0, 255]
        tgt_tensor: Target image tensor to match features with [0, 255]
        ensemble_extractor: Ensemble feature extractor model
        ensemble_loss: Ensemble loss function
        source_crop: Optional transform for cropping source images
        target_crop: Optional transform for cropping target images
        img_index: Index of the image (for logging)
        num_iters: Number of optimization steps
        epsilon: Maximum perturbation magnitude
        alpha: Step size (learning rate)
        device: Device to run on
        use_source_crop: Whether to use source cropping
        use_target_crop: Whether to use target cropping

    Returns:
        Generated adversarial image tensor [0, 1]
    """
    # Initialize perturbation and optimizer
    delta = torch.zeros_like(image_tensor, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=alpha)

    # Progress bar for optimization
    pbar = tqdm(range(num_iters), desc=f"M-Attack PGD #{img_index}")

    # Main optimization loop
    for epoch in pbar:
        with torch.no_grad():
            ensemble_loss.set_ground_truth(target_crop(tgt_tensor))

        # Forward pass
        adv_image = image_tensor + delta
        adv_features = ensemble_extractor(adv_image)

        # Calculate loss based on configuration
        global_sim = ensemble_loss(adv_features)

        if use_source_crop:
            # If using source crop, calculate additional local similarity
            local_cropped = source_crop(adv_image)
            local_features = ensemble_extractor(local_cropped)
            local_sim = ensemble_loss(local_features)
            loss = -local_sim  # Negative because we want to maximize similarity
        else:
            # Otherwise use global similarity as loss
            loss = -global_sim

        optimizer.zero_grad()
        loss.backward()

        # PGD update
        optimizer.step()
        delta.data = torch.clamp(
            delta,
            min=-epsilon,
            max=epsilon,
        )

        # Update progress bar
        pbar.set_postfix({
            "max_delta": f"{torch.max(torch.abs(delta)).item():.3f}",
            "global_sim": f"{global_sim.item():.3f}",
        })

    # Create final adversarial image
    adv_image = image_tensor + delta
    adv_image = torch.clamp(adv_image / 255.0, 0.0, 1.0)

    return adv_image
