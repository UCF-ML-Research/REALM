"""Physical adversarial patch attack algorithms (PGD and MI-FGSM)."""

import os
import random
import numpy as np
import torch

from .utils import apply_patch
from .mask_generator import DynamicPatchGenerator
from .transforms import My_T


def set_environment(seed=42):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pgd(
    image_tensor, tgt_tensor, ensemble_extractor, ensemble_loss,
    source_crop, target_crop, center=None,
    num_iters=10, epsilon=8, alpha=1.0, decay=1.0, device='cuda'
):
    """
    Projected Gradient Descent (PGD) attack for physical adversarial patches.

    Args:
        image_tensor: Clean image tensor (B, C, H, W)
        tgt_tensor: Target image tensor (B, C, H, W)
        ensemble_extractor: Ensemble of feature extractors
        ensemble_loss: Ensemble loss function
        source_crop: Crop function for source (adversarial) image
        target_crop: Crop function for target image
        center: Normalized patch center coordinates (B, 2) in [-1, 1]
        num_iters: Number of optimization iterations
        epsilon: L-infinity perturbation budget (in [0, 255] scale)
        alpha: Step size for gradient updates
        decay: Momentum decay (unused in PGD)
        device: Device to run on

    Returns:
        final_image: Adversarial image with patch applied
        final_patch: Final patch tensor
    """
    set_environment(42)
    init_patch = image_tensor.clone().detach().to(device)
    print("patch:", torch.min(init_patch), torch.max(init_patch), init_patch.shape)
    print("ori:", torch.min(image_tensor), torch.max(image_tensor), image_tensor.shape)
    print("tgt:", torch.min(tgt_tensor), torch.max(tgt_tensor), tgt_tensor.shape)

    delta = torch.zeros_like(init_patch, requires_grad=True).to(device)

    _, _, H, W = image_tensor.shape
    patch_generator = DynamicPatchGenerator(image_size=(W, H), device=device).to(device)
    threshold_list = [round(min(0.6 + i * 0.02, 0.95), 6) for i in range(num_iters)]
    mask_expanded = torch.ones([1, 1, H, W]).to(device)
    trans = My_T(num_copies=1)
    # Scale threshold proportionally to image area (120*120 is for 900x1600)
    th = int(H * W * (120 * 120) / (900 * 1600))
    for iter in range(num_iters):
        with torch.no_grad():
            tgt = target_crop(tgt_tensor)
            ensemble_loss.set_ground_truth(tgt)
            ensemble_loss.set_full_feature(tgt_tensor)


        adv_patch = init_patch + delta
        if center is not None:

            if mask_expanded.sum()  > th:
                mask_expanded = patch_generator(center, threshold_list[iter])
            adv_image = image_tensor * (1 - mask_expanded) + adv_patch * mask_expanded
        else:
            adv_image = apply_patch(image=image_tensor, patch=adv_patch)

        local = torch.cat([trans.transform(source_crop(adv_image)) for _ in range(1)])
        features, features_local = ensemble_extractor(local)
        full = ensemble_extractor.encode_img(adv_image)
        loss_a = ensemble_loss(features, features_local, full)
        loss = loss_a


        if mask_expanded.sum()  > th:
            grads = torch.autograd.grad(loss, [delta, mask_expanded], create_graph=False)
            grad = grads[0]
            grad_mask = grads[1]
            print(iter, ":", "loss:", loss.detach().cpu().numpy(), "grad:", grad.mean().detach().cpu().numpy(), "mask_grads:", grad_mask.mean().detach().cpu().numpy())
        else:
            grads = torch.autograd.grad(loss, [delta], create_graph=False)
            grad = grads[0]
            print(iter, ":", "loss:", loss.detach().cpu().numpy(), "grad:", grad.mean().detach().cpu().numpy())

        delta.data = delta + alpha * torch.sign(grad)
        delta.data = torch.clamp(delta.data, min=-epsilon, max=epsilon)
        if mask_expanded.sum()  > th:
            with torch.no_grad():
                patch_generator.update_potential_field(grad_mask, iter)

    final_patch = init_patch + delta
    final_image = image_tensor * (1 - mask_expanded) + final_patch * mask_expanded
    final_image = torch.clamp(final_image/255.0, 0.0, 1.0)
    return final_image.detach(), final_patch


def mifgsm(
    image_tensor, tgt_tensor, ensemble_extractor, ensemble_loss,
    source_crop, target_crop, center=None,
    num_iters=10, epsilon=8, alpha=1.0, decay=1.0, device='cuda'
):
    set_environment(42)
    init_patch = image_tensor.clone().detach().to(device)
    print("patch:", torch.min(init_patch), torch.max(init_patch), init_patch.shape)
    print("ori:", torch.min(image_tensor), torch.max(image_tensor), image_tensor.shape)
    print("tgt:", torch.min(tgt_tensor), torch.max(tgt_tensor), tgt_tensor.shape)
    momentum = torch.zeros_like(init_patch).to(device)
    delta = torch.zeros_like(init_patch, requires_grad=True).to(device)
    _, _, H, W = image_tensor.shape
    patch_generator = DynamicPatchGenerator(image_size=(W, H), device=device).to(device)
    threshold_list = [round(min(0.6 + i * 0.02, 0.95), 6) for i in range(num_iters)]
    mask_expanded = torch.ones([1, 1, H, W]).to(device)
    trans = My_T(num_copies=1)
    th = int(H * W * (120 * 120) / (900 * 1600))
    for iter in range(num_iters):
        with torch.no_grad():
            tgt = target_crop(tgt_tensor)
            ensemble_loss.set_ground_truth(tgt)
            ensemble_loss.set_full_feature(tgt_tensor)


        adv_patch = init_patch + delta
        if center is not None:

            if mask_expanded.sum()  > th:
                mask_expanded = patch_generator(center, threshold_list[iter])
            adv_image = image_tensor * (1 - mask_expanded) + adv_patch * mask_expanded
        else:
            adv_image = apply_patch(image=image_tensor, patch=adv_patch)

        local = torch.cat([trans.transform(source_crop(adv_image)) for _ in range(1)])
        features, features_local = ensemble_extractor(local)
        full = ensemble_extractor.encode_img(adv_image)
        loss_a = ensemble_loss(features, features_local, full)
        loss = loss_a


        if mask_expanded.sum()  > th:
            grads = torch.autograd.grad(loss, [delta, mask_expanded], create_graph=False)
            grad = grads[0]
            grad_mask = grads[1]
            print(iter, ":", "loss:", loss.detach().cpu().numpy(), "grad:", grad.mean().detach().cpu().numpy(), "mask_grads:", grad_mask.mean().detach().cpu().numpy())
        else:
            grads = torch.autograd.grad(loss, [delta], create_graph=False)
            grad = grads[0]
            print(iter, ":", "loss:", loss.detach().cpu().numpy(), "grad:", grad.mean().detach().cpu().numpy())

        momentum = decay * momentum + grad
        delta.data = torch.clamp(
            delta + alpha * torch.sign(momentum),
            min=-epsilon,
            max=epsilon,
        )
        if mask_expanded.sum()  > th:
            with torch.no_grad():
                patch_generator.update_potential_field(grad_mask, iter)

    final_patch = init_patch + delta
    final_image = image_tensor * (1 - mask_expanded) + final_patch * mask_expanded
    final_image = torch.clamp(final_image/255.0, 0.0, 1.0)
    return final_image.detach(), final_patch
