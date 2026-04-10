#!/usr/bin/env python3
"""FARE: Unsupervised adversarial fine-tuning of the CLIP vision encoder.

Implements the FARE loss from Schlarmann et al., ICML 2024:

  L = E_x [ || f_θ(x_adv) - f_orig(x) ||^2 ]

where x_adv = argmax_{||δ||∞ ≤ ε} || f_θ(x+δ) - f_orig(x) ||^2

No labels are required — training is fully unsupervised on ImageNet images.

Usage:
    python -m vlm_benchmark.defense.robustclip.core.train_fare \
        --imagenet_dir /path/to/imagenet/train \
        --output_dir vlm_benchmark/defense/robustclip/checkpoints/fare2 \
        --epsilon 0.00784 \
        --device cuda:0
"""

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# CLIP ImageNet normalization (applied before encode_image)
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _clip_normalize(x: torch.Tensor) -> torch.Tensor:
    """Apply CLIP ImageNet normalization to [0,1] tensor."""
    mean = torch.tensor(_CLIP_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(_CLIP_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


def _normalize_grad(grad: torch.Tensor) -> torch.Tensor:
    """L-inf gradient normalization (sign)."""
    return grad.sign()


def pgd_attack(model_fn, normalize_fn, x_clean, anchor_emb, epsilon, alpha,
               steps, momentum=0.9):
    """Inner momentum-PGD maximization for FARE (matches official implementation).

    Finds x_adv that maximizes || f_θ(normalize(x_adv)) - f_orig(normalize(x_clean)) ||^2
    subject to || x_adv - x_clean ||∞ ≤ ε.

    Args:
        model_fn:      callable(normalized_x) → embedding
        normalize_fn:  CLIP pixel normalization
        x_clean:       [B, C, H, W] tensor in [0, 1]
        anchor_emb:    [B, D] frozen clean embedding from original model
        epsilon:       L-inf perturbation bound
        alpha:         PGD step size
        steps:         number of PGD iterations
        momentum:      momentum coefficient (default: 0.9, matches official)

    Returns:
        x_adv: [B, C, H, W] adversarial images in [0, 1]
    """
    # Random start within ε-ball
    delta = torch.empty_like(x_clean).uniform_(-epsilon, epsilon)
    delta = delta.clamp(-(x_clean.detach()), 1 - x_clean.detach())  # ensure [0,1]
    velocity = torch.zeros_like(x_clean)

    for _ in range(steps):
        delta.requires_grad_(True)
        x_adv = x_clean + delta
        emb = model_fn(normalize_fn(x_adv))
        loss = F.mse_loss(emb, anchor_emb, reduction="none").sum(dim=1).mean()
        grad = torch.autograd.grad(loss, delta)[0]

        # Momentum PGD (matches official: normalize grad, accumulate, normalize velocity)
        grad_normed = _normalize_grad(grad)
        velocity = momentum * velocity + grad_normed
        velocity_normed = _normalize_grad(velocity)

        # Maximize → step in gradient direction
        delta = delta.detach() + alpha * velocity_normed
        # Project onto ε-ball and [0, 1]
        delta = delta.clamp(-epsilon, epsilon)
        delta = delta.clamp(-(x_clean.detach()), 1 - x_clean.detach())

    return (x_clean + delta).detach()


def fare_loss(trainable_emb, anchor_emb):
    """FARE outer loss: L2 embedding matching (unnormalized).

    sum over embedding dim, mean over batch — matches official.
    """
    return F.mse_loss(trainable_emb, anchor_emb, reduction="none").sum(dim=1).mean()


def build_dataloader(imagenet_dir, batch_size, num_workers=4):
    """Build ImageNet training dataloader (no labels needed)."""
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(imagenet_dir, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def train_fare(args):
    """Main FARE training loop."""
    import open_clip

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load models ───────────────────────────────────────────────────────
    print(f"Loading CLIP {args.clip_model_name} (pretrained={args.pretrained})")
    model, _, _ = open_clip.create_model_and_transforms(
        args.clip_model_name, pretrained=args.pretrained,
    )
    model = model.to(device)

    # Frozen original model for anchor embeddings
    orig_model, _, _ = open_clip.create_model_and_transforms(
        args.clip_model_name, pretrained=args.pretrained,
    )
    orig_model = orig_model.to(device).eval()
    for p in orig_model.parameters():
        p.requires_grad_(False)

    # Freeze text encoder — only train vision
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.visual.parameters():
        p.requires_grad_(True)

    trainable_params = [p for p in model.visual.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in trainable_params):,}")

    # ── Optimizer + scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    dataloader = build_dataloader(args.imagenet_dir, args.batch_size)
    total_steps = args.epochs * len(dataloader)
    warmup_steps = args.warmup_steps

    def lr_schedule(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # ── Training ──────────────────────────────────────────────────────────
    print(f"\nFARE Training:")
    print(f"  Epsilon:    {args.epsilon:.5f} ({args.epsilon * 255:.1f}/255)")
    print(f"  PGD steps:  {args.pgd_steps}")
    print(f"  PGD alpha:  {args.pgd_alpha:.5f} ({args.pgd_alpha * 255:.1f}/255)")
    print(f"  LR:         {args.lr}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Total steps: {total_steps}")
    print(f"  Output:     {output_dir}\n")

    global_step = 0
    t0 = time.time()

    def encode_fn(x_norm):
        """Encode already-normalized image tensor."""
        return model.encode_image(x_norm, normalize=False)

    def orig_encode_fn(x_norm):
        """Encode already-normalized image tensor through frozen model."""
        return orig_model.encode_image(x_norm, normalize=False)

    for epoch in range(args.epochs):
        model.visual.train()
        epoch_loss = 0.0
        n_batches = 0

        for images, _ in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            images = images.to(device, non_blocking=True)
            # images are in [0, 1] from ToTensor()

            # 1. Get anchor embeddings from frozen original model
            with torch.no_grad():
                anchor_emb = orig_encode_fn(_clip_normalize(images))

            # 2. PGD inner maximization (model.eval() during attack, matches official)
            model.eval()
            x_adv = pgd_attack(
                encode_fn, _clip_normalize, images, anchor_emb,
                epsilon=args.epsilon,
                alpha=args.pgd_alpha,
                steps=args.pgd_steps,
            )

            # 3. Outer minimization (model.train() for outer loss, matches official)
            model.visual.train()
            optimizer.zero_grad()
            adv_emb = model.encode_image(_clip_normalize(x_adv), normalize=False)
            loss = fare_loss(adv_emb, anchor_emb)

            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            if global_step % 500 == 0:
                avg = epoch_loss / n_batches
                lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - t0
                print(f"  step {global_step:>6d}  loss={avg:.4f}  lr={lr:.2e}  "
                      f"elapsed={elapsed / 60:.1f}min")

        avg_loss = epoch_loss / max(1, n_batches)
        print(f"Epoch {epoch + 1}: avg_loss={avg_loss:.4f}")

        # Save checkpoint (vision encoder only, matches official)
        ckpt_path = output_dir / f"fare_vision_epoch{epoch + 1}.pt"
        torch.save(model.visual.state_dict(), ckpt_path)
        print(f"  Saved: {ckpt_path}")

    # Final save
    final_path = output_dir / "fare_vision_final.pt"
    torch.save(model.visual.state_dict(), final_path)
    print(f"\nTraining complete. Final checkpoint: {final_path}")
    print(f"Total time: {(time.time() - t0) / 60:.1f} minutes")


def main():
    from ..config import FARE_TRAIN_DEFAULTS as D

    parser = argparse.ArgumentParser(description="FARE: adversarial fine-tuning of CLIP vision encoder")
    parser.add_argument("--imagenet_dir", required=True, help="Path to ImageNet train directory")
    parser.add_argument("--output_dir", required=True, help="Output directory for checkpoints")
    parser.add_argument("--clip_model_name", default=D["clip_model_name"])
    parser.add_argument("--pretrained", default=D["pretrained"])
    parser.add_argument("--epochs", type=int, default=D["epochs"])
    parser.add_argument("--batch_size", type=int, default=D["batch_size"])
    parser.add_argument("--lr", type=float, default=D["lr"])
    parser.add_argument("--weight_decay", type=float, default=D["weight_decay"])
    parser.add_argument("--warmup_steps", type=int, default=D["warmup_steps"])
    parser.add_argument("--epsilon", type=float, default=D["epsilon"],
                        help="L-inf perturbation bound in [0,1] space (default: 2/255)")
    parser.add_argument("--pgd_steps", type=int, default=D["pgd_steps"])
    parser.add_argument("--pgd_alpha", type=float, default=D["pgd_alpha"],
                        help="PGD step size in [0,1] space (default: 1/255)")
    parser.add_argument("--device", default="cuda:0")
    train_fare(parser.parse_args())


if __name__ == "__main__":
    main()
