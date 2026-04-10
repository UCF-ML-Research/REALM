#!/usr/bin/env python3
"""FARE: Unsupervised adversarial fine-tuning of the Qwen3-VL vision encoder.

Adapts the FARE loss from Schlarmann et al., ICML 2024 (github.com/chs20/RobustVLM)
to the Qwen3-VL vision architecture (Conv3d patch embed, RoPE, spatial merger,
DeepStack multi-level features).

  L = E_x [ || f_θ(x_adv) - f_orig(x) ||^2 ]

where x_adv = argmax_{||δ||∞ ≤ ε} || f_θ(x+δ) - f_orig(x) ||^2

The perturbation is applied in [0,1] pixel space. A differentiable
pixel_to_patches function converts images to the flattened patch format
expected by Qwen3VLVisionModel.forward(), ensuring gradients flow back to
pixels through reshape/permute/expand operations only.

Usage:
    python -m vlm_benchmark.defense.robustclip.core.train_fare_qwen \
        --imagenet_dir /path/to/imagenet/train \
        --output_dir vlm_benchmark/defense/robustclip/checkpoints/fare_qwen \
        --device cuda:0
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# Qwen3.5 uses simple [0.5, 0.5, 0.5] normalization (from preprocessor_config.json),
# NOT the CLIP normalization that older Qwen2-VL models use.
_QWEN_MEAN = (0.5, 0.5, 0.5)
_QWEN_STD = (0.5, 0.5, 0.5)


# ── Vision encoder loading ───────────────────────────────────────────────────

def load_qwen_vision_encoder(model_id, device, dtype=torch.bfloat16):
    """Load only the vision encoder from a Qwen3-VL model.

    Instantiates Qwen3VLVisionModel from the config and loads only the
    visual weights from the relevant safetensors shard, avoiding loading
    the full LLM into memory.

    Args:
        model_id: HuggingFace model ID (e.g. "Qwen/Qwen3-VL-8B-Instruct")
        device: torch device
        dtype: weight dtype (default: bf16)

    Returns:
        Qwen3VLVisionModel with loaded weights
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    # Step 1: Get vision config from model's config.json
    config_path = hf_hub_download(model_id, "config.json")
    with open(config_path) as f:
        full_config = json.load(f)

    vision_cfg_dict = full_config["vision_config"]
    vision_config = Qwen3VLVisionConfig(**vision_cfg_dict)
    # Use SDPA for efficient attention (flash_attn not required)
    vision_config._attn_implementation = "sdpa"

    # Step 2: Instantiate empty vision model
    vision_model = Qwen3VLVisionModel(vision_config).to(dtype=dtype, device="cpu")

    # Step 3: Find and load safetensors shards containing visual weights
    index_path = hf_hub_download(model_id, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)

    # Find which shards contain visual weights
    visual_shards = set()
    for key, shard in index["weight_map"].items():
        if key.startswith("model.visual."):
            visual_shards.add(shard)

    print(f"Loading visual weights from shards: {visual_shards}")

    # Load and filter visual weights from each shard
    visual_state_dict = {}
    for shard_name in visual_shards:
        shard_path = hf_hub_download(model_id, shard_name)
        shard_weights = load_file(shard_path)
        for key, value in shard_weights.items():
            if key.startswith("model.visual."):
                # Strip "model.visual." prefix to match vision model keys
                clean_key = key[len("model.visual."):]
                visual_state_dict[clean_key] = value

    print(f"Loaded {len(visual_state_dict)} visual weight keys")
    vision_model.load_state_dict(visual_state_dict, strict=True)
    vision_model = vision_model.to(device)

    return vision_model


# ── Differentiable pixel-to-patches ──────────────────────────────────────────

def pixel_to_patches(images, temporal_patch_size=2, patch_size=16, merge_size=2):
    """Convert [B, 3, H, W] images to flattened patches in merge-aware order.

    Replicates the Qwen2VLImageProcessor._preprocess patch layout using only
    reshape/permute/expand so that gradients flow back to pixel inputs.

    The merge-aware ordering groups patches within each (merge_size × merge_size)
    spatial block contiguously, matching what the vision model expects.

    Args:
        images: [B, 3, H, W] tensor (any range — pre- or post-normalization)
        temporal_patch_size: temporal patch size (default: 2, duplicates frame)
        patch_size: spatial patch size (default: 16)
        merge_size: spatial merge size (default: 2)

    Returns:
        hidden_states: [B * grid_h * grid_w, C * temporal_patch_size * patch_size²]
        grid_thw: [B, 3] LongTensor with (grid_t, grid_h, grid_w) per image
    """
    B, C, H, W = images.shape
    grid_t = 1  # single image → 1 temporal group
    grid_h = H // patch_size
    grid_w = W // patch_size

    # Duplicate frame for temporal patch: [B, temporal_patch_size, C, H, W]
    x = images.unsqueeze(1).expand(-1, temporal_patch_size, -1, -1, -1)

    # Reshape to patch grid matching Qwen2VLImageProcessor._preprocess:
    # dims: (B, grid_t, temporal_patch_size, C,
    #         grid_h//merge, merge_h, patch_h,
    #         grid_w//merge, merge_w, patch_w)
    x = x.reshape(
        B,
        grid_t,
        temporal_patch_size,
        C,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )

    # Transpose to merge-aware order:
    # (B, grid_t, h_blocks, w_blocks, merge_h, merge_w, C, temporal, patch_h, patch_w)
    x = x.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)

    # Flatten to: (B * grid_t * grid_h * grid_w, C * temporal_patch_size * patch_size²)
    x = x.reshape(
        B * grid_t * grid_h * grid_w,
        C * temporal_patch_size * patch_size * patch_size,
    )

    grid_thw = torch.tensor(
        [[grid_t, grid_h, grid_w]], dtype=torch.long, device=images.device,
    ).expand(B, -1)

    return x, grid_thw


# ── Normalization ────────────────────────────────────────────────────────────

def _qwen_normalize(x: torch.Tensor) -> torch.Tensor:
    """Apply Qwen-VL ImageNet normalization to [0,1] tensor."""
    mean = torch.tensor(_QWEN_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(_QWEN_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


def _normalize_grad(grad: torch.Tensor) -> torch.Tensor:
    """L-inf gradient normalization (sign)."""
    return grad.sign()


# ── Batched vision forward (bypasses slow per-chunk attention loop) ───────────

def _fast_patch_embed(patch_embed, hidden_states):
    """Fast patch embedding using Linear instead of Conv3d.

    Conv3d with kernel_size==stride is mathematically identical to reshape+linear,
    but Conv3d's CUDA kernel is ~5000× slower for large batch counts (~6272 patches).
    We replace it with an equivalent Linear operation.
    """
    # hidden_states: (N, C * T * Hp * Wp) e.g. (N, 1536) for 3*2*16*16
    # Conv3d weight: (embed_dim, C, T, Hp, Wp) → reshape to (embed_dim, C*T*Hp*Wp)
    weight = patch_embed.proj.weight.reshape(patch_embed.embed_dim, -1)  # (1152, 1536)
    bias = patch_embed.proj.bias  # (1152,)
    target_dtype = weight.dtype
    return F.linear(hidden_states.to(dtype=target_dtype), weight, bias)


def _batched_vision_forward(vision_encoder, hidden_states, grid_thw):
    """Fast batched forward through Qwen3VL vision encoder.

    Two key optimizations over the default forward:
    1. Replaces Conv3d patch_embed with Linear (~5000× faster)
    2. Uses batched SDPA attention instead of per-chunk Python loop

    Args:
        vision_encoder: Qwen3VLVisionModel
        hidden_states: (B * seq_per_image, patch_dim) flattened patches
        grid_thw: (B, 3) grid dimensions

    Returns:
        (merged_output, deepstack_features)
    """
    B = grid_thw.shape[0]
    seq_per_image = int(grid_thw[0, 0] * grid_thw[0, 1] * grid_thw[0, 2])
    num_heads = vision_encoder.config.num_heads
    head_dim = vision_encoder.config.hidden_size // num_heads

    # Fast patch embed via Linear (replaces slow Conv3d)
    hidden_states = _fast_patch_embed(vision_encoder.patch_embed, hidden_states)

    # Position embeddings
    pos_embeds = vision_encoder.fast_pos_embed_interpolate(grid_thw)
    hidden_states = hidden_states + pos_embeds

    # Rotary embeddings
    rotary_pos_emb = vision_encoder.rot_pos_emb(grid_thw)
    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)

    # Reshape to batched: (B, seq_per_image, hidden_size)
    hidden_states = hidden_states.reshape(B, seq_per_image, -1)
    emb_batched = emb.reshape(B, seq_per_image, -1)
    cos_batched = emb_batched.cos()
    sin_batched = emb_batched.sin()

    # Transformer blocks with batched SDPA
    deepstack_feature_lists = []
    for layer_num, blk in enumerate(vision_encoder.blocks):
        hidden_states = _batched_block_forward(
            blk, hidden_states, cos_batched, sin_batched, num_heads, head_dim,
        )
        if layer_num in vision_encoder.deepstack_visual_indexes:
            idx = vision_encoder.deepstack_visual_indexes.index(layer_num)
            flat_hs = hidden_states.reshape(B * seq_per_image, -1)
            ds_feat = vision_encoder.deepstack_merger_list[idx](flat_hs)
            deepstack_feature_lists.append(ds_feat)

    # Final merger
    hidden_states = hidden_states.reshape(B * seq_per_image, -1)
    merged_output = vision_encoder.merger(hidden_states)

    return merged_output, deepstack_feature_lists


def _batched_block_forward(blk, hidden_states, cos, sin, num_heads, head_dim):
    """Batched vision block: (B, seq, dim) throughout, single SDPA call."""
    from transformers.models.qwen2_vl.modeling_qwen2_vl import apply_rotary_pos_emb_vision

    B, S, D = hidden_states.shape

    # Attention
    residual = hidden_states
    normed = blk.norm1(hidden_states)

    qkv = blk.attn.qkv(normed.reshape(B * S, D)).reshape(B, S, 3, num_heads, head_dim)
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

    # Rotary embeddings
    q_flat = q.reshape(B * S, num_heads, head_dim)
    k_flat = k.reshape(B * S, num_heads, head_dim)
    cos_flat = cos.reshape(B * S, -1)
    sin_flat = sin.reshape(B * S, -1)
    q_flat, k_flat = apply_rotary_pos_emb_vision(q_flat, k_flat, cos_flat, sin_flat)
    q = q_flat.reshape(B, S, num_heads, head_dim).transpose(1, 2)
    k = k_flat.reshape(B, S, num_heads, head_dim).transpose(1, 2)
    v = v.transpose(1, 2)

    # Single batched SDPA call
    attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    attn_out = attn_out.transpose(1, 2).reshape(B * S, D)
    attn_out = blk.attn.proj(attn_out).reshape(B, S, D)
    hidden_states = residual + attn_out

    # MLP
    residual = hidden_states
    normed = blk.norm2(hidden_states)
    mlp_out = blk.mlp(normed.reshape(B * S, D)).reshape(B, S, D)
    hidden_states = residual + mlp_out

    return hidden_states


# ── Embedding extraction ─────────────────────────────────────────────────────

def encode_images(vision_encoder, images_01, normalize_fn, patch_size=16,
                  temporal_patch_size=2, merge_size=2):
    """Encode [B, 3, H, W] images in [0,1] through the Qwen3VL vision encoder.

    Uses batched SDPA forward (bypassing slow per-chunk attention loop) for
    ~100× speedup over the default cu_seqlens path without flash_attn.

    Returns mean-pooled merger output: [B, out_hidden_size] embedding per image.

    Args:
        vision_encoder: Qwen3VLVisionModel instance
        images_01: [B, 3, H, W] tensor in [0, 1]
        normalize_fn: pixel normalization function
        patch_size: spatial patch size
        temporal_patch_size: temporal patch size
        merge_size: spatial merge size

    Returns:
        [B, out_hidden_size] mean-pooled embeddings
    """
    B = images_01.shape[0]

    # Normalize pixels
    x_norm = normalize_fn(images_01)

    # Differentiable pixel → patch conversion
    hidden_states, grid_thw = pixel_to_patches(
        x_norm, temporal_patch_size, patch_size, merge_size,
    )

    # Forward through vision encoder (batched — avoids per-chunk loop)
    hidden_states = hidden_states.to(dtype=vision_encoder.patch_embed.proj.weight.dtype)
    merged_output, _deepstack = _batched_vision_forward(
        vision_encoder, hidden_states, grid_thw,
    )

    # Mean-pool over sequence dimension per image
    # After merger: (total_merged_tokens, out_hidden_size)
    # Merged tokens per image = grid_h * grid_w / merge_size²
    tokens_per_image = (grid_thw[0, 1] * grid_thw[0, 2]) // (merge_size ** 2)
    merged_output = merged_output.reshape(B, tokens_per_image, -1)
    embeddings = merged_output.mean(dim=1)  # [B, out_hidden_size]

    return embeddings


# ── PGD attack ───────────────────────────────────────────────────────────────

def pgd_attack(encode_fn, x_clean, anchor_emb, epsilon, alpha, steps,
               momentum=0.9):
    """Inner momentum-PGD maximization for FARE.

    Finds x_adv that maximizes || f_θ(x_adv) - f_orig(x_clean) ||²
    subject to || x_adv - x_clean ||∞ ≤ ε.

    Args:
        encode_fn: callable(images_01) → [B, D] embedding
        x_clean: [B, C, H, W] tensor in [0, 1]
        anchor_emb: [B, D] frozen clean embedding from original model
        epsilon: L-inf perturbation bound
        alpha: PGD step size
        steps: number of PGD iterations
        momentum: momentum coefficient

    Returns:
        x_adv: [B, C, H, W] adversarial images in [0, 1]
    """
    delta = torch.empty_like(x_clean).uniform_(-epsilon, epsilon)
    delta = delta.clamp(-(x_clean.detach()), 1 - x_clean.detach())
    velocity = torch.zeros_like(x_clean)

    for _ in range(steps):
        delta.requires_grad_(True)
        x_adv = x_clean + delta
        emb = encode_fn(x_adv)
        loss = F.mse_loss(emb, anchor_emb, reduction="none").sum(dim=1).mean()
        grad = torch.autograd.grad(loss, delta)[0]

        grad_normed = _normalize_grad(grad)
        velocity = momentum * velocity + grad_normed
        velocity_normed = _normalize_grad(velocity)

        # Maximize → step in gradient direction
        delta = delta.detach() + alpha * velocity_normed
        delta = delta.clamp(-epsilon, epsilon)
        delta = delta.clamp(-(x_clean.detach()), 1 - x_clean.detach())

    return (x_clean + delta).detach()


def fare_loss(trainable_emb, anchor_emb):
    """FARE outer loss: L2 embedding matching (unnormalized).

    sum over embedding dim, mean over batch — matches official.
    """
    return F.mse_loss(trainable_emb, anchor_emb, reduction="none").sum(dim=1).mean()


# ── Data ─────────────────────────────────────────────────────────────────────

def build_dataloader(imagenet_dir, image_size, batch_size, num_workers=4):
    """Build ImageNet training dataloader (no labels needed)."""
    transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
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


# ── Training loop ────────────────────────────────────────────────────────────

def train_fare_qwen(args):
    """Main FARE training loop for Qwen3-VL vision encoder."""
    device = torch.device(args.device)
    dtype = torch.bfloat16
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load models ───────────────────────────────────────────────────────
    print(f"Loading Qwen3VL vision encoder from: {args.model_id}")
    trainable_encoder = load_qwen_vision_encoder(args.model_id, device, dtype)
    frozen_encoder = load_qwen_vision_encoder(args.model_id, device, dtype)
    frozen_encoder.eval()
    for p in frozen_encoder.parameters():
        p.requires_grad_(False)

    # Train all vision encoder parameters (matches official FARE — no freezing)
    for p in trainable_encoder.parameters():
        p.requires_grad_(True)

    trainable_params = [p for p in trainable_encoder.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in trainable_params):,}")
    print(f"Frozen params: {sum(p.numel() for p in trainable_encoder.parameters() if not p.requires_grad):,}")

    # Vision config for patch dimensions
    v_cfg = trainable_encoder.config
    patch_size = v_cfg.patch_size
    temporal_patch_size = v_cfg.temporal_patch_size
    merge_size = v_cfg.spatial_merge_size

    # ── Optimizer + scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    dataloader = build_dataloader(
        args.imagenet_dir, args.image_size, args.batch_size,
    )
    total_steps = args.epochs * len(dataloader)
    # Cap warmup at configured value or 7% of total (matching official FARE)
    warmup_steps = min(args.warmup_steps, max(1, int(total_steps * 0.07)))

    def lr_schedule(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # ── Encode helpers ────────────────────────────────────────────────────
    def encode_trainable(images_01):
        return encode_images(
            trainable_encoder, images_01, _qwen_normalize,
            patch_size, temporal_patch_size, merge_size,
        )

    def encode_frozen(images_01):
        return encode_images(
            frozen_encoder, images_01, _qwen_normalize,
            patch_size, temporal_patch_size, merge_size,
        )

    # ── Training ──────────────────────────────────────────────────────────
    print(f"\nFARE Training (Qwen3-VL):")
    print(f"  Model:      {args.model_id}")
    print(f"  Image size: {args.image_size}")
    print(f"  Epsilon:    {args.epsilon:.5f} ({args.epsilon * 255:.1f}/255)")
    print(f"  PGD steps:  {args.pgd_steps}")
    print(f"  PGD alpha:  {args.pgd_alpha:.5f} ({args.pgd_alpha * 255:.1f}/255)")
    print(f"  LR:         {args.lr}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Warmup:     {warmup_steps} steps")
    print(f"  Total steps: {total_steps}")
    print(f"  Output:     {output_dir}\n")

    global_step = 0
    t0 = time.time()

    for epoch in range(args.epochs):
        trainable_encoder.train()

        epoch_loss = 0.0
        n_batches = 0

        for images, _ in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            images = images.to(device, non_blocking=True)
            # images are in [0, 1] from ToTensor()

            # 1. Get anchor embeddings from frozen original model
            with torch.no_grad():
                anchor_emb = encode_frozen(images)

            # 2. PGD inner maximization
            trainable_encoder.eval()
            with torch.amp.autocast("cuda", dtype=dtype):
                x_adv = pgd_attack(
                    encode_trainable, images, anchor_emb,
                    epsilon=args.epsilon,
                    alpha=args.pgd_alpha,
                    steps=args.pgd_steps,
                )

            # 3. Outer minimization
            trainable_encoder.train()
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", dtype=dtype):
                adv_emb = encode_trainable(x_adv)
                loss = fare_loss(adv_emb, anchor_emb)

            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            log_every = max(1, min(100, total_steps // 20))
            if global_step % log_every == 0:
                avg = epoch_loss / n_batches
                lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - t0
                print(f"  step {global_step:>6d}  loss={avg:.4f}  lr={lr:.2e}  "
                      f"elapsed={elapsed / 60:.1f}min")

        avg_loss = epoch_loss / max(1, n_batches)
        print(f"Epoch {epoch + 1}: avg_loss={avg_loss:.4f}")

        # Save checkpoint (vision encoder only)
        ckpt_path = output_dir / f"fare_qwen_vision_epoch{epoch + 1}.pt"
        torch.save(trainable_encoder.state_dict(), ckpt_path)
        print(f"  Saved: {ckpt_path}")

    # Final save
    final_path = output_dir / "fare_qwen_vision_final.pt"
    torch.save(trainable_encoder.state_dict(), final_path)
    print(f"\nTraining complete. Final checkpoint: {final_path}")
    print(f"Total time: {(time.time() - t0) / 60:.1f} minutes")


def main():
    from ..config import FARE_QWEN_TRAIN_DEFAULTS as D

    parser = argparse.ArgumentParser(
        description="FARE: adversarial fine-tuning of Qwen3-VL vision encoder",
    )
    parser.add_argument("--imagenet_dir", required=True,
                        help="Path to ImageNet train directory")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for checkpoints")
    parser.add_argument("--model_id", default=D["model_id"],
                        help="HuggingFace model ID")
    parser.add_argument("--image_size", type=int, default=D["image_size"])
    parser.add_argument("--epochs", type=int, default=D["epochs"])
    parser.add_argument("--batch_size", type=int, default=D["batch_size"])
    parser.add_argument("--lr", type=float, default=D["lr"])
    parser.add_argument("--weight_decay", type=float, default=D["weight_decay"])
    parser.add_argument("--warmup_steps", type=int, default=D["warmup_steps"])
    parser.add_argument("--epsilon", type=float, default=D["epsilon"],
                        help="L-inf perturbation bound in [0,1] space")
    parser.add_argument("--pgd_steps", type=int, default=D["pgd_steps"])
    parser.add_argument("--pgd_alpha", type=float, default=D["pgd_alpha"],
                        help="PGD step size in [0,1] space")
    parser.add_argument("--device", default="cuda:0")
    train_fare_qwen(parser.parse_args())


if __name__ == "__main__":
    main()
