# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AdvDiffVLM generates **targeted and transferable** adversarial examples for Vision-Language Models (VLMs) using latent diffusion models.

**Attack Type:**
- **Targeted**: Maximizes similarity to specific target images
- **White-box during generation**: Computes gradients through CLIP ensemble (surrogate models)
- **Black-box transfer goal**: Intended to transfer to other VLMs not used during generation

**Key Techniques:**
- **AEGE (Adaptive Ensemble Gradient Estimation)**: Uses gradient-based optimization through an ensemble of 4 CLIP models to improve transferability
- **GCMG (GradCAM-guided Mask Generation)**: Spatially distributes perturbations across salient regions

Based on CompVis/latent-diffusion. Paper: "Efficient Generation of Targeted and Transferable Adversarial Examples for Vision-Language Models Via Diffusion Models" (IEEE TIFS 2024).

## Quick Start

### 1. Data Preparation

Prepare your data following this structure:
```
data/
  attack_clean/0/           # Clean input images (00000.png, 00001.png, ...)
  attack_target/0/          # Target images (same naming)
  images_subset.csv         # ImageId,TrueLabel mapping
```

**Create images_subset.csv:**
```python
import pandas as pd
data = [{'ImageId': f'sample_{i:05d}', 'TrueLabel': 920} for i in range(5)]
pd.DataFrame(data).to_csv('data/images_subset.csv', index=False)
```

### 2. Generate GradCAM Masks

```bash
python scripts/generate_gradcam_masks.py
```
Creates masks in `data/attack_masks/` named by ImageId (e.g., `sample_00000.png`).

### 3. Run Attack

Edit `main.py` line 191 to set number of images:
```python
if i >= 5:  # Process 5 images
    break
```

Then run:
```bash
python main.py
```

Outputs: `00000.png`, `00001.png`, etc. in the working directory.

## Verified Results

**Test run (5 samples, stop sign target):**
- Average CLIP similarity: **0.93** (scale 0-1, higher = more similar)
- All images achieved >0.90 similarity to target
- Time: ~70 seconds per image on GPU (200 DDIM steps × 10 iterations)

## Architecture

### Pipeline
```
Clean Image → VAE Encoder → Latent z (64×64×3)
                                ↓
                    DDIM Sampling (200 steps, 10 iterations)
                    + AEGE gradient guidance
                    + GradCAM mask
                                ↓
                    VAE Decoder → Adversarial Image
```

### Key Attack Code

**In `ldm/models/diffusion/ddim_main.py` (lines 453-477):**
```python
with torch.enable_grad():
    img_n = img.detach().requires_grad_(True)
    # Decode latent → image
    img_transformed = self.model.differentiable_decode_first_stage(img_n)
    img_transformed = torch.clamp((img_transformed+1.0)/2.0, min=0.0, max=1.0)
    img_transformed = self.preprocess(img_transformed)

    # Encode with CLIP ensemble
    adv_image_feature_list = []
    for model in self.models:
        adv_image_features = model.encode_image(img_transformed)
        adv_image_features = adv_image_features / adv_image_features.norm(dim=1, keepdim=True)
        adv_image_feature_list.append(adv_image_features)

    # Compute loss (maximize similarity to target)
    loss = torch.zeros(1).to(device)
    for model_i, (pred_i, target_i) in enumerate(zip(adv_image_feature_list, tgt_image_features_list)):
        crit1 = torch.mean(torch.sum(pred_i * target_i, dim=1))  # Cosine similarity
        loss.add_(crit1, alpha=weights[i-idx_time, model_i])

    # Backprop through CLIP → gradient on latent
    gradient = torch.autograd.grad(loss, img_n)[0]

gradient = torch.clamp(gradient, min=-0.0025, max=0.0025)
img = img + s * gradient  # s=35 by default
```

**Key insight:** Attack only modifies the last 20% of diffusion timesteps (line 384: `if index > total_steps * 0.2: continue`).

### Important Files

- **`main.py`**: Main attack script, loads data and runs attack loop
- **`ldm/models/diffusion/ddim_main.py`**: DDIMSampler with AEGE implementation
- **`ldm/models/diffusion/ddpm.py`**: Base LatentDiffusion model
- **`scripts/generate_gradcam_masks.py`**: GradCAM mask generation
- **`configs/latent-diffusion/cin256-v2.yaml`**: Model configuration

## Hyperparameters

**Attack parameters** (in `main.py` line 253 and 269):
```python
sampler.sample(
    S=200,              # DDIM steps
    K=1,                # Ensemble iterations per timestep
    s=35,               # Gradient scale (higher = stronger attack)
    a=5,                # Adversarial strength (unused in current code)
    unconditional_guidance_scale=5.0,  # Classifier-free guidance
    ...
)
```

**Trade-offs:**
- Increase `s` (35→50): Stronger attack, less natural appearance
- Increase `ddim_steps` (200→500): Better quality, slower
- Increase refinement iterations (10→20): Higher similarity, much slower

## Models Required

1. **Latent Diffusion Model**: `models/ldm/cin256-v2/model.ckpt` (1.8GB)
   - Download from CompVis/latent-diffusion

2. **CLIP Models** (cached in `data/clip_cache/`):
   - RN50 (256MB)
   - RN101 (292MB)
   - ViT-B/16 (351MB)
   - ViT-B/32 (354MB)

## Common Issues

1. **KeyError in GradCAM generation**: Ensure `images_subset.csv` has entries for all `00000.png`, `00001.png`, etc. files in `data/attack_clean/0/`

2. **CUDA OOM**: Reduce batch size (already 1) or use smaller CLIP ensemble (edit line 97 in `main.py`)

3. **demo.py merge conflicts**: Use `main.py` instead

4. **Slow execution**: First 20% of timesteps skip gradient computation (fast), last 20% compute CLIP gradients (slow ~7s per iteration)
