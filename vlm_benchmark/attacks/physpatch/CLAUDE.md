# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the official implementation of **PhysPatch**, a physically realizable and transferable adversarial patch attack for Multimodal Large Language Models (MLLMs) used in autonomous driving systems. The research was accepted by AAAI 2026.

The project generates adversarial patches that can be physically placed in driving scenes to mislead MLLM-based autonomous driving perception systems. The attack is designed to be both transferable across different models and physically printable.

## Main Pipeline

The attack pipeline consists of four sequential stages:

### 1. Dataset Preparation
- **Dataset**: nuScenes autonomous driving dataset
- **Command**: `python dataset.py`
- **Location**: Images stored in `./nuscenes/samples`

### 2. Generate Placement Coordinates
This stage uses Set-of-Mark (SoM) prompting to identify optimal patch placement locations.

**Step 2.1** - Generate SoM masks and labels:
```bash
cd SoM
python batch_som.py \
  --input_dir ./nuscenes/samples \
  --output_dir ./som \
  --label_dir ./sam_label \
  --sam_ckpt ./checkpoints/sam_vit_h_4b8939.pth \
  --granularity 2.6 \
  --alpha 0.1 \
  --label_mode Number \
  --anno_mode Mask Mark
```

**Step 2.2** - Generate coordinate file using GPT-4V:
```bash
cd ..
python som_gpt.py \
  --original_folder ./nuscenes/samples \
  --sam_folder ./som \
  --label_folder ./sam_label \
  --output_path ./coords.txt \
  --api_key sk-xxx \
  --api_base base_url \
  --model gpt-4o
```

### 3. Generate Adversarial Examples
Main attack script using PGD or MI-FGSM:
```bash
python main.py \
  --cle_data_path ./data/clean \
  --tgt_data_path ./data/target \
  --output_dir ./results/pgd \
  --txt_path ./coords.txt \
  --epsilon 16 \
  --alpha 1.0 \
  --num_iters 300 \
  --num_samples 1000
```

Key parameters:
- `--epsilon`: Maximum perturbation magnitude (e.g., 16/255)
- `--alpha`: Step size for gradient updates
- `--num_iters`: Number of optimization iterations
- `--K`: SVD component count for feature extraction (default: 10)

### 4. Evaluation

**Step 4.1** - Generate perceptual descriptions from VLMs:
```bash
python vlm_response.py \
  --image_dir ./results/pgd/samples \
  --output_dir ./results \
  --model gpt-4o \
  --query "Describe the main object in the scene..."
```

**Step 4.2** - Calculate Attack Success Rate (ASR) and Average Similarity:
```bash
python evaluation.py \
  --file_path ./results/GPT-4o_response.txt \
  --model_name GPT-4o \
  --reference_text "A stop sign is visible" \
  --start 0 \
  --end 1000 \
  --api_key sk-xxx
```

## Important Files and Functions Reference

### Entry Points and Main Scripts
- **code/main.py** - Main attack pipeline entry point
  - `main()` - Line 137: Orchestrates full attack workflow
  - `get_models_ours()` - Line 62: Initializes ensemble of CLIP models
  - `get_ensemble_loss_ours()` - Line 82: Creates ensemble loss function
  - `parse_args()` - Line 118: Command-line argument parsing

- **code/dataset.py** - Dataset preparation and processing

- **code/som_gpt.py** - GPT-4V coordinate generation
  - `Gpt_model` class - Handles GPT-4V API calls for patch placement
  - Outputs coordinates to `coords.txt`

- **code/vlm_response.py** - VLM evaluation inference
  - `process_model()` - Line 7: Batch processes images through VLMs
  - `main()` - Line 50: Entry point for VLM evaluation

- **code/evaluation.py** - Attack metrics calculation
  - `GPTScorer` class - Line 25: Semantic similarity scoring
  - `compute_similarity()` - Line 32: Scores text similarity using GPT
  - `main()` - Line 87: Calculates ASR and AvgSim metrics

### Attack Algorithms
- **code/attacks.py**
  - `pgd()` - Line 76: Projected Gradient Descent attack implementation
  - `mifgsm()` - Line 138: Momentum Iterative FGSM attack
  - `tv_loss()` - Line 22: Total Variation loss for smoothness
  - `nps_loss()` - Line 56: Non-Printability Score loss
  - `My_T` class - Line 205: Input transformation/augmentation
    - `transform()` - Line 305: Applies random augmentations
    - `blocktransform()` - Line 300: Randomly selects one transformation
    - `resize()` - Line 237: Scale transformation
    - `dct()` - Line 249: DCT frequency filtering
    - `add_noise()` - Line 265: Noise injection
    - `adjust_brightness()` - Line 277: Brightness adjustment
    - `adjust_contrast()` - Line 286: Contrast adjustment

### Dynamic Mask Generation
- **code/dyn_mask.py**
  - `DynamicPatchGenerator` class - Line 51: Dynamic patch shape learning
    - `__init__()` - Line 52: Initializes potential field and coordinate grid
    - `initialize_gaussian_field()` - Line 71: Creates Gaussian-centered field
    - `get_mask()` - Line 84: Generates binary mask from potential field
    - `update_potential_field()` - Line 105: Updates mask based on gradients
    - `forward()` - Line 119: Main forward pass to generate mask
  - `remove_small_regions()` - Line 19: Filters out small disconnected regions

### Surrogate Models and Feature Extraction
- **code/surrogates/FeatureExtractors/__init__.py** - Exports all extractors

- **code/surrogates/FeatureExtractors/Base.py**
  - `BaseFeatureExtractor` - Line 8: Abstract base class
  - `EnsembleFeatureExtractor_ours` - Line 17: Ensemble wrapper
    - `forward()` - Line 23: Extracts global and local features
    - `encode_img()` - Line 34: Extracts full image features
    - `get_svd_feature()` - Line 41: SVD dimensionality reduction
  - `EnsembleFeatureLoss_ours` - Line 50: Multi-objective loss
    - `set_ground_truth()` - Line 66: Sets target features
    - `set_full_feature()` - Line 60: Sets full image features
    - `__call__()` - Line 77: Computes combined loss
  - `EnsembleFeatureLoss_ours_auto` - Automatic version of ensemble loss

- **code/surrogates/FeatureExtractors/ClipB16.py** - CLIP ViT-B/16 extractor
- **code/surrogates/FeatureExtractors/ClipB32.py** - CLIP ViT-B/32 extractor
- **code/surrogates/FeatureExtractors/ClipLaion.py** - CLIP LAION extractor

### VLM API Wrappers
- **code/models/gpt_model.py**
  - `Gpt_model` class - OpenAI API wrapper
  - `encode_image()` - Base64 encoding for images
  - `response()` - Sends images to GPT-4V/GPT-4o

- **code/models/claude_model.py** - Anthropic Claude API wrapper
- **code/models/gemini_model.py** - Google Gemini API wrapper
- **code/models/qwen_model.py** - Qwen VLM API wrapper

### Set-of-Mark (SoM) Components
- **code/SoM/batch_som.py** - Batch SoM annotation generation
- **code/SoM/demo_som.py** - Interactive SoM demo
- **code/SoM/demo_gpt4v_som.py** - SoM + GPT-4V integration demo
- **code/SoM/task_adapter/** - Segmentation model adapters
  - `sam/` - Segment Anything Model adapter
  - `seem/` - SEEM segmentation adapter
  - `semantic_sam/` - Semantic-SAM adapter

### Utility Functions
- **code/utils.py**
  - `RandomPointConstrainedCrop` - Crops around specified coordinates
  - `apply_patch()` - Applies patch to image
  - Image transformation utilities

## Code Architecture

### Core Modules

**main.py**: Main entry point for adversarial example generation
- Orchestrates the entire attack pipeline
- Loads surrogate models (CLIP variants)
- Applies attack algorithms (PGD/MI-FGSM)
- Saves adversarial images

**attacks.py**: Attack algorithms implementation
- `pgd()`: Projected Gradient Descent attack with dynamic mask
- `mifgsm()`: Momentum Iterative FGSM attack
- `My_T`: Input transformation class with 12+ augmentation operations (resize, shift, flip, DCT, noise, brightness/contrast adjustment)
- Loss functions: `tv_loss()` (Total Variation), `nps_loss()` (Non-Printability Score)

**dyn_mask.py**: Dynamic patch mask generation
- `DynamicPatchGenerator`: Learns patch shape during optimization using potential fields
- Uses Gaussian-based initialization centered at target coordinates
- Adaptively updates mask based on gradient feedback
- Applies morphological operations and spatial filtering for realistic patch shapes

**surrogates/**: Surrogate model feature extractors
- `ClipB16FeatureExtractor`: CLIP ViT-B/16 backbone
- `ClipB32FeatureExtractor`: CLIP ViT-B/32 backbone
- `ClipLaionFeatureExtractor`: CLIP trained on LAION dataset
- `EnsembleFeatureExtractor_ours`: Ensemble wrapper that extracts both global and local (SVD-reduced) features
- `EnsembleFeatureLoss_ours_auto`: Multi-objective loss combining global similarity, local feature MSE, and full-image consistency

**som_gpt.py**: Coordinate generation using GPT-4V
- Analyzes SoM-annotated images to identify optimal patch placement
- Uses VLM reasoning to select high-impact locations
- Outputs normalized (x, y) coordinates to `coords.txt`

**vlm_response.py**: VLM inference for evaluation
- Queries vision-language models (GPT-4o, Claude, Gemini, Qwen)
- Generates perceptual descriptions of adversarial images
- Supports batch processing of image directories

**evaluation.py**: Attack effectiveness metrics
- `GPTScorer`: Uses GPT-4 to compute semantic similarity scores
- Calculates Attack Success Rate (ASR) based on similarity threshold (>0.5)
- Computes Average Similarity (AvgSim) to measure attack strength

**utils.py**: Utility functions including image transformations and data loading

**models/**: VLM API wrappers
- `gpt_model.py`: OpenAI GPT-4V/GPT-4o interface
- `claude_model.py`: Anthropic Claude API interface
- `gemini_model.py`: Google Gemini API interface
- `qwen_model.py`: Qwen VLM interface

### SoM Directory

The `SoM/` directory contains the Set-of-Mark visual prompting implementation (from Microsoft Research). Key files:
- `batch_som.py`: Batch processing to generate SoM annotations
- `demo_som.py`, `demo_gpt4v_som.py`: Interactive demos
- `task_adapter/`: Adapters for SAM, SEEM, and Semantic-SAM segmentation models

## Key Technical Details

### Surrogate Model Ensemble
The attack uses an ensemble of three CLIP variants (B/16, B/32, LAION) as surrogate models. Features are extracted at two levels:
- **Global features**: Full image embeddings for coarse alignment
- **Local features**: SVD-reduced patch embeddings (top-k components, default k=8) for fine-grained matching

### Loss Function
The combined loss optimizes:
1. Global feature similarity (cosine similarity between adversarial crop and target)
2. Local feature MSE (SVD components of patch region vs. target)
3. Full-image consistency (0.1× weighted, maintains overall image coherence)

### Dynamic Mask Optimization
Unlike fixed rectangular patches, the mask shape is learned during optimization:
- Initialized as Gaussian centered at GPT-4V-selected coordinates
- Updated via gradient feedback on mask parameters
- Threshold increases over iterations (0.6 → 0.6+0.02×iter) to gradually refine shape
- Minimum area threshold (800 pixels) prevents degenerate solutions

### Input Transformations
`My_T` applies diverse augmentations to improve attack transferability:
- Geometric: resize, shifts, flips, rotation
- Frequency: DCT low-pass filtering
- Appearance: brightness, contrast, noise injection
- Regularization: dropout

## Important Notes

- This is adversarial ML research for autonomous driving security evaluation
- API keys are required for GPT-4V/VLM inference (not included in repo)
- SoM checkpoints must be downloaded separately (`download_ckpt.sh` in SoM/)
- The attack requires clean images and target images as input pairs
- Results are saved as PNG images with perturbed patches applied

## Research Context

This implementation focuses on physical-world adversarial attacks specifically targeting MLLM-based autonomous driving perception. The patches are designed with printability constraints (NPS loss) and smoothness (TV loss) to ensure they can be fabricated and deployed in real driving scenarios.

## Setup and Installation

### Required Dependencies

The code requires several dependencies that must be installed in the correct order:

```bash
# 1. Core dependencies
pip install torch torchvision opencv-python matplotlib tqdm pillow scipy numpy

# 2. Install detectron2 (without build isolation to avoid torch import errors)
cd /tmp && git clone https://github.com/facebookresearch/detectron2.git
cd detectron2 && pip install --no-build-isolation -e .

# 3. Install CLIP (without build isolation)
cd /tmp && git clone https://github.com/openai/CLIP.git
cd CLIP && pip install --no-build-isolation -e .

# 4. Install additional dependencies
pip install torch_dct kornia wandb pytorch-lightning

# 5. Install Segment Anything Model
pip install git+https://github.com/facebookresearch/segment-anything.git

# 6. Install OpenAI API (use version 0.28 for compatibility)
pip install openai==0.28
```

### SoM Checkpoints

Download the SAM checkpoint:
```bash
cd code/SoM
# Download sam_vit_h_4b8939.pth to checkpoints/
mkdir -p ../checkpoints
# Place sam_vit_h_4b8939.pth in ../checkpoints/
```

### Environment Variables

Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

Optional: Set custom API base URL:
```bash
export OPENAI_API_BASE="https://your-custom-endpoint.com"
```

## Code Fixes and Modifications

The following fixes were applied to make the code functional:

### 1. Fixed Import Errors in main.py

**Issue**: Missing `config_schema` module causing import errors.

**Fix**: Commented out unused imports (lines 16-24):
```python
# import hydra
# from omegaconf import DictConfig
# from config_schema import MainConfig
# from pytorch_lightning import seed_everything
# import wandb
# from omegaconf import OmegaConf
```

### 2. Fixed utils.py Configuration Dependencies

**Issue**: Functions requiring `MainConfig` type that doesn't exist.

**Fix**: Commented out config-dependent functions (lines 57-150):
- `hash_training_config()`
- `setup_wandb()`
- `get_output_paths()`

**Fix**: Added tensor handling in `RandomPointConstrainedCrop._point_to_pixel()`:
```python
def _point_to_pixel(self, H: int, W: int) -> Tuple[int, int]:
    x = (self.norm_coord[0] + 1) * 0.5 * W
    y = (self.norm_coord[1] + 1) * 0.5 * H
    # Convert to scalar if tensor
    if isinstance(x, torch.Tensor):
        x = x.item()
    if isinstance(y, torch.Tensor):
        y = y.item()
    return int(round(x)), int(round(y))
```

### 3. Fixed Variable Initialization in attacks.py

**Issue**: `init_patch` used before assignment in both `pgd()` and `mifgsm()`.

**Fix**: Moved initialization before print statements (lines 81-85, 143-147):
```python
set_environment(42)
init_patch = image_tensor.clone().detach().to(device)  # Initialize first
print("patch:", torch.min(init_patch), torch.max(init_patch), init_patch.shape)
```

**Issue**: `loss` variable used before initialization.

**Fix**: Changed `loss += loss_a` to `loss = loss_a` (lines 114, 176):
```python
loss = ensemble_loss(features, features_local, full)  # Direct assignment instead of +=
```

### 4. Fixed dyn_mask.py Import Error

**Issue**: Missing `constants` module.

**Fix**: Commented out unused import (line 11):
```python
# from constants import Const, N
```

### 5. Fixed models/gpt_model.py API Configuration

**Issue**: Hardcoded empty API key and base URL.

**Fix**: Use environment variables (lines 7-14):
```python
import os

class Gpt_model:
    def __init__(self, model_name):
        self.model_name = model_name
        openai.api_key = os.environ.get("OPENAI_API_KEY", "")
        # Don't set api_base if not provided - use default OpenAI endpoint
        if os.environ.get("OPENAI_API_BASE"):
            openai.api_base = os.environ.get("OPENAI_API_BASE")
```

### 6. Fixed Coordinate Indexing in main.py

**Issue**: `RandomPointConstrainedCrop` expects single coordinate pair, but receives batch tensor.

**Fix**: Extract first element from batch (line 172):
```python
source_crop = RandomPointConstrainedCrop(input_res, scale=crop_scale, norm_coord=center[0])
```

## Common Issues and Troubleshooting

### Issue: "No module named 'detectron2'"
**Solution**: Install detectron2 without build isolation:
```bash
cd /tmp && git clone https://github.com/facebookresearch/detectron2.git
cd detectron2 && pip install --no-build-isolation -e .
```

### Issue: "No module named 'clip'"
**Solution**: Install CLIP without build isolation:
```bash
cd /tmp && git clone https://github.com/openai/CLIP.git
cd CLIP && pip install --no-build-isolation -e .
```

### Issue: "Invalid URL '/chat/completions': No scheme supplied"
**Solution**: Ensure `OPENAI_API_KEY` environment variable is set and don't set `OPENAI_API_BASE` unless using custom endpoint.

### Issue: "cannot access local variable 'init_patch' where it is not associated with a value"
**Solution**: This has been fixed in attacks.py. If you see this, ensure init_patch is initialized before use.

### Issue: "IndexError: index 1 is out of bounds for dimension 0 with size 1"
**Solution**: This has been fixed in main.py by using `center[0]` instead of `center`.

## Performance and Runtime Estimates

### Single Image Attack (300 iterations)
- **Time**: ~6-7 minutes
- **GPU**: CUDA required

### 100 Images Full Pipeline
- **SoM Processing**: ~30-60 minutes
- **Coordinate Generation (GPT-4V)**: ~10-15 minutes (100 API calls)
- **Adversarial Attack**: ~10-12 hours (300 iterations × 100 images)
- **Evaluation (GPT-4o)**: ~15-20 minutes (100 API calls)
- **Total**: ~11-14 hours

### API Usage Estimates
- **Coordinate Generation**: 100 GPT-4V API calls
- **Evaluation**: 100 GPT-4o API calls
- Ensure sufficient API credits before running full pipeline

## Example Workflow

### Quick Test (5 samples)
```bash
# 1. Generate SoM masks
cd SoM
python batch_som.py --input_dir ../nuscene/samples_test --output_dir ../som_test \
  --label_dir ../sam_label_test --sam_ckpt ../checkpoints/sam_vit_h_4b8939.pth \
  --granularity 2.6 --alpha 0.1 --label_mode Number --anno_mode Mask Mark

# 2. Generate coordinates
cd ..
python som_gpt.py --original_folder ./nuscene/samples_test --sam_folder ./som_test \
  --label_folder ./sam_label_test --output_path ./coords_test.txt \
  --api_key $OPENAI_API_KEY --model gpt-4o

# 3. Run attack
python main.py --cle_data_path ./data/clean --tgt_data_path ./data/target \
  --output_dir ./results/pgd_test --txt_path ./coords_test.txt \
  --epsilon 16 --alpha 1.0 --num_iters 300 --num_samples 5 --K 10

# 4. Evaluate
python vlm_response.py --image_dir ./results/pgd_test/samples --output_dir ./results \
  --model gpt-4o --query "Describe the main object in the scene..."

python evaluation.py --file_path ./results/gpt-4o_response.txt --model_name gpt-4o \
  --reference_text "A stop sign is visible" --start 0 --end 5 --api_key $OPENAI_API_KEY
```

### Full Pipeline (100 samples)
Follow the same steps but use full directories and adjust `--num_samples 100`.
