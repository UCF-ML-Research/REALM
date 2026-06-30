<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/icon/realm-white.png">
  <img src="docs/icon/realm.png" alt="REALM" width="100">
</picture>

# ReaLM: A Unified Red-Teaming Benchmark for Physical-World VLMs

</div>

## Overview

REALM is a red-teaming **attack-and-defense library** for evaluating the adversarial robustness of Vision-Language Models (VLMs) deployed in safety-critical physical-world tasks — object detection and embodied AI. It provides **12 attack methods**, **3 model-agnostic defenses**, and an automated evaluation pipeline that **measures both attack and defense effectiveness**. All attacks are **black-box** against the victim VLM — optimized on CLIP surrogates and transferred to a range of proprietary and open-source models, reflecting realistic threat models.

## Key Features

- 🎯 **12 attack methods** — gradient-based visual perturbations, localized patches, diffusion generation and semantic editing, and prompt/typographic injection, plus a non-adversarial baseline.
- 🛡️ **3 model-agnostic defenses** — PAD, FreqPure, and BlueSuffix, applied as input preprocessing with no model retraining.
- 🔒 **Black-box threat model** — attacks are optimized on CLIP surrogates and transferred; no access to the victim's weights, gradients, or logits.
- 🧩 **Extensible** — plugin registries; add a new attack or defense by implementing a single base class and registering it.
- 🔌 **Flexible backends** — evaluate proprietary models through the OpenRouter API or open-source VLMs served locally with vLLM.

## Quick Start

### Installation

```bash
git clone https://github.com/UCF-ML-Research/REALM.git
cd REALM
bash install.sh
```

`install.sh` auto-matches PyTorch to your GPU driver and installs everything — REALM, CLIP, Segment-Anything, Detectron2, vLLM, and model weights. CLIP surrogates auto-download on first run.

### Generate Adversarial Samples

**NIPS 2017 dataset** (100 ImageNet source-target pairs):

```bash
# Gradient-based (CLIP surrogate)
python scripts/generate_adversarial.py foa --dataset nips2017 -o dataset/nips2017/adversarial/foa

# Untargeted
python scripts/generate_adversarial.py paattack --dataset nips2017 -o dataset/nips2017/adversarial/paattack

# Text-guided
python scripts/generate_adversarial.py vattack \
    --dataset nips2017 --labels_file dataset/nips2017/labels.json \
    -o dataset/nips2017/adversarial/vattack

# Typographic injection (with VLM-generated text)
python scripts/generate_adversarial.py figstep \
    --dataset nips2017 --labels_file dataset/nips2017/labels.json \
    --vlm_url http://localhost:8001 --vlm_model Qwen/Qwen3-VL-8B-Instruct \
    -o dataset/nips2017/adversarial/figstep

# Prompt manipulation
python scripts/generate_adversarial.py promptinject \
    --dataset nips2017 --labels_file dataset/nips2017/labels.json \
    --question "What is the main object in this image?" \
    --vlm_url http://localhost:8001 --vlm_model Qwen/Qwen3-VL-8B-Instruct \
    -o dataset/nips2017/adversarial/promptinject
```

### Evaluate

Score an attack's outputs with a VLM judge. Start a server for the victim model first (or pass `--server_url`):

```bash
# Serve the victim VLM (OpenAI-compatible endpoint)
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-VL-8B-Instruct --port 8000

# Evaluate
python scripts/evaluate_adversarial.py \
    --attack_dirs dataset/nips2017/adversarial/foa dataset/nips2017/cleaned/foa \
    --question "What is the main object in this image?" \
    --labels_json dataset/nips2017/labels.json \
    --judge vlm --model Qwen/Qwen3-VL-8B-Instruct \
    --server_url http://localhost:8000 \
    -o results/eval/foa
```

Reports **Attack Success Rate (ASR)** for each input set against a clean baseline, measuring both **attack effectiveness** (ASR on adversarial inputs) and **defense effectiveness** (the drop in ASR once a defense is applied). Pass any mix of adversarial and defended (cleaned) directories to `--attack_dirs` to compare them in one table.

### Apply Defenses

```bash
python scripts/clean_adversarial.py \
    --defense freqpure --adversarial_images dataset/nips2017/adversarial/foa \
    -o dataset/nips2017/cleaned/foa
```

### Python API

```python
from vlm_benchmark.attacks import AttackRegistry

attack = AttackRegistry.create('foa', epsilon=16, max_iterations=300, device='cuda')
result = attack.generate(model=None, sample=sample)
result.adversarial_sample.save("adversarial.jpg")
```

## Attacks

| Attack | Category | ε |
|--------|----------|---|
| FOA | Perturbation (optimal transport) | 16/255 |
| M-Attack | Perturbation (cosine) | 16/255 |
| V-Attack | Perturbation (value features) | 16/255 |
| CoA | Perturbation (multimodal co-optimization) | 8/255 |
| PhysPatch | Perturbation (patch) | 16/255 |
| PA-Attack | Perturbation (untargeted, OOD prototypes) | 8/255 |
| AdvDiffVLM | Generation (diffusion latent) | ∞ |
| AdvEDM | Editing (semantic) | 8/255 |
| AnyAttack | Generation (single forward pass) | 16/255 |
| FigStep | Injection (typographic) | — |
| PromptInject | Injection (prompt) | — |
| ImageMix | Injection (alpha-blend baseline) | — |

## Defenses

| Defense | Type | Description |
|---------|------|-------------|
| PAD | Patch removal | MI/CD heatmap fusion → SAM segmentation → patch removal |
| FreqPure | Frequency purification | FFT amplitude swap + phase clipping + diffusion denoising |
| BlueSuffix | Multimodal purification | Image denoising + text purification + defensive suffix |

## Acknowledgements

This project integrates adversarial attack methods proposed by prior research. We thank the original authors for making their work publicly available.

## Citation

If you find our repo useful, please cite our paper:

```bibtex
@article{zhao2026realm,
  title={REALM: A Unified Red-Teaming Benchmark for Physical-World VLMs},
  author={Zhao, Yifei and Lou, Qian and Zheng, Mengxin},
  journal={arXiv preprint arXiv:2606.23892},
  year={2026}
}
```
