<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/icon/realm-white.png">
  <img src="docs/icon/realm.png" alt="REALM" width="100">
</picture>

# REALM

**A Unified Red Teaming Benchmark for Vision-Language Models in the Physical World**

</div>

## Overview

REALM is a red-teaming framework for evaluating adversarial robustness of Vision-Language Models (VLMs) deployed in safety-critical physical-world domains — autonomous driving, robotic manipulation, and embodied AI. It provides 13 attack methods, 4 defenses, and an automated evaluation pipeline.

All attacks are **black-box** against the victim VLM — optimized on CLIP surrogates and transferred to closed-source models (GPT-4o, Claude, etc.), reflecting realistic threat models.

## Features

- **13 Attack Methods**: Gradient-based (single/ensemble surrogate), attention-guided, diffusion-based, multimodal injection, natural corruption, and non-gradient attacks
- **4 Defenses**: Patch detection, diffusion purification, frequency-domain filtering, and multi-modal purification
- **Modular Architecture**: Plugin-based registries — add a new attack by implementing `BaseAttack` and registering it
- **Dual Metrics**: ASR (Attack Success Rate) and MR (Misclassification Rate) with per-category breakdown

## Quick Start

### Installation

```bash
git clone https://github.com/UCF-ML-Research/REALM.git
cd REALM
pip install -e .
```

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

# Natural corruption baseline
python scripts/generate_adversarial.py corruption \
    --dataset nips2017 --corruption_mode fog --corruption_severity 3 \
    -o dataset/nips2017/adversarial/corruption_fog
```

### Evaluate

```bash
# Start VLM server
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-8B-Instruct --port 8000

# Start LLM extractor server (for MCQ answer extraction)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-8B --port 8002

# Evaluate
python agent/adversarial/evaluate.py \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --dataset pai_bench \
    --server_url http://localhost:8000 \
    --attack_dirs dataset/pai_bench_red_teaming/foa dataset/pai_bench_red_teaming/mattack \
    --extractor_model Qwen/Qwen3-8B \
    --extractor_url http://localhost:8002 \
    --output_dir eval_results/pai_bench/Qwen3-VL-8B-Instruct

# Or run full multi-model evaluation
bash scripts/run_eval_all.sh
```

Outputs per-attack **ASR** (response matches attack target) and **MR** (response differs from correct answer).

### Apply Defenses

```bash
python scripts/clean_adversarial.py \
    --defense freqpure --adversarial_images dataset/pai_bench_red_teaming/foa \
    --output_dir results/cleaned
```

### Python API

```python
from vlm_benchmark.attacks import AttackRegistry

attack = AttackRegistry.create('foa', epsilon=16, max_iterations=300, device='cuda')
result = attack.generate(model=None, sample=sample)
result.adversarial_sample.save("adversarial.jpg")
```

## Attack Methods

| # | Attack | Category | Surrogate | ε | Speed |
|---|--------|----------|-----------|---|-------|
| 1 | **FOA** | Gradient (OT loss) | 3× CLIP | 16/255 | ~100s/img |
| 2 | **M-Attack** | Gradient (cosine) | 3× CLIP | 16/255 | ~20s/img |
| 3 | **CoA** | Multimodal (CLIP+ClipCap) | CLIP + GPT-2 | 8/255 | ~35s/img |
| 4 | **V-Attack** | Text-guided gradient | 3× CLIP | 16/255 | ~30s/img |
| 5 | **PhysPatch** | Patch-based | 3× CLIP + SAM | 16/255 | ~90s/img |
| 6 | **AdvDiffVLM** | Diffusion (AEGE) | LDM + 4× CLIP | ∞ | ~70s/img |
| 7 | **ADVEDM-A** | Semantic addition | 4× CLIP (SSA-CWA) | 16/255 | ~45s/img |
| 8 | **ADVEDM-R** | Semantic removal | 4× CLIP (SSA-CWA) | 16/255 | ~45s/img |
| 9 | **AnyAttack** | Learned decoder | CLIP + Decoder | 16/255 | <1s/img |
| 10 | **PA-Attack** | Untargeted (OOD proto) | CLIP ViT-L | 4/255 | ~15s/img |
| 11 | **FigStep** | Typographic injection | None | 0 | <1s/img |
| 12 | **PromptInject** | Prompt manipulation | None | 0 | <1s/img |
| 13 | **Corruption** | Natural corruption | None | 0 | <1s/img |

## Defenses

| Defense | Category | Model | Description |
|---------|----------|-------|-------------|
| **PAD** | Patch detection | SAM ViT-L | MI/CD heatmap fusion → SAM segmentation → patch removal |
| **FreqPure** | Frequency filtering | Guided Diffusion | FFT amplitude swap + phase clipping + diffusion denoising |
| **BlueSuffix** | Multi-modal purification | Diffusion + GPT-4o + GPT-2 LoRA | Image denoising + text purification + defensive suffix |
| **SystemPrompt** | Prompt engineering | None | Hardcoded safety system prompt prepend |

## Project Structure

```
Red_Teaming/
├── vlm_benchmark/                 # Core framework
│   ├── attacks/                   # 13 attack implementations
│   │   ├── registry.py            # Attack registry + factory
│   │   ├── base_attack.py         # BaseAttack abstract class
│   │   ├── foa/ mattack/ coa/ physpatch/ paattack/
│   │   ├── advdiffvlm/ advedm/ vattack/ anyattack/
│   │   ├── figstep/ promptinject/
│   │   └── corruption/
│   ├── defense/                   # 4 defense implementations
│   │   ├── pad/ freqpure/ bluesuffix/ systemprompt/
│   │   └── registry.py
│   ├── models/                    # VLM model factory (vLLM, transformers, API)
│   ├── data/                      # Dataset loaders
│   └── evaluation/                # VLM inference + scoring
├── scripts/                       # CLI scripts
│   ├── generate_adversarial.py    # Generate adversarial samples (any attack)
│   ├── evaluate_adversarial.py    # Evaluate ASR + MR
│   ├── clean_adversarial.py       # Apply defenses
│   └── run_eval_all.sh            # Multi-model PAI-bench evaluation
├── dataset/                       # Datasets + generated outputs
│   ├── pai_bench/                 # Source data (images, behaviors, manifest)
│   ├── pai_bench_red_teaming/     # Per-attack adversarial outputs
│   └── nips2017/                  # 100 ImageNet source-target pairs
└── eval_results/                  # Per-model evaluation results
```

## Adding New Attacks

```python
# 1. Implement in vlm_benchmark/attacks/my_attack/my_attack_attack.py
from vlm_benchmark.attacks.base_attack import BaseAttack, AttackConfig, AttackResult

class MyAttackConfig(AttackConfig):
    my_param: float = 1.0

class MyAttack(BaseAttack):
    def __init__(self, config: MyAttackConfig):
        super().__init__(config)

    def generate(self, model, sample, **kwargs) -> AttackResult:
        adversarial_image = self._perturb(sample.images[0])
        return AttackResult(success=True, adversarial_sample=adversarial_image)

# 2. Register in vlm_benchmark/attacks/registry.py
# 3. Add config.py with resolve_cli_kwargs()
```

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.36.0,<5.0
Pillow>=9.0.0
qwen-vl-utils>=0.0.2
open_clip_torch>=2.20.0
openai>=1.3.0
vllm>=0.15.0
```

**GPU**: NVIDIA GPU with ≥16 GB VRAM (24 GB recommended for diffusion attacks or local VLM serving).

## Acknowledgements

This project integrates adversarial attack methods proposed by prior research. We thank the original authors for making their work publicly available.
