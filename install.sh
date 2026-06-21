#!/usr/bin/env bash
# REALM smart installer — detects your GPU / CUDA driver and installs everything.
#   bash install.sh               # install REALM + all dependencies + vLLM + model weights
#   bash install.sh --no-weights  # skip the ~10 GB checkpoint download
set -e

PY="${PYTHON:-python3}"
PIP="$PY -m pip"

# ── 0. Sanity: Python version ──────────────────────────────────────────────────
PYVER=$($PY -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || echo "0.0")
echo ">> Python $PYVER ($($PY -c 'import sys;print(sys.executable)'))"
$PY -c 'import sys;exit(0 if sys.version_info[:2]>=(3,9) else 1)' || {
  echo "!! Python >= 3.9 required"; exit 1; }
$PIP install -q --upgrade pip wheel setuptools

# ── 1. Detect hardware (GPU, CUDA driver, compute capability) ──────────────────
CUDA_TAG=""; ARCH=""; HAVE_GPU=0
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
  DRV_CUDA=$(nvidia-smi 2>/dev/null | grep -oE "CUDA Version: [0-9]+\.[0-9]+" | grep -oE "[0-9]+\.[0-9]+" | head -1)
  ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
  if [ -n "$DRV_CUDA" ]; then
    HAVE_GPU=1
    echo ">> GPU: ${GPU_NAME:-unknown} | driver CUDA ${DRV_CUDA} | compute capability ${ARCH:-unknown}"
    # Pick the highest PyTorch CUDA wheel <= the driver's CUDA version.
    major=${DRV_CUDA%%.*}; minor=${DRV_CUDA##*.}
    if   [ "$major" -ge 13 ]; then CUDA_TAG="cu130"
    elif [ "$major" -eq 12 ] && [ "$minor" -ge 8 ]; then CUDA_TAG="cu128"
    elif [ "$major" -eq 12 ] && [ "$minor" -ge 6 ]; then CUDA_TAG="cu126"
    elif [ "$major" -eq 12 ] && [ "$minor" -ge 4 ]; then CUDA_TAG="cu124"
    elif [ "$major" -eq 12 ];                       then CUDA_TAG="cu121"
    elif [ "$major" -eq 11 ];                       then CUDA_TAG="cu118"
    fi
  fi
fi
if [ "$HAVE_GPU" -eq 0 ]; then
  echo ">> No NVIDIA GPU detected — installing CPU-only PyTorch"
  CUDA_TAG="cpu"
fi
echo ">> Selected PyTorch build: ${CUDA_TAG}"

# ── 2. PyTorch (matched to the driver) ─────────────────────────────────────────
$PIP install --index-url "https://download.pytorch.org/whl/${CUDA_TAG}" torch torchvision

# ── 3. REALM package + PyPI dependencies ───────────────────────────────────────
echo ">> Installing REALM + core dependencies"
$PIP install -e .

# ── 4. Git-only dependencies (CLIP, Segment Anything) ──────────────────────────
echo ">> Installing OpenAI CLIP + Segment Anything"
$PIP install "git+https://github.com/openai/CLIP.git" \
             "git+https://github.com/facebookresearch/segment-anything.git"

# ── 5. Detectron2 (PhysPatch Set-of-Mark) — build for this GPU's arch ───────────
if [ "$HAVE_GPU" -eq 1 ]; then
  echo ">> Building Detectron2 (arch ${ARCH}) for PhysPatch"
  TORCH_CUDA_ARCH_LIST="${ARCH}" $PIP install --no-build-isolation \
    "git+https://github.com/facebookresearch/detectron2.git" \
    || echo "!! Detectron2 build failed (affects PhysPatch only) — see https://detectron2.readthedocs.io"
else
  echo ">> Skipping Detectron2 (no GPU; PhysPatch needs a GPU)"
fi

# ── 6. vLLM (Evaluate stage + figstep/promptinject injection text) ─────────────
if [ "$HAVE_GPU" -eq 1 ]; then
  echo ">> Installing vLLM (matched to ${CUDA_TAG})"
  case "$CUDA_TAG" in
    cu118|cu121|cu124) $PIP install "vllm==0.8.5.post1" || $PIP install vllm ;;
    *)                 $PIP install vllm ;;
  esac
  # vLLM may pull a different torch build — restore the driver-matched one.
  $PIP install --index-url "https://download.pytorch.org/whl/${CUDA_TAG}" torch torchvision
fi

# ── 7. Model weights (default; skip with --no-weights) ─────────────────────────
if [ "$1" != "--no-weights" ]; then
  echo ">> Downloading model weights"
  PYTHON="$PY" bash check_and_download_weights.sh
fi

# ── 8. Verify ──────────────────────────────────────────────────────────────────
echo ">> Verifying install ..."
$PY - <<'PYEOF'
import torch
print(f"   torch {torch.__version__} | CUDA available: {torch.cuda.is_available()}"
      + (f" | {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
from vlm_benchmark.attacks.registry import register_all_attacks, AttackRegistry
from vlm_benchmark.defense import DefenseRegistry
register_all_attacks()
print("   REALM package OK — attacks/defenses registered")
PYEOF

echo ""
echo ">> Done. CLIP surrogates auto-download on first run."
