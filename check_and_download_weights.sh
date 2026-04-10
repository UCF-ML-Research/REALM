#!/usr/bin/env bash
# check_and_download_weights.sh
# Manage all model weights for the Red Teaming VLM benchmark.
#
# Usage:
#   bash check_and_download_weights.sh                # check + download missing
#   bash check_and_download_weights.sh --check        # check only, no changes
#   bash check_and_download_weights.sh --download-all # force re-download everything
#   bash check_and_download_weights.sh --delete       # delete all weights (asks confirmation)

set -uo pipefail

# ─── Args ────────────────────────────────────────────────────────────────────
MODE="default"   # default | check | download-all | delete
for arg in "$@"; do
    case "$arg" in
        --check)        MODE="check" ;;
        --download-all) MODE="download-all" ;;
        --delete)       MODE="delete" ;;
    esac
done

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE="$(cd "$(dirname "$0")" && pwd)"
VLM="${BASE}/vlm_benchmark"

PP_CKPT="${VLM}/attacks/physpatch/assets/checkpoints"
PP_SOM="${VLM}/attacks/physpatch/assets/som"

AD_CKPT="${VLM}/attacks/advdiffvlm/assets/checkpoints"
AD_CLIP="${VLM}/attacks/advdiffvlm/assets/clip_cache"
AD_TORCH="${VLM}/attacks/advdiffvlm/data/torch_cache/hub/checkpoints"

COA_ASSET="${VLM}/attacks/coa/assets"
AA_CKPT="${VLM}/attacks/anyattack/assets/checkpoints"

DP_CKPT="${VLM}/defense/diffpure/assets/pretrained/guided_diffusion"
PAD_SAM="${VLM}/defense/pad/assets/models"

HF_CACHE="${HF_HOME:-${HOME}/.cache/huggingface}/hub"
CLIP_CACHE="${HOME}/.cache/clip"

# ─── Colors ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

# ─── Delete mode ─────────────────────────────────────────────────────────────
if [[ "$MODE" == "delete" ]]; then
    echo -e "${RED}WARNING: This will permanently delete all model weights listed below.${NC}"
    echo ""
    echo "Local weight dirs/files:"
    echo "  ${PP_CKPT}/sam_vit_h_4b8939.pth"
    echo "  ${PP_SOM}/seem_focall_v1.pt"
    echo "  ${PP_SOM}/swinl_only_sam_many2many.pth"
    echo "  ${AD_CKPT}/ldm_cin256-v2.ckpt"
    echo "  ${AD_CLIP}/"
    echo "  ${AD_TORCH}/resnet50-0676ba61.pth"
    echo "  ${COA_ASSET}/conceptual_weights.pt"
    echo "  ${AA_CKPT}/*.pt  (AnyAttack decoders)"
    echo "  ${DP_CKPT}/256x256_diffusion_uncond.pt"
    echo "  ${PAD_SAM}/sam_vit_l_0b3195.pth"
    echo ""
    echo "HF cache dirs:"
    echo "  ${HF_CACHE}/models--meta-llama--Llama-3.2-3B-Instruct/"
    echo "  ${HF_CACHE}/models--Qwen--Qwen2.5-VL-7B-Instruct/"
    echo "  ${HF_CACHE}/models--stabilityai--stable-diffusion-3.5-large/"
    echo "  ${HF_CACHE}/models--laion--CLIP-ViT-g-14-laion2B-s12B-b42K/"
    echo "  ${HF_CACHE}/models--laion--CLIP-ViT-G-14-laion2B-s12B-b42K/"
    echo "  ${HF_CACHE}/models--openai--clip-vit-large-patch14-336/"
    echo "  ${HF_CACHE}/models--openai--clip-vit-base-patch16/"
    echo "  ${HF_CACHE}/models--openai--clip-vit-base-patch32/"
    echo "  ${HF_CACHE}/models--Qwen--Qwen3-VL-8B-Instruct/"
    echo "  ${HF_CACHE}/models--laion--CLIP-ViT-B-32-laion2B-s34B-b79K/"
    echo "  ${HF_CACHE}/models--laion--CLIP-ViT-bigG-14-laion2B-39B-b160k/"
    echo "  ${CLIP_CACHE}/ViT-L-14.pt"
    echo ""
    read -r -p "Type YES to confirm deletion: " confirm
    if [[ "$confirm" != "YES" ]]; then
        echo "Aborted."
        exit 0
    fi
    echo ""

    delete_path() {
        if [[ -e "$1" ]]; then
            rm -rf "$1"
            echo -e "  ${RED}DELETED${NC}  $1"
        else
            echo -e "  ${YELLOW}SKIP${NC}     $1  (not found)"
        fi
    }

    echo "Deleting local weights..."
    delete_path "${PP_CKPT}/sam_vit_h_4b8939.pth"
    delete_path "${PP_SOM}/seem_focall_v1.pt"
    delete_path "${PP_SOM}/swinl_only_sam_many2many.pth"
    delete_path "${AD_CKPT}/ldm_cin256-v2.ckpt"
    delete_path "${AD_CLIP}"
    delete_path "${AD_TORCH}/resnet50-0676ba61.pth"
    delete_path "${COA_ASSET}/conceptual_weights.pt"
    delete_path "${AA_CKPT}"
    delete_path "${DP_CKPT}/256x256_diffusion_uncond.pt"
    delete_path "${PAD_SAM}/sam_vit_l_0b3195.pth"

    echo ""
    echo "Deleting HF cache entries..."
    delete_path "${HF_CACHE}/models--meta-llama--Llama-3.2-3B-Instruct"
    delete_path "${HF_CACHE}/models--Qwen--Qwen2.5-VL-7B-Instruct"
    delete_path "${HF_CACHE}/models--stabilityai--stable-diffusion-3.5-large"
    delete_path "${HF_CACHE}/models--laion--CLIP-ViT-g-14-laion2B-s12B-b42K"
    delete_path "${HF_CACHE}/models--laion--CLIP-ViT-G-14-laion2B-s12B-b42K"
    delete_path "${HF_CACHE}/models--openai--clip-vit-large-patch14-336"
    delete_path "${HF_CACHE}/models--openai--clip-vit-base-patch16"
    delete_path "${HF_CACHE}/models--openai--clip-vit-base-patch32"
    delete_path "${HF_CACHE}/models--Qwen--Qwen3-VL-8B-Instruct"
    delete_path "${HF_CACHE}/models--laion--CLIP-ViT-B-32-laion2B-s34B-b79K"
    delete_path "${HF_CACHE}/models--laion--CLIP-ViT-bigG-14-laion2B-39B-b160k"
    delete_path "${CLIP_CACHE}/ViT-L-14.pt"

    echo ""
    echo -e "${GREEN}Done.${NC}"
    exit 0
fi

# ─── Check / Download helpers ─────────────────────────────────────────────────
MISSING=0

check_file() {
    # $1=label  $2=path  $3=expected_size (display only)
    if [[ -f "$2" ]]; then
        local actual
        actual=$(du -sh "$2" 2>/dev/null | cut -f1)
        echo -e "  ${GREEN}OK${NC}  $1  [${actual}]"
        return 0
    else
        echo -e "  ${RED}MISSING${NC}  $1  [$3]"
        echo "         → $2"
        MISSING=$((MISSING + 1))
        return 1
    fi
}

check_dir() {
    # $1=label  $2=dir  $3=expected_size
    if [[ -d "$2" ]] && [[ -n "$(ls -A "$2" 2>/dev/null)" ]]; then
        local actual
        actual=$(du -sh "$2" 2>/dev/null | cut -f1)
        echo -e "  ${GREEN}OK${NC}  $1  [${actual}]"
        return 0
    else
        echo -e "  ${RED}MISSING${NC}  $1  [$3]"
        echo "         → $2"
        MISSING=$((MISSING + 1))
        return 1
    fi
}

check_hf_cache() {
    # $1=label  $2=slug (e.g. "Qwen--Qwen3-VL-8B-Instruct")  $3=expected_size
    local dir="${HF_CACHE}/models--$2"
    if [[ -d "$dir" ]] && [[ -n "$(ls -A "$dir" 2>/dev/null)" ]]; then
        local actual
        actual=$(du -sh "$dir" 2>/dev/null | cut -f1)
        echo -e "  ${GREEN}OK${NC}  $1  [${actual}]"
        return 0
    else
        echo -e "  ${RED}MISSING${NC}  $1  [$3]"
        echo "         → ${dir}"
        MISSING=$((MISSING + 1))
        return 1
    fi
}

# Run download only if mode allows it; in download-all, always run
should_download() {
    # Returns 0 (true) if we should download, 1 if we should skip
    [[ "$MODE" == "check" ]] && return 1
    return 0
}

download_file() {
    # $1=dest  $2=URL
    mkdir -p "$(dirname "$1")"
    echo -e "  ${CYAN}→${NC} Downloading $(basename "$1") ..."
    wget -q --show-progress -O "$1" "$2"
}

download_hf_local() {
    # $1=HF repo  $2=local dir
    echo -e "  ${CYAN}→${NC} huggingface-cli download $1 → $2"
    huggingface-cli download "$1" --local-dir "$2" --repo-type model
}

download_hf_cache() {
    # $1=HF repo
    echo -e "  ${CYAN}→${NC} huggingface-cli download $1"
    huggingface-cli download "$1" --repo-type model
}

# ─── 1. PhysPatch — SAM / SEEM / Semantic-SAM ────────────────────────────────
echo ""
echo "━━━ PhysPatch — SAM / SEEM / Semantic-SAM checkpoints ━━━━━━━━━━━━━━━━━━"

if [[ "$MODE" == "download-all" ]] || \
   ! check_file "SAM ViT-H" "${PP_CKPT}/sam_vit_h_4b8939.pth" "2.4 GB"; then
    should_download && download_file "${PP_CKPT}/sam_vit_h_4b8939.pth" \
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
fi

if [[ "$MODE" == "download-all" ]] || \
   ! check_file "SEEM focal-l" "${PP_SOM}/seem_focall_v1.pt" "1.3 GB"; then
    should_download && download_file "${PP_SOM}/seem_focall_v1.pt" \
        "https://huggingface.co/xdecoder/SEEM/resolve/main/seem_focall_v1.pt"
fi

if [[ "$MODE" == "download-all" ]] || \
   ! check_file "Semantic-SAM SwinL" "${PP_SOM}/swinl_only_sam_many2many.pth" "855 MB"; then
    should_download && download_file "${PP_SOM}/swinl_only_sam_many2many.pth" \
        "https://github.com/UX-Decoder/Semantic-SAM/releases/download/checkpoint/swinl_only_sam_many2many.pth"
fi

if [[ "$MODE" == "download-all" ]] || \
   ! check_hf_cache "laion/CLIP-ViT-g-14-laion2B-s12B-b42K" "laion--CLIP-ViT-g-14-laion2B-s12B-b42K" "3.6 GB"; then
    should_download && download_hf_cache "laion/CLIP-ViT-g-14-laion2B-s12B-b42K"
fi

# ─── 2. AdvDiffVLM — LDM + CLIP + ResNet50 ───────────────────────────────────
echo ""
echo "━━━ AdvDiffVLM — LDM + CLIP ensemble + ResNet50 ━━━━━━━━━━━━━━━━━━━━━━━━"

if [[ "$MODE" == "download-all" ]] || \
   ! check_file "LDM CIN256-v2 (VQ-f4)" "${AD_CKPT}/ldm_cin256-v2.ckpt" "1.5 GB"; then
    should_download && {
        mkdir -p "${AD_CKPT}"
        echo -e "  ${CYAN}→${NC} Downloading cin256-v2 (VQ-f4) checkpoint ..."
        wget -q --show-progress -O "${AD_CKPT}/ldm_cin256-v2.ckpt" \
            "https://ommer-lab.com/files/latent-diffusion/nitro/cin/model.ckpt"
    }
fi

_ad_clip_ok=1
[[ "$MODE" != "download-all" ]] && {
    for clip_file in RN50.pt RN101.pt ViT-B-16.pt ViT-B-32.pt ViT-L-14.pt; do
        check_file "CLIP ${clip_file}" "${AD_CLIP}/${clip_file}" \
            "$(python3 -c "d={'RN50.pt':'244 MB','RN101.pt':'279 MB','ViT-B-16.pt':'335 MB','ViT-B-32.pt':'338 MB','ViT-L-14.pt':'890 MB'}; print(d.get('${clip_file}','?'))" 2>/dev/null || echo '?')" \
            || _ad_clip_ok=0
    done
}

if [[ "$MODE" == "download-all" ]] || [[ $_ad_clip_ok -eq 0 ]]; then
    should_download && bash -c "python - <<'PY'
import clip, shutil
from pathlib import Path
cache = Path('${AD_CLIP}')
cache.mkdir(parents=True, exist_ok=True)
for name, stem in [('RN50','RN50'),('RN101','RN101'),
                   ('ViT-B/16','ViT-B-16'),('ViT-B/32','ViT-B-32'),('ViT-L/14','ViT-L-14')]:
    dest = cache / f'{stem}.pt'
    print(f'  Downloading CLIP {name} ...')
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = clip.clip._download(clip.clip._MODELS[name], tmp)
        shutil.copy(path, dest)
print('Done.')
PY"
fi

if [[ "$MODE" == "download-all" ]] || \
   ! check_file "ResNet50 (torchvision)" "${AD_TORCH}/resnet50-0676ba61.pth" "98 MB"; then
    should_download && bash -c "python - <<'PY'
import torchvision.models as m, os
os.environ['TORCH_HOME'] = '${VLM}/attacks/advdiffvlm/data/torch_cache'
m.resnet50(pretrained=True)
print('Done.')
PY"
fi

# ─── 3. COA — ClipCap conceptual weights ─────────────────────────────────────
echo ""
echo "━━━ COA — ClipCap conceptual weights ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [[ "$MODE" == "download-all" ]] || \
   ! check_file "ClipCap conceptual_weights.pt" "${COA_ASSET}/conceptual_weights.pt" "607 MB"; then
    if should_download; then
        if command -v gdown &>/dev/null; then
            echo -e "  ${CYAN}→${NC} gdown detected — downloading from Google Drive ..."
            mkdir -p "${COA_ASSET}"
            gdown "14pXWwB4Zm82rsDdvbGguLfx9F8aM7ovT" -O "${COA_ASSET}/conceptual_weights.pt"
        else
            echo -e "  ${YELLOW}ACTION REQUIRED${NC}: gdown not installed. Install with: pip install gdown"
            echo "  Then run: gdown 14pXWwB4Zm82rsDdvbGguLfx9F8aM7ovT -O ${COA_ASSET}/conceptual_weights.pt"
            echo "  Or download manually: https://drive.google.com/file/d/14pXWwB4Zm82rsDdvbGguLfx9F8aM7ovT/view"
        fi
    fi
fi

# ─── 4. AnyAttack — pre-trained decoders (HuggingFace) ───────────────────────
echo ""
echo "━━━ AnyAttack — decoder checkpoints ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

_aa_ok=1
[[ "$MODE" != "download-all" ]] && {
    for aa_file in coco_bi.pt coco_cos.pt flickr30k_bi.pt flickr30k_cos.pt pre-trained.pt snli_ve_cos.pt; do
        check_file "AnyAttack ${aa_file}" "${AA_CKPT}/${aa_file}" "320 MB" || _aa_ok=0
    done
}

if [[ "$MODE" == "download-all" ]] || [[ $_aa_ok -eq 0 ]]; then
    should_download && {
        mkdir -p "${AA_CKPT}"
        echo -e "  ${CYAN}→${NC} Downloading AnyAttack decoders from HuggingFace ..."
        for aa_file in coco_bi.pt coco_cos.pt flickr30k_bi.pt flickr30k_cos.pt pre-trained.pt snli_ve_cos.pt; do
            if [[ ! -f "${AA_CKPT}/${aa_file}" ]] || [[ "$MODE" == "download-all" ]]; then
                wget -q --show-progress -O "${AA_CKPT}/${aa_file}" \
                    "https://huggingface.co/Jiaming94/anyattack/resolve/main/${aa_file}"
            fi
        done
    }
fi

# ─── 5. Defense — DiffPure ────────────────────────────────────────────────────
echo ""
echo "━━━ Defense — DiffPure ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [[ "$MODE" == "download-all" ]] || \
   ! check_file "256x256_diffusion_uncond.pt" "${DP_CKPT}/256x256_diffusion_uncond.pt" "2.1 GB"; then
    should_download && download_file "${DP_CKPT}/256x256_diffusion_uncond.pt" \
        "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt"
fi

# ─── 6. Defense — PAD ────────────────────────────────────────────────────────
echo ""
echo "━━━ Defense — PAD (Segment Anything) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [[ "$MODE" == "download-all" ]] || \
   ! check_file "SAM ViT-L" "${PAD_SAM}/sam_vit_l_0b3195.pth" "1.2 GB"; then
    should_download && download_file "${PAD_SAM}/sam_vit_l_0b3195.pth" \
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
fi

# ─── 7. FOA / MAttack / MAttackV2 — HuggingFace CLIP surrogates (HF cache) ──
echo ""
echo "━━━ FOA / MAttack / MAttackV2 — CLIP surrogates (HF cache) ━━━━━━━━━━━━"

_hf_entries=(
    "laion/CLIP-ViT-G-14-laion2B-s12B-b42K|laion--CLIP-ViT-G-14-laion2B-s12B-b42K|5.1 GB"
    "openai/clip-vit-large-patch14-336|openai--clip-vit-large-patch14-336|3.2 GB"
    "openai/clip-vit-base-patch16|openai--clip-vit-base-patch16|1.2 GB"
    "openai/clip-vit-base-patch32|openai--clip-vit-base-patch32|1.2 GB"
    "laion/CLIP-ViT-B-32-laion2B-s34B-b79K|laion--CLIP-ViT-B-32-laion2B-s34B-b79K|1.2 GB"
    "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k|laion--CLIP-ViT-bigG-14-laion2B-39B-b160k|10 GB"
)

for entry in "${_hf_entries[@]}"; do
    IFS='|' read -r repo slug size <<< "$entry"
    if [[ "$MODE" == "download-all" ]] || ! check_hf_cache "$repo" "$slug" "$size"; then
        should_download && download_hf_cache "$repo"
    fi
done

# ─── 8. Eval Server — Qwen3-VL-8B-Instruct ───────────────────────────────────
echo ""
echo "━━━ Eval Server — Qwen3-VL-8B-Instruct (HF cache) ━━━━━━━━━━━━━━━━━━━━━"

if [[ "$MODE" == "download-all" ]] || \
   ! check_hf_cache "Qwen/Qwen3-VL-8B-Instruct" "Qwen--Qwen3-VL-8B-Instruct" "17 GB"; then
    should_download && download_hf_cache "Qwen/Qwen3-VL-8B-Instruct"
fi

# ─── 9. PA-Attack — open_clip ViT-L-14 (openai) ─────────────────────────────
echo ""
echo "━━━ PA-Attack — open_clip ViT-L-14 (openai) ━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [[ "$MODE" == "download-all" ]] || \
   ! check_file "CLIP ViT-L-14 (open_clip/openai)" "${CLIP_CACHE}/ViT-L-14.pt" "890 MB"; then
    should_download && bash -c "python3 - <<'PY'
import open_clip
print('  Downloading ViT-L-14 (openai) via open_clip ...')
model, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
del model
print('  Done.')
PY"
fi

# Prototypes are bundled at vlm_benchmark/attacks/paattack/prototypes/ (no download needed)
_pa_proto="${VLM}/attacks/paattack/prototypes/prototypes_tokens_3000_20_1024.pt"
check_file "PA-Attack prototypes" "${_pa_proto}" "20 MB"

# ─── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
case "$MODE" in
    check)
        if [[ $MISSING -eq 0 ]]; then
            echo -e "${GREEN}All weights present.${NC}"
        else
            echo -e "${RED}${MISSING} weight(s) missing.${NC} Re-run without --check to download."
        fi
        ;;
    download-all)
        echo -e "${GREEN}Download-all complete.${NC}"
        ;;
    *)
        if [[ $MISSING -eq 0 ]]; then
            echo -e "${GREEN}All weights present.${NC}"
        else
            echo -e "${GREEN}Download complete.${NC} (${MISSING} item(s) were missing before this run.)"
        fi
        ;;
esac
echo ""
