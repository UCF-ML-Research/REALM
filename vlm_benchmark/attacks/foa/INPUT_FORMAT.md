# FOA Input Format and Legacy Alignment

## Input Images
- Clean images: `dataset/foa/images/clean/` (any subfolders ok; stems used).
- Target images: `dataset/foa/images/target/` must contain a file with the **same stem** as each clean image.
- Strict matching: no fallback to clean image. Missing target raises error.

Supported extensions: `.png`, `.jpg`, `.jpeg`, `.JPEG`.

## Cluster Strategy (Legacy/Paper)
- Adaptive cluster selection: **3 → 5**.
- If GPT similarity at cluster=3 meets threshold (default `0.5`), stop; else rerun at cluster=5.

## Attack Process (Matches Legacy FOAttack.py)
1. Load clean image and matched target image; resize/center-crop to `input_res` (default 224) using bicubic.
2. Build CLIP ensemble (`B16`, `B32`, `Laion`) and OT loss.
3. Run attack (FGSM / MI-FGSM / PGD) to align features to target.
4. Generate GPT descriptions of target and adversarial images.
5. Score similarity using GPT; if below threshold at cluster=3, escalate to cluster=5.

## Config (Defaults)
- `epsilon=16` (pixel scale 0–255)
- `max_iterations=300`
- `alpha=1.0`
- `input_res=224`
- `attack_method=fgsm`
- `backbone=["B16","B32","Laion"]`
- `use_adaptive_cluster=True`
- `llm_description_model=gpt4o`, `llm_scorer_model=gpt-4o`, `llm_similarity_threshold=0.5`

## Loss and Optimization (Legacy Behavior)
- OT loss on global features for all attacks.
- Local crop loss:
  - **FGSM**: uses global + local OT (legacy).
  - **MI-FGSM/PGD**: **global only** (no local OT), matching legacy.
- Updates:
  - FGSM: sign gradient step.
  - MI-FGSM: momentum + sign step.
  - PGD: Adam on delta with L_inf clamp.

## API Keys
- Requires `api_keys.yaml|yml|json` at repo root with legacy keys (e.g., `gpt4o`).
