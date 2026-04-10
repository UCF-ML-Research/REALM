# CoA Input Format and Process

## Input Format
- Clean images: `dataset/coa/images/clean/`
- Target images: `dataset/coa/images/target/`
- Target captions: `dataset/coa/captions/target_captions.txt` (one line per target image)
- Clean captions (optional): `dataset/coa/captions/clean_captions_qwen.txt`
- Pairing rule: **sorted filename order**; counts must match for clean images, target images, and target captions.

## Caption Source
- If clean captions are missing or incomplete, **Qwen3‑VL auto‑generates** captions on the fly.

## Attack Process (Core)
1. Resize + center‑crop to `input_res` (default 224) and compute CLIP features.
2. Generate current caption each PGD step via ClipCap (optionally reuse every `caption_update_steps`).
3. Fuse image/text embeddings (`cat`, `add_weight`, or `multiplication`).
4. Optimize PGD with triplet‑style loss:
   `relu(sim(cur,tgt) - p_neg*sim(cur,clean) + (1 - p_neg))`.
5. Clamp `delta` to `[-epsilon, epsilon]` and output adversarial image.
