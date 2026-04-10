# M-Attack Input Format and Attack Process

## Input Format
- Clean images: `dataset/mattack/images/clean/`
- Target images: `dataset/mattack/images/target/1/` (ImageFolder layout required)
- Pairing rule: **index order**. Clean and target lists are matched by sorted filename order.
- Supported extensions: `.png`, `.jpg`, `.jpeg`, `.JPEG`

## Attack Process (Aligned to Legacy)
1. Resize + center-crop to `input_res` (default 224) with bicubic.
2. Build CLIP ensemble (default `B16`, `B32`, `Laion`).
3. Compute cosine similarity loss (no OT), using optional local crop.
4. Optimize with FGSM / MI-FGSM / PGD (default FGSM), clamp `delta` to `[-epsilon, epsilon]`.
5. Output adversarial image in `[0,1]`.

## Defaults
- `epsilon=16`, `steps=300`, `alpha=1.0`
- `attack_method=fgsm`
- `input_res=224`
- `backbone=["B16","B32","Laion"]`
