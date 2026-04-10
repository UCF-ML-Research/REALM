#!/usr/bin/env python3
"""LEAF: Adversarial fine-tuning of the CLIP text encoder.

Implements the LEAF loss from Schlarmann et al., NeurIPS 2025:

  L = E_t [ || g_θ(t_adv) - g_orig(t) ||^2 ]

where t_adv is found by a character-level Levenshtein-distance-bounded
discrete attack (insert/delete/replace characters to maximize embedding shift).

Usage:
    python -m vlm_benchmark.defense.robustclip.core.train_leaf \
        --data_dir /path/to/datacomp-small-shards/ \
        --output_dir vlm_benchmark/defense/robustclip/checkpoints/leaf \
        --pretrained hf-hub:chs20/fare2-clip \
        --device cuda:0

    # Or using HuggingFace datasets:
    python -m vlm_benchmark.defense.robustclip.core.train_leaf \
        --hf_dataset mlfoundations/datacomp_small \
        --output_dir vlm_benchmark/defense/robustclip/checkpoints/leaf \
        --device cuda:0
"""

import argparse
import math
import random
import string
import time
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from tqdm import tqdm

# Character vocabulary for LEAF attacks (matches official: ascii_lowercase + space +
# ascii_uppercase + digits + punctuation, plus -1 as deletion sentinel)
_VOCAB = [-1] + [ord(c) for c in
    string.ascii_lowercase + " " + string.ascii_uppercase + string.digits + string.punctuation
]


def _apply_edit(text: str, pos: int, char_ord: int) -> str:
    """Apply a single character edit at position `pos` in the expanded string.

    In the expanded representation, even positions (0, 2, 4, ...) are insertion
    slots between characters, and odd positions (1, 3, 5, ...) correspond to
    existing character indices (pos // 2).

    char_ord=-1 means deletion (only valid at odd positions) or no-op (even).
    """
    if pos % 2 == 0:
        # Insertion slot
        if char_ord == -1:
            return text  # no-op
        insert_idx = pos // 2
        return text[:insert_idx] + chr(char_ord) + text[insert_idx:]
    else:
        # Character position
        char_idx = pos // 2
        if char_idx >= len(text):
            return text
        if char_ord == -1:
            return text[:char_idx] + text[char_idx + 1:]  # deletion
        return text[:char_idx] + chr(char_ord) + text[char_idx + 1:]  # replacement


def _has_new_dict_word(original: str, candidate: str, dict_words: set) -> bool:
    """Check if candidate introduces a new dictionary word not in original."""
    new_words = set(candidate.lower().split()) - set(original.lower().split())
    return any(w in dict_words for w in new_words)


def leaf_attack(
    model_fn,
    anchor_emb,
    texts: List[str],
    tokenizer_fn,
    k: int = 1,
    rho: int = 50,
    constrain: bool = True,
):
    """Two-phase batched character-level Levenshtein attack (matches official LEAF).

    Phase 1: For ALL sentences, sample rho positions, probe with space, pick best
             position per sentence — ONE batched forward pass of B*rho texts.
    Phase 2: At each sentence's best position, sample rho chars, pick best
             character per sentence — ONE batched forward pass of B*rho texts.

    Repeated k times (Levenshtein distance budget).

    Args:
        model_fn:     callable(token_ids) → [B, D] embeddings
        anchor_emb:   [B, D] frozen clean embeddings
        texts:        list of B strings
        tokenizer_fn: callable(list[str]) → token_ids tensor
        k:            Levenshtein distance budget (iterations)
        rho:          number of random candidates per phase
        constrain:    if True, reject edits that introduce new dictionary words

    Returns:
        adv_texts: list of B adversarial strings
    """
    sentences = list(texts)
    batch_size = len(texts)

    # Load dictionary for constraint checking
    dict_words = None
    if constrain:
        try:
            import nltk
            try:
                dict_words = set(nltk.corpus.words.words())
            except LookupError:
                nltk.download("words", quiet=True)
                dict_words = set(nltk.corpus.words.words())
            dict_words = {w.lower() for w in dict_words}
        except ImportError:
            dict_words = None

    for _step in range(k):
        # ── Phase 1: Position selection (batched, matches official) ───────
        # For each sentence, sample rho positions and probe with space char.
        all_candidates = []  # flat list: B * rho strings
        positions_per_sent = []  # list of B arrays, each of length rho

        for text in sentences:
            n_pos = 2 * len(text) + 1
            replace = rho > n_pos
            pos_array = random.choices(range(n_pos), k=rho) if replace else random.sample(range(n_pos), rho)
            positions_per_sent.append(pos_array)

            for pos in pos_array:
                candidate = _apply_edit(text, pos, ord(" "))
                # If constrained and invalid, fall back to original (matches official)
                if constrain and dict_words and candidate and _has_new_dict_word(text, candidate, dict_words):
                    candidate = text
                if not candidate:
                    candidate = text
                all_candidates.append(candidate)

        # One batched forward pass: [B*rho] → [B, rho, D]
        tokens = tokenizer_fn(all_candidates)
        with torch.no_grad():
            text_features = model_fn(tokens).view(batch_size, rho, -1)

        # L2 loss per candidate: [B, rho]
        loss = ((text_features - anchor_emb.unsqueeze(1)) ** 2).sum(dim=-1)
        ids_best = torch.argmax(loss, dim=-1)  # [B]

        # Extract best position per sentence
        best_pos = []
        for i, idx in enumerate(ids_best.tolist()):
            best_pos.append(positions_per_sent[i][idx])

        # ── Phase 2: Character selection at best position (batched) ───────
        all_candidates = []
        for i, text in enumerate(sentences):
            char_samples = random.sample(_VOCAB, min(rho, len(_VOCAB)))
            # Pad to exactly rho candidates if needed
            while len(char_samples) < rho:
                char_samples.append(-1)

            for char_ord in char_samples:
                candidate = _apply_edit(text, best_pos[i], char_ord)
                if constrain and dict_words and candidate and _has_new_dict_word(text, candidate, dict_words):
                    candidate = text
                if not candidate:
                    candidate = text
                all_candidates.append(candidate)

        tokens = tokenizer_fn(all_candidates)
        with torch.no_grad():
            text_features = model_fn(tokens).view(batch_size, rho, -1)

        loss = ((text_features - anchor_emb.unsqueeze(1)) ** 2).sum(dim=-1)
        ids_best = torch.argmax(loss, dim=-1)  # [B]

        # Update sentences
        sentences = [
            all_candidates[i * rho + idx]
            for i, idx in enumerate(ids_best.tolist())
        ]

    return sentences


def leaf_loss(trainable_emb, anchor_emb):
    """LEAF outer loss: L2 embedding matching for text encoder.

    sum over embedding dim, mean over batch — matches official.
    """
    return F.mse_loss(trainable_emb, anchor_emb, reduction="none").sum(dim=-1).mean()


_cached_texts = None  # module-level cache to avoid re-downloading


def load_texts(data_dir=None, hf_dataset=None, max_samples=80_000) -> List[str]:
    """Load all captions into memory (cached across epochs)."""
    global _cached_texts
    if _cached_texts is not None:
        return _cached_texts

    if hf_dataset:
        from datasets import load_dataset

        print(f"Loading text data from HuggingFace: {hf_dataset}")
        ds = load_dataset(hf_dataset, split="train", streaming=True)
        texts = []
        for sample in ds:
            caption = sample.get("text") or sample.get("caption") or ""
            if caption:
                texts.append(caption)
            if len(texts) >= max_samples:
                break
    elif data_dir:
        import webdataset as wds

        data_path = str(Path(data_dir) / "*.tar")
        print(f"Loading text data from webdataset: {data_path}")
        texts = []
        dataset = (
            wds.WebDataset(data_path)
            .decode("pil")
            .to_tuple("txt")
        )
        for (caption,) in dataset:
            texts.append(caption)
            if len(texts) >= max_samples:
                break
    else:
        raise ValueError("Must specify either --data_dir or --hf_dataset")

    print(f"Loaded {len(texts)} captions")
    _cached_texts = texts
    return texts


def batch_iter(texts: List[str], batch_size: int):
    """Yield batches of strings from a flat list."""
    for i in range(0, len(texts), batch_size):
        yield texts[i : i + batch_size]


def _build_optimizer(model, lr, weight_decay, betas=(0.9, 0.98), eps=1e-6):
    """Build AdamW optimizer with two param groups (matches official LEAF).

    Bias, LayerNorm, and logit_scale params are exempt from weight decay.
    """
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "bn" in name or "ln" in name or "bias" in name or "logit_scale" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=betas,
        eps=eps,
    )


def train_leaf(args):
    """Main LEAF training loop."""
    import open_clip

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load models ───────────────────────────────────────────────────────
    print(f"Loading CLIP {args.clip_model_name} (pretrained={args.pretrained})")
    model, _, _ = open_clip.create_model_and_transforms(
        args.clip_model_name, pretrained=args.pretrained,
    )
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer(args.clip_model_name)

    # Frozen original model for anchor embeddings
    orig_model, _, _ = open_clip.create_model_and_transforms(
        args.clip_model_name, pretrained=args.pretrained,
    )
    orig_model = orig_model.to(device).eval()
    for p in orig_model.parameters():
        p.requires_grad_(False)

    # Freeze vision encoder — only train text
    for p in model.parameters():
        p.requires_grad_(False)
    # Unfreeze text encoder (everything except visual)
    for name, p in model.named_parameters():
        if "visual" not in name:
            p.requires_grad_(True)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params (text encoder): {sum(p.numel() for p in trainable_params):,}")

    # ── Optimizer + scheduler (matches official: two param groups) ────────
    optimizer = _build_optimizer(model, args.lr, args.weight_decay)

    # Load all texts once (cached for subsequent epochs)
    all_texts = load_texts(
        data_dir=args.data_dir,
        hf_dataset=args.hf_dataset,
        max_samples=args.data_samples,
    )

    est_batches_per_epoch = len(all_texts) // args.batch_size
    total_steps = args.epochs * est_batches_per_epoch
    warmup_steps = args.warmup_steps

    def lr_schedule(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # ── Training ──────────────────────────────────────────────────────────
    print(f"\nLEAF Training:")
    print(f"  k_adv:       {args.k_adv} (Levenshtein budget)")
    print(f"  rho:         {args.rho} (candidates per phase)")
    print(f"  constrain:   {args.constrain}")
    print(f"  LR:          {args.lr}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  Total steps: {total_steps}")
    print(f"  Output:      {output_dir}\n")

    scaler = torch.amp.GradScaler("cuda")
    global_step = 0
    t0 = time.time()

    def tokenize_fn(texts):
        return tokenizer(texts).to(device)

    def text_encode_fn(tokens):
        return model.encode_text(tokens, normalize=False)

    for epoch in range(args.epochs):
        model.train()
        # Re-freeze vision just in case
        for p in model.visual.parameters():
            p.requires_grad_(False)

        epoch_loss = 0.0
        n_batches = 0

        # Shuffle texts each epoch
        shuffled = list(all_texts)
        random.shuffle(shuffled)

        for batch_txts in tqdm(batch_iter(shuffled, args.batch_size),
                               total=est_batches_per_epoch,
                               desc=f"Epoch {epoch + 1}/{args.epochs}"):
            if not batch_txts:
                continue

            # 1. Get anchor embeddings from frozen original model
            tokens_clean = tokenize_fn(batch_txts)
            with torch.no_grad():
                anchor_emb = orig_model.encode_text(tokens_clean, normalize=False)

            # 2. LEAF two-phase attack (model.eval during attack, matches official)
            model.eval()
            adv_texts = leaf_attack(
                text_encode_fn,
                anchor_emb,
                batch_txts,
                tokenize_fn,
                k=args.k_adv,
                rho=args.rho,
                constrain=args.constrain,
            )

            # 3. Outer minimization (model.train for loss computation)
            model.train()
            for p in model.visual.parameters():
                p.requires_grad_(False)

            tokens_adv = tokenize_fn(adv_texts)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                adv_emb = model.encode_text(tokens_adv, normalize=False)
                loss = leaf_loss(adv_emb, anchor_emb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # Clamp logit_scale (matches official)
            with torch.no_grad():
                model.logit_scale.clamp_(0, math.log(100))

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            if global_step % 100 == 0:
                avg = epoch_loss / n_batches
                elapsed = time.time() - t0
                print(f"  step {global_step:>6d}  loss={avg:.4f}  "
                      f"elapsed={elapsed / 60:.1f}min")

        avg_loss = epoch_loss / max(1, n_batches)
        print(f"Epoch {epoch + 1}: avg_loss={avg_loss:.4f}")

        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            ckpt_path = output_dir / f"leaf_epoch{epoch + 1}.pt"
            # Save full model state_dict (matches official — both visual + text)
            torch.save({
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, ckpt_path)
            print(f"  Saved: {ckpt_path}")

    # Final save
    final_path = output_dir / "leaf_final.pt"
    torch.save({
        "epoch": args.epochs,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, final_path)
    print(f"\nTraining complete. Final checkpoint: {final_path}")
    print(f"Total time: {(time.time() - t0) / 60:.1f} minutes")


def main():
    from ..config import LEAF_TRAIN_DEFAULTS as D

    parser = argparse.ArgumentParser(description="LEAF: adversarial fine-tuning of CLIP text encoder")
    parser.add_argument("--data_dir", default=None, help="Path to webdataset shards (DataComp-small)")
    parser.add_argument("--hf_dataset", default=None, help="HuggingFace dataset name")
    parser.add_argument("--output_dir", required=True, help="Output directory for checkpoints")
    parser.add_argument("--clip_model_name", default=D["clip_model_name"])
    parser.add_argument("--pretrained", default=D["pretrained"],
                        help="Starting checkpoint (default: hf-hub:chs20/fare2-clip)")
    parser.add_argument("--epochs", type=int, default=D["epochs"])
    parser.add_argument("--batch_size", type=int, default=D["batch_size"])
    parser.add_argument("--lr", type=float, default=D["lr"])
    parser.add_argument("--weight_decay", type=float, default=D["weight_decay"])
    parser.add_argument("--warmup_steps", type=int, default=D["warmup_steps"])
    parser.add_argument("--k_adv", type=int, default=D["k_adv"],
                        help="Levenshtein distance budget (default: 1)")
    parser.add_argument("--rho", type=int, default=D["rho"],
                        help="Random candidate positions per phase (default: 50)")
    parser.add_argument("--no_constrain", action="store_true",
                        help="Disable dictionary word constraint (default: constrain=True)")
    parser.add_argument("--data_samples", type=int, default=D["data_samples"])
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.constrain = not args.no_constrain
    train_leaf(args)


if __name__ == "__main__":
    main()
