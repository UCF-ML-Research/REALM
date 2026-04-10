import torch
import torch.nn.functional as F


class ComputeLossWrapper:
    """Stores original tokens, target prototype tokens, and attention mask.

    Delegates to compute_loss() when called.
    """

    def __init__(self, embedding_orig, tokens_orig, target_proto_tokens, tokens_mask, reduction='mean'):
        self.embedding_orig = embedding_orig
        self.tokens_orig = tokens_orig
        self.target_proto_tokens = target_proto_tokens
        self.tokens_mask = tokens_mask
        self.reduction = reduction

    def __call__(self, embedding, tokens):
        return compute_loss(
            embedding=embedding,
            embedding_orig=self.embedding_orig,
            tokens_mask=self.tokens_mask,
            tokens=tokens,
            tokens_orig=self.tokens_orig,
            target_proto_tokens=self.target_proto_tokens,
            reduction=self.reduction,
        )


def compute_loss(embedding, embedding_orig, tokens_mask, tokens, tokens_orig, target_proto_tokens, reduction='mean'):
    """Weighted cosine similarity loss: push tokens away from originals, toward prototypes."""
    loss_orig = cosine_similarity_loss(out=tokens, targets=tokens_orig, reduction=reduction)
    loss_target = cosine_similarity_loss(out=tokens, targets=target_proto_tokens, reduction=reduction)
    loss = ((loss_orig - loss_target) * tokens_mask).sum()
    return loss


def cosine_similarity_loss(out, targets, reduction='none', eps=1e-8):
    """Per-token 1 - cosine_similarity.

    Args:
        out: (B, L, D)
        targets: (B, L, D) or broadcastable
        reduction: 'none' returns (B, L), 'mean' returns scalar

    Returns:
        Loss tensor.
    """
    out_norm = F.normalize(out, p=2, dim=-1)
    targets_norm = F.normalize(targets, p=2, dim=-1)
    cosine_sim = (out_norm * targets_norm).sum(dim=-1)
    loss = 1.0 - cosine_sim
    if reduction == 'mean':
        return loss.mean()
    return loss
