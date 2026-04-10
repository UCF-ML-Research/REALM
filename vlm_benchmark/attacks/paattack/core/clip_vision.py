import torch
import torch.nn as nn
import torch.nn.functional as F


class ClipVisionModel(nn.Module):
    """Wraps an open_clip ViT model to return embeddings, tokens, and attention."""

    def __init__(self, model, normalize):
        super().__init__()
        self.model = model
        self.normalize = normalize
        self._attn_weights = {}
        self._attn_hooks = []

    def _register_attn_hooks(self):
        """Monkey-patch each ResidualAttentionBlock to capture attention weights."""
        self._remove_attn_hooks()
        self._attn_weights = {}
        for i, block in enumerate(self.model.transformer.resblocks):
            original_attention = block.attention
            layer_idx = i

            def make_hook(blk, idx):
                def hooked_attention(q_x, k_x=None, v_x=None, attn_mask=None):
                    k_x = k_x if k_x is not None else q_x
                    v_x = v_x if v_x is not None else q_x
                    if attn_mask is not None:
                        attn_mask = attn_mask.to(q_x.dtype)
                    attn_output, attn_weight = blk.attn(
                        q_x, k_x, v_x,
                        need_weights=True,
                        attn_mask=attn_mask,
                        average_attn_weights=False,
                    )
                    self._attn_weights[idx] = attn_weight
                    return attn_output
                return hooked_attention

            block.attention = make_hook(block, layer_idx)
            self._attn_hooks.append((block, original_attention))

    def _remove_attn_hooks(self):
        """Restore original attention methods."""
        for block, orig_fn in self._attn_hooks:
            block.attention = orig_fn
        self._attn_hooks = []
        self._attn_weights = {}

    def forward(self, vision, output_normalize, tokens=False, attention=False):
        if not tokens:
            feature = self.model(self.normalize(vision))
            if output_normalize:
                feature = F.normalize(feature, dim=-1)
            return feature
        else:
            self.model.output_tokens = True
            if attention:
                self._register_attn_hooks()
                feature, tok = self.model(self.normalize(vision))
                attentions = self._attn_weights
                self._remove_attn_hooks()
                if output_normalize:
                    feature = F.normalize(feature, dim=-1)
                    tok = F.normalize(tok, dim=-1)
                return feature, tok, attentions
            else:
                feature, tok = self.model(self.normalize(vision))
                if output_normalize:
                    feature = F.normalize(feature, dim=-1)
                    tok = F.normalize(tok, dim=-1)
                return feature, tok
