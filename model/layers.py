import math

import torch
from torch import nn


class MultiHeadCausalSelfAttention(nn.Module):
    """Multi-head self-attention with causal and padding-key masks (from scratch)."""

    def __init__(self, d_model, n_heads, dropout, causal=True):
        super().__init__()
        # causal=False is ONLY for the report's mask study (leaks future tokens).
        self.causal = causal
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)  # fused Q, K, V projections
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x, pad_mask=None):
        """x: [B, T, D]; pad_mask: bool [B, T], True = real token, False = PAD key."""
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=-1)
        # [B, T, D] -> [B, H, T, d_head]
        q, k, v = (t.view(B, T, self.n_heads, self.d_head).transpose(1, 2) for t in (q, k, v))

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)  # [B, H, T, T]
        # Causal mask: query i may attend only to keys j <= i.
        if self.causal:
            allowed = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))[None, None]
        else:
            allowed = torch.ones(1, 1, T, T, dtype=torch.bool, device=x.device)
        # Padding key mask: block PAD keys.
        if pad_mask is not None:
            allowed = allowed & pad_mask[:, None, None, :].bool()
        scores = scores.masked_fill(~allowed, float("-inf"))
        # Every row keeps at least BOS (position 0, never PAD), so softmax is NaN-free.
        attn = self.attn_drop(torch.softmax(scores, dim=-1))

        out = (attn @ v).transpose(1, 2).reshape(B, T, D)  # concat heads
        return self.resid_drop(self.out_proj(out))


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))


class TransformerBlock(nn.Module):
    """Pre-LayerNorm block: x + Attn(LN(x)), then x + FFN(LN(x))."""

    def __init__(self, d_model, n_heads, d_ff, dropout, causal=True):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadCausalSelfAttention(d_model, n_heads, dropout, causal)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, x, pad_mask=None):
        x = x + self.attn(self.ln1(x), pad_mask)
        x = x + self.ffn(self.ln2(x))
        return x
