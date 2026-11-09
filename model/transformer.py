"""Model interface for ROCStories next-token prediction."""

import torch
from torch import nn

if __package__:
    from .layers import TransformerBlock
else:
    from layers import TransformerBlock


class StoryTransformerLM(nn.Module):
    """Autoregressive language model over a fixed token vocabulary."""

    def __init__(
        self,
        vocab_size,
        max_len,
        d_model,
        n_heads,
        n_layers,
        d_ff,
        dropout,
        pad_id,
    ):
        """Args:
            vocab_size: Number of tokens in the vocabulary.
            max_len: Maximum supported sequence length.
            d_model: Model feature dimension; divisible by n_heads.
            n_heads: Number of attention heads per block.
            n_layers: Number of Transformer blocks.
            d_ff: Intermediate feed-forward feature dimension.
            dropout: Dropout probability in [0, 1).
            pad_id: Padding token ID in [0, vocab_size).
        """
        super().__init__()
        self.max_len = max_len
        self.pad_id = pad_id
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, d_model)  # learned positions
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying: shared params count once
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, input_ids):
        """Args:
            input_ids: LongTensor [B, T] with token IDs in [0, vocab_size),
                where 1 <= T <= max_len.

        Returns:
            Raw floating-point logits [B, T, vocab_size], without softmax.
            Position t predicts the token following input_ids[:, t].
            Future tokens and padding keys must not affect valid positions.
        """
        B, T = input_ids.shape
        if not 1 <= T <= self.max_len:
            raise ValueError(f"Sequence length {T} outside [1, {self.max_len}]")
        pad_mask = input_ids != self.pad_id  # True = real token
        pos = torch.arange(T, device=input_ids.device)
        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(pos)[None])
        for block in self.blocks:
            x = block(x, pad_mask)
        return self.head(self.ln_f(x))


def build_model(config):
    """Return an nn.Module from a dictionary of StoryTransformerLM arguments."""
    return StoryTransformerLM(**config)
