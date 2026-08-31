import torch
import torch.nn as nn

from src.models.layers import PointwiseFeedForward
from src.models.attention import SASRecAttention


class SASRec(nn.Module):
    """
    SASRec: Self-Attentive Sequential Recommender in PyTorch.

    This model encodes sequential user interaction history using self-attention blocks.
    It takes a sequence of item IDs, applies item and positional embeddings,
    passes them through stacked Transformer-style self-attention and pointwise
    feed-forward layers, and outputs a contextual sequence representation.

    The model implements a clean, internally consistent Pre-LN Transformer structure:
    LN -> Attention -> Residual -> LN -> FeedForward -> Residual.
    """
    def __init__(
        self,
        item_count: int,
        maxlen: int,
        hidden_units: int,
        num_blocks: int,
        num_heads: int,
        dropout_rate: float,
    ):
        super().__init__()
        self.hidden_units = hidden_units
        self.maxlen = maxlen

        # 1. Embeddings
        # ID 0 is reserved for padding, and is kept at zero.
        self.item_emb = nn.Embedding(
            num_embeddings=item_count + 1,
            embedding_dim=hidden_units,
            padding_idx=0,
        )
        self.pos_emb = nn.Embedding(
            num_embeddings=maxlen,
            embedding_dim=hidden_units,
        )
        self.emb_dropout = nn.Dropout(p=dropout_rate)

        # 2. Transformer blocks
        self.attention_blocks = nn.ModuleList([
            SASRecAttention(hidden_units, num_heads, dropout_rate)
            for _ in range(num_blocks)
        ])
        self.ffn_blocks = nn.ModuleList([
            PointwiseFeedForward(hidden_units, dropout_rate)
            for _ in range(num_blocks)
        ])
        self.ln1_blocks = nn.ModuleList([
            nn.LayerNorm(hidden_units, eps=1e-8)
            for _ in range(num_blocks)
        ])
        self.ln2_blocks = nn.ModuleList([
            nn.LayerNorm(hidden_units, eps=1e-8)
            for _ in range(num_blocks)
        ])

        # 3. Final layer normalization
        self.final_ln = nn.LayerNorm(hidden_units, eps=1e-8)

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of SASRec.

        Args:
            input_seq: Tensor of shape (batch_size, maxlen) containing item IDs.

        Returns:
            Contextual sequence representation tensor of shape (batch_size, maxlen, hidden_units)
        """
        batch_size, seq_len = input_seq.size()

        # 1. Create attention masks
        # Key padding mask: True where the item is padding (ID 0)
        # Shape: (batch_size, 1, 1, seq_len)
        key_padding_mask = (input_seq == 0).unsqueeze(1).unsqueeze(2)

        # Causal mask: True where query position cannot attend to future key positions
        # Shape: (1, 1, seq_len, seq_len)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_seq.device),
            diagonal=1,
        ).bool().unsqueeze(0).unsqueeze(1)

        # Combine masks: element-wise logical OR
        # Shape: (batch_size, 1, seq_len, seq_len)
        attn_mask = causal_mask | key_padding_mask

        # 2. Compute embeddings and add positional encoding
        # Scale item embeddings by sqrt(hidden_units) matching the reference
        seq_emb = self.item_emb(input_seq) * (self.hidden_units ** 0.5)

        # Generate positions: [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=input_seq.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_emb(positions)

        x = seq_emb + pos_emb
        x = self.emb_dropout(x)

        # Binary mask to zero out sequence representations at padding positions
        # Shape: (batch_size, seq_len, 1)
        pad_mask = (input_seq != 0).unsqueeze(-1).float()
        x = x * pad_mask

        # 3. Pass through stacked blocks
        for ln1, attn, ln2, ffn in zip(
            self.ln1_blocks, self.attention_blocks, self.ln2_blocks, self.ffn_blocks
        ):
            # --- Self-Attention Block (Pre-LN) ---
            x_norm = ln1(x)
            attn_out = attn(
                queries=x_norm,
                keys=x_norm,
                values=x_norm,
                attn_mask=attn_mask,
            )
            x = x + attn_out

            # --- Pointwise Feed-Forward Block (Pre-LN) ---
            x_norm = ln2(x)
            ffn_out = ffn(x_norm)
            x = x + ffn_out

            # Zero-mask padded positions at the end of each block
            x = x * pad_mask

        # 4. Final Layer Normalization
        x = self.final_ln(x)
        x = x * pad_mask

        return x
