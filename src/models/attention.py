import math
import torch
import torch.nn as nn


class SASRecAttention(nn.Module):
    """
    Multi-Head Self-Attention module for the SASRec recommender.

    This module projects the input into Query (Q), Key (K), and Value (V) tensors,
    splits them across multiple heads, scales the dot-product similarity scores,
    applies the combined causal/padding mask, computes attention weights via softmax,
    and returns the weighted sum of values.
    """
    def __init__(self, hidden_units: int, num_heads: int, dropout_rate: float):
        super().__init__()
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.head_dim = hidden_units // num_heads

        assert hidden_units % num_heads == 0, "hidden_units must be divisible by num_heads"

        # Linear projections for Query, Key, and Value
        self.q_proj = nn.Linear(hidden_units, hidden_units)
        self.k_proj = nn.Linear(hidden_units, hidden_units)
        self.v_proj = nn.Linear(hidden_units, hidden_units)

        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass for Multi-Head Self-Attention.

        Args:
            queries: Tensor of shape (batch_size, seq_len_q, hidden_units)
            keys: Tensor of shape (batch_size, seq_len_k, hidden_units)
            values: Tensor of shape (batch_size, seq_len_k, hidden_units)
            attn_mask: Boolean tensor of shape (batch_size, 1, seq_len_q, seq_len_k)
                       or similar broadcastable shape, where True indicates
                       positions to be masked out.

        Returns:
            Output tensor of shape (batch_size, seq_len_q, hidden_units)
        """
        batch_size, seq_len_q, _ = queries.size()
        _, seq_len_k, _ = keys.size()

        # 1. Project inputs to Q, K, V
        # Shape: (batch_size, seq_len, hidden_units)
        Q = self.q_proj(queries)
        K = self.k_proj(keys)
        V = self.v_proj(values)

        # Split projections into multiple heads and transpose
        # Shape: (batch_size, num_heads, seq_len, head_dim)
        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)

        # 2. Compute attention scores and scale
        # Shape: (batch_size, num_heads, seq_len_q, seq_len_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 3. Apply the combined causal and padding attention mask
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, -1e9)

        # 4. Compute Softmax attention weights
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 5. Compute the weighted sum of Values (V)
        # Shape: (batch_size, num_heads, seq_len_q, head_dim)
        context = torch.matmul(attn_weights, V)

        # 6. Concatenate heads back to retrieve original hidden dimension
        # Shape: (batch_size, seq_len_q, hidden_units)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.hidden_units)

        return context
