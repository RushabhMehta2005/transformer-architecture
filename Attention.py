import torch
import torch.nn as nn
from common import clones
import math


def attention(query, key, value, mask=None, dropout=None):
    """Compute scaled dot-product attention with optional masking and dropout."""
    d_k = key.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)  # Mask positions to ignore in attention
    p_attn = scores.softmax(dim=-1)  # Apply softmax to get attention weights
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0  # Ensure model size is divisible by the number of heads
        self.h = h
        self.d_k = d_model // h
        self.dropout = nn.Dropout(p=dropout)
        self.linears = clones(nn.Linear(d_model, d_model), 4)  # Create linear layers for query, key, value, output

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)  # Align mask dimensions for multi-head attention
        nbatches = query.size(0)

        # Apply linear layers, reshape, and split into heads
        query, key, value = [
            linear_layer(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for linear_layer, x in zip(self.linears, (query, key, value))
        ]

        # Perform attention and combine heads
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # Concatenate heads and apply final linear layer
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
