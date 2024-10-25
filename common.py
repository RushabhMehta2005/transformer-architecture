import torch
import torch.nn as nn
import copy


class LayerNorm(nn.Module):
    def __init__(self, num_features, epsilon=1e-6):
        super().__init__()
        self.scale_factor = nn.Parameter(torch.ones(num_features))  # Learnable scale
        self.bias = nn.Parameter(torch.zeros(num_features))  # Learnable bias
        self.epsilon = epsilon  # Small constant to avoid division by zero

    def forward(self, x):
        mean = x.mean(-1, keepdims=True)
        std = x.std(-1, keepdims=True)
        return self.scale_factor * (x - mean) / (std + self.epsilon) + self.bias


class SubLayerConnection(nn.Module):
    def __init__(self, size, p_dropout):
        super().__init__()
        self.norm = LayerNorm(size)  # Normalize input
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))  # Apply residual connection


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, p_dropout=0.1):
        super().__init__()
        self.layer1 = nn.Linear(d_model, d_ff)  # First linear layer
        self.layer2 = nn.Linear(d_ff, d_model)  # Second linear layer
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x):
        return self.layer2(self.dropout(self.layer1(x).relu()))  # Apply non-linearity and dropout


def clones(module, N):
    """Create N identical copies of a module."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
