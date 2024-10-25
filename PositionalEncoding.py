import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, p_dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=p_dropout)  # Dropout layer for regularization

        # Create position indices and frequencies for positional encoding
        pos = torch.arange(0, max_len).unsqueeze(1)
        frequency = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        
        # Initialize positional encoding matrix
        pos_encoding = torch.zeros(max_len, d_model)
        pos_encoding[:, 0::2] = torch.sin(pos * frequency)  # Even indices
        pos_encoding[:, 1::2] = torch.cos(pos * frequency)  # Odd indices

        pos_encoding = pos_encoding.unsqueeze(0)  # Add batch dimension
        self.register_buffer("pos_encoding", pos_encoding)  # Register as a buffer

    def forward(self, embedding):
        # Add positional encoding to the input embeddings
        embedding = embedding + self.pos_encoding[:, : embedding.size(1)].requires_grad_(False)
        return self.dropout(embedding)  # Apply dropout
