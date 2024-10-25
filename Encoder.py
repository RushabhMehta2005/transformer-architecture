import torch
import torch.nn as nn
import copy
from common import *


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)  # Create N identical encoder layers
        self.norm = LayerNorm(layer.size)  # Final layer normalization

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)  # Pass input through each encoder layer
        return self.norm(x)  # Normalize output


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, p_dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SubLayerConnection(size, p_dropout), 2)  # Two sublayers: self-attn and feed-forward
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))  # Self-attention
        return self.sublayer[1](x, self.feed_forward)  # Feed-forward

