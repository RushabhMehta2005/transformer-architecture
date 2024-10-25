import torch
import torch.nn as nn
from common import clones, LayerNorm, SubLayerConnection


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)  # Clone N decoder layers
        self.norm = LayerNorm(layer.size)  # Final layer normalization

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)  # Pass through each decoder layer
        return self.norm(x)  # Normalize the output


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, p_dropout):
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SubLayerConnection(size, p_dropout), 3)  # Sublayer connections for each component

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))  # Self-attention
        x = self.sublayer[1](x, lambda x: self.self_attn(x, memory, memory, src_mask))  # Cross-attention
        x = self.sublayer[2](x, self.feed_forward)  # Feed-forward
        return x
