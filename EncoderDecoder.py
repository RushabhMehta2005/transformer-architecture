import torch
import torch.nn as nn
from torch.nn.functional import log_softmax


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed  # Embedding for source input
        self.tgt_embed = tgt_embed  # Embedding for target input
        self.generator = generator  # Final output projection layer

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)  # Encode embedded source input

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)  # Decode with embedded target


class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.projection = nn.Linear(d_model, vocab_size)  # Linear layer to map to vocab size

    def forward(self, x):
        return log_softmax(self.projection(x), dim=-1)  # Apply log softmax for output probabilities
