import torch
import torch.nn as nn
import math


class Embedding(nn.Module):
    def __init__(self, vocab, d_model):
        super().__init__()
        self.lookup_table = nn.Embedding(vocab, d_model)  # Embedding layer for vocabulary lookup
        self.d_model = d_model

    def forward(self, x):
        embeddings = self.lookup_table(x)
        return embeddings * math.sqrt(self.d_model)  # Scale embeddings by model dimension
