import torch
import torch.nn as nn
import copy
from Attention import MultiHeadAttention
from Embeddings import Embedding
from EncoderDecoder import EncoderDecoder, Generator
from Encoder import Encoder, EncoderLayer
from Decoder import Decoder, DecoderLayer
from common import FeedForward
from PositionalEncoding import PositionalEncoding


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, p_dropout=0.1):
    # Create a copy function for duplicating layers
    c = copy.deepcopy
    
    # Initialize components of the model
    attn = MultiHeadAttention(h, d_model, p_dropout)  # Multi-head attention mechanism
    ff = FeedForward(d_model, d_ff, p_dropout)  # Feed-forward network
    positional = PositionalEncoding(d_model, p_dropout, max_len=5000)  # Positional encoding layer
    
    # Construct the Encoder-Decoder architecture
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), p_dropout), N),  # Encoder with layers
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), p_dropout), N),  # Decoder with layers
        nn.Sequential(Embedding(src_vocab, d_model), c(positional)),  # Source embedding with positional encoding
        nn.Sequential(Embedding(tgt_vocab, d_model), c(positional)),  # Target embedding with positional encoding
        Generator(d_model, tgt_vocab),  # Linear layer to project to target vocabulary
    )

    # Initialize parameters using Xavier uniform distribution
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model  # Return the constructed model
