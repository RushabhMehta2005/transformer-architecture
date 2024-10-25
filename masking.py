import torch


def subsequent_mask(size):
    "Mask out subsequent positions."
    """
    for size = 3:
    returns 
    [
        [
            [True, False, False],
            [True, True, False],
            [True, True, True]
        ]
    ]
    """
    attn_shape = (1, size, size) # Shape of the attention mask
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8) # Upper triangular matrix
    return subsequent_mask == 0 # Mask out subsequent positions

