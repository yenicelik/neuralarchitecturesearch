"""
    This file initializes the weights to the appropriate values
"""

import torch

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0)
