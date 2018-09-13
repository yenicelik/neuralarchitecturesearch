"""
    This file contains the logic where we generate the individual weights.
    The weights are the weights used for the linear connections between individual blocks in the
    RNN cell
"""

import collections
import torch
from torch import nn


def generate_weights(hidden_size, num_blocks):
    out = collections.defaultdict(dict)

    for idx in range(num_blocks):
        for jdx in range(idx + 1, num_blocks):
            print("Generating the linear connection between blocks: ", idx, jdx)
            out[idx][jdx] = nn.Linear(
                in_features=hidden_size,
                out_features=hidden_size,
                bias=False
            )

    return out


if __name__ == "__main__":
    print("Starting to generate the connections...")

    module_list = generate_weights(8, 5)

    print(module_list)
