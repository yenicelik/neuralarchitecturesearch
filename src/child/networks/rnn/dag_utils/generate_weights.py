"""
    This file contains the logic where we generate the individual weights.
    The weights are the weights used for the linear connections between individual blocks in the
    RNN cell
"""

import collections
import torch
from torch import nn


def generate_weights(hidden_size, num_blocks):

    hidden2block = collections.defaultdict(dict)
    block2block = collections.defaultdict(dict)

    for idx in range(1, num_blocks+1):
        print("Generating the lienar connection between the previous hidden input, and the block: ", idx)
        hidden2block[idx] = nn.Linear(
            in_features=hidden_size,
            out_features=hidden_size,
            bias=False
        )

    for idx in range(1, num_blocks+1):
        for jdx in range(idx + 1, num_blocks+1):
            print("Generating the linear connection between blocks: ", idx, jdx)
            block2block[idx][jdx] = nn.Linear(
                in_features=hidden_size,
                out_features=hidden_size,
                bias=False
            )

    return hidden2block, block2block


if __name__ == "__main__":
    print("Starting to generate the connections...")

    hidden2block, block2block = generate_weights(8, 5)

    print(module_list)
