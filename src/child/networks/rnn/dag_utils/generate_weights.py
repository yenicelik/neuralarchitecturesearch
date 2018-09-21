"""
    This file contains the logic where we generate the individual weights.
    The weights are the weights used for the linear connections between individual blocks in the
    RNN cell
"""

import collections
import torch
from torch import nn

def generate_weights(input_size, hidden_size, num_blocks):
    """
        This number 0 is reserved for the input.
    :param input_size:
    :param hidden_size:
    :param num_blocks:
    :return:
    """

    block2block = collections.defaultdict(dict)

    # TODO:!! Do we have one single one for all, or do we have different ones for each individual one?
    # I think we have both. One is for when the last block used is "0"
    print("Generating the lienar connection between the previous hidden output, and the first block")
    hidde2firstblock = nn.Linear(
        in_features=input_size,
        out_features=hidden_size,
        bias=False
    )

    for idx in range(0, num_blocks):
        for jdx in range(idx+1, num_blocks): # Do we add a +1 here?
            print("Generating the linear connection between blocks: ", idx, jdx)
            block2block[idx][jdx] = nn.Linear(
                in_features=hidden_size,
                out_features=hidden_size,
                bias=False
            )

    return hidde2firstblock, block2block

if __name__ == "__main__":
    print("Starting to generate the connections...")

    hidden2block, block2block = generate_weights(50, 8, 5)
