"""
    This is one example RNN which uses the interface given by the Base class.
    We will use
"""

import numpy as np
import torch
from torch import nn

from src.child.networks.rnn.Base import dlxRNNModelBase
from src.utils.random_tensor import random_tensor


class dlxExampleRNNModule(dlxRNNModelBase):
    """
        We don't need a backward pass, as this is implicitly computed by the forward pass
        -- Write tests if the backward pass actually optimizes the parameters
    """

    def _name(self):
        return "ExampleRNN_using_LSTM"

    def build_network(self, description_string):
        """

        :param description_string: The string which defines the cell of the unrolled RNN in the ENAS paper
        :return:
        """
        # This is not needed for this example network (which uses LSTMs)
        assert False

    def __init__(self):
        super(dlxExampleRNNModule).__init__()

        # Later on, we will spawn these cells ourselves
        # self.cell = nn.LSTM(128, 128, 2)

    def cell(self, X):
        """
            Use an LSTM as an example cell
        :return:
        """
        # return nn.LSTM(128, 128, 2, dropout=0.05)
        # raise NotImplementedError
        return self.cell(X)

    def forward(self, X):
        time_steps = X.size(0)
        batch_size = X.size(1)

        # Dynamic unrolling of the cell
        for i in range(time_steps):
            print("Unrolling...", i)



if __name__ == "__main__":
    print("Do a bunch of forward passes: ")
    model = dlxExampleRNNModule()

    # Example forward pass
    X = random_tensor((12, 2, 4))
    model.forward(X)