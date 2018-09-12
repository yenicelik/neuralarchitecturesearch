"""
    This class defines the abse elements that our neural network need to incorporate.
    This neural network incorporates only the model logic, which includes:
        - generating the network
        - forward step
        - backward step

    One should notice that torch does not have a "building" and one "computation" phase.
    The pass through immediately builds the network, and then passes the tensor through it. (this is due to dynamic computation)
"""

import numpy as np
import torch
from torch import nn

class dlxRNNModelBase(nn.Module):
    """
        We don't need a backward pass, as this is implicitly computed by the forward pass
        -- Write tests if the backward pass actually optimizes the parameters
        Everything before __init__ is done only once (as a setup). Everything after is run during runtime.
    """

    def _name(self):
        raise NotImplementedError

    def build_network(self, description_string):
        """

        :param description_string: The string which defines the cell of the unrolled RNN in the ENAS paper
        :return:
        """
        raise NotImplementedError

    def __init__(self):
        super(dlxRNNModelBase).__init__()

    def cell(self, X):
        raise NotImplementedError

    def forward(self, X):
        """

            time_steps = X.size(0)
            batch_size = X.size(1)
        :param X: Input in the format of (time_steps, batch_size, *{data.shape})
        :return:
        """
        raise NotImplementedError