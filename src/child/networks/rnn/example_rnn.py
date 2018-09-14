"""
    This is one example RNN which uses the interface given by the Base class.
    We will use the RNN cell as this provides the right interface for the NAS we will later implement
"""

import numpy as np
import torch
from torch import nn

from src.child.networks.rnn.Base import dlxRNNModelBase

class dlxExampleRNNModule(dlxRNNModelBase):
    """
        We don't need a backward pass, as this is implicitly computed by the forward pass
        -- Write tests if the backward pass actually optimizes the parameters
    """

    def _name(self):
        return "ExampleRNN_using_RNNCell"

    def build_network(self, description_string):
        """

        :param description_string: The string which defines the cell of the unrolled RNN in the ENAS paper
        :return:
        """
        # This is not needed for this example network (which uses LSTMs)
        assert False

    def __init__(self):
        super(dlxExampleRNNModule, self).__init__()

        # Used probably for every application
        self.embedding_module_encoder = nn.Linear(50, 8)  # 2 words in vocab, 5 dimensional embeddings
        self.embedding_module_decoder = nn.Linear(7, 50)

        # Use probably only in this example
        self.cell_module = nn.RNN(input_size=8, hidden_size=7)

        # Later on, we will spawn these cells ourselves
        # self.cell = nn.LSTM(128, 128, 2)

    def embedding_encoder(self, inputx):
        """
            Pass a tensor through an embeddings, such that the shape is appropriate for the LSTM
        :param X:
        :return:
        """
        return self.embedding_module_encoder(inputx)[None, :]

    def embedding_decoder(self, inputx):
        """
                    Pass a tensor through an embeddings, such that the shape is appropriate for the LSTM
                :param X:
                :return:
                """
        return self.embedding_module_decoder(inputx)

    def cell(self, inputx, hidden):
        """
            Use an LSTM as an example cell
        :return:
        """
        # return nn.LSTM(128, 128, 2, dropout=0.05)
        # raise NotImplementedError
        return self.cell_module(inputx, hidden)

    def forward(self, X):
        assert len(X.size()) > 2, ("Not enough dimensions! Expected more than 2 dimensions, but have ", X.size())

        time_steps = X.size(0)
        batch_size = X.size(1)

        outputs = []

        # First input to cell
        embedded_X = self.embedding_encoder(X[0, :])
        logit, hidden = self.cell(inputx=embedded_X, hidden=None)
        decoded_logit = self.embedding_decoder(logit)
        outputs.append(decoded_logit)

        # Dynamic unrolling of the cell for the rest of the timesteps
        for i in range(1, time_steps):
            # print("Unrolling...", i)

            # print(X[i, :].size())

            embedded_X = self.embedding_encoder(X[i, :])
            logit, hidden = self.cell(inputx=embedded_X, hidden=hidden)
            decoded_logit = self.embedding_decoder(logit)
            outputs.append(decoded_logit)

            # print(embedded_X[:].size())
            # print(logit.size())

        output = torch.cat(outputs, 0)
        # print(output.size())

        return output


if __name__ == "__main__":
    print("Do a bunch of forward passes: ")
    model = dlxExampleRNNModule()

    # Example forward pass
    X = torch.randn((5, 4, 50))
    model.forward(X)
