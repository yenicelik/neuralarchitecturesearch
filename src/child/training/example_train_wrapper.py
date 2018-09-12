"""
    This is one example training wrapper which uses the interface given by the Base class.
    We will use the RNN cell as this provides the right interface for the NAS we will later implement
"""
import torch
import numpy as np
from torch import nn

from src.child.networks.rnn.example_rnn import dlxExampleRNNModule
from src.child.training.train_wrapper_base import TrainWrapperBase
from src.utils.random_tensor import random_tensor


class ExampleTrainWrapper(TrainWrapperBase):

    def __init__(self, model):
        super(TrainWrapperBase, self).__init__()

        self.model = model

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01) # The .parameters is required, and automatically built-in into any torch model

    def train(self, X, Y):
        """
            Trains the model on a certain dataset.
            The datasets have to be of the shape:

            X.size() <- (total_length, batch_size, **data_size )

            --> Watch out! There should be a cutoff and padding amongst batches!

        :param X: The data
        :param Y: The shape
        :return:
        """

        assert X.size() == Y.size(), ("Not same size! (X, Y) :: ", X.size(), Y.size())

        data_size = X.size(0)
        timestep_length = 10
        losses = torch.empty(data_size)

        # Do exactly one epoch
        for train_idx in range(0, data_size, timestep_length):

            if train_idx + timestep_length > data_size:
                break

            print("Training! At step: ", train_idx)
            X_cur = X[train_idx:train_idx+timestep_length, :]
            Y_cur = Y[train_idx:train_idx+timestep_length, :]
            print("Shapes are: ", X_cur.size(), Y_cur.size())

            Y_hat = self.model.forward(X)
            loss = self.criterion(Y_hat, Y)
            loss.backward()
            self.optimizer.step()

            # losses[train_idx] = loss



if __name__ == "__main__":
    print("Do a bunch of forward passes: ")

    model = dlxExampleRNNModule()

    trainer = ExampleTrainWrapper(model)
    # Example forward pass
    X = random_tensor((101, 4, 50))
    trainer.train(X[:100,:], X[1:,:])
