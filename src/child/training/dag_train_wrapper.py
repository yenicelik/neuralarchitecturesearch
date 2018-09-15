"""
    This is one example training wrapper which uses the interface given by the Base class.
    We will use the RNN cell as this provides the right interface for the NAS we will later implement
"""
import torch
from torch import nn
from tensorboardX import SummaryWriter
from torch.autograd import Variable

from src.child.networks.rnn.dag_rnn import dlxDAGRNNModule
from src.child.training.train_wrapper_base import TrainWrapperBase


class DAGTrainWrapper(TrainWrapperBase):

    def debug_tools(self):
        """
            Includes any logic which includes having
        :return:
        """
        # Debugging tools
        self.writer = SummaryWriter(log_dir="/Users/david/neuralarchitecturesearch/tmp/runs/")

    def __init__(self, model):
        super(TrainWrapperBase, self).__init__()

        self.model = model

        self.criterion = nn.NLLLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters()) # The .parameters is required, and automatically built-in into any torch model

        self.debug_tools()

    def train(self, X, Y):
        """
            Trains the model on a certain dataset.
            The datasets have to be of the shape:

            X.size() <- (total_data_size, time_length, **data_size )

            --> Watch out! There should be a cutoff and padding amongst batches!

        :param X: The data
        :param Y: The shape
        :return:
        """

        assert X.size() == Y.size(), ("Not same size! (X, Y) :: ", X.size(), Y.size())

        data_size = X.size(0)
        timestep_length = 10
        losses = torch.empty(data_size//timestep_length)

        # Do exactly one epoch
        for train_idx in range(0, data_size, timestep_length):

            if train_idx + timestep_length > data_size:
                break

            print("Training! At step: ", train_idx)
            X_cur = X[train_idx:train_idx+timestep_length, :].transpose(0, 1)
            Y_cur = Y[train_idx:train_idx+timestep_length, :].transpose(0, 1)
            print("Shapes are: ", X_cur.size(), Y_cur.size())

            Y_hat = self.model.forward(X).long()
            loss = self.criterion(Y_hat, Y)
            print("Loss: ", loss)
            loss.backward()
            self.optimizer.step()

            losses[train_idx//timestep_length] = loss

            self.writer.add_scalar('loss/train_loss', loss, train_idx)

            if train_idx % 100 == 0: # Export the tensorboard representation
                self.writer.export_scalars_to_json("/Users/david/neuralarchitecturesearch/tmp/all_scalar.json")

        losses = losses / (data_size // timestep_length)

        print(losses)


if __name__ == "__main__":
    print("Do a bunch of forward passes: ")

    dag_description = "0 0 0 1 1 2 1 2 0 2 0 5 1 1 0 6 1 8 1 8 1 8 1"
    dag_list = [int(x) for x in dag_description.split()]
    print(dag_list)

    # model = dlxExampleRNNModule()
    model = dlxDAGRNNModule(dag=dag_list)

    trainer = DAGTrainWrapper(model)
    # Example forward pass
    X = torch.randn((401, 4, 50))
    trainer.train(X[:400,:], X[1:,:])
