"""
    This is one example training wrapper which uses the interface given by the Base class.
    We will use the RNN cell as this provides the right interface for the NAS we will later implement
"""
import numpy as np
import torch
from torch import nn
from tensorboardX import SummaryWriter

from src.child.training.train_wrapper_base import TrainWrapperBase

# Debug tools
from src.preprocessor.text import Corpus, batchify
# corpus = Corpus("/Users/david/neuralarchitecturesearch/data/ptb/")

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

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters()) # The .parameters is required, and automatically built-in into any torch model

        self.debug_tools()

    def predict(self, X, n=1):
        """
            Predicts the next n elements given past X elements.

                X.size() <- (total_data_size, time_length, **data_size )

        :param X:
        :param Y:
        :return:
        """

        assert n==1, ("Cases where n>1 are not implemented yet!", n)

        # Take the very last output from a "forward"
        Y_hat = self.model.forward(X)
        prediction = Y_hat[:, -1]
        print("Predictions have size: ", prediction.size())

        prediction_index = torch.argmax(prediction, dim=-1, keepdim=True)
        print("Prediction index has size: ", prediction_index.size())

        tmax, _ = torch.max(prediction, dim=-1, keepdim=True)
        print(tmax)
        print("Prediction and tmax shape: ", prediction.size(), tmax.size())
        e_x = torch.sub(prediction, tmax)
        class_probabilities = e_x / torch.sum(e_x, dim=-1, keepdim=True)

        print("Summed probabilites are: (should be all 1)", torch.sum(class_probabilities, dim=-1))
        print(class_probabilities.size())

        return prediction_index, class_probabilities

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

            X_cur = X[train_idx:train_idx+timestep_length, :]
            Y_cur = Y[train_idx:train_idx+timestep_length, :]
            # print_batches(X_cur, Y_cur)
            # exit(0)
            # X_cur = X_cur.transpose(0, 1)
            # Y_cur = Y_cur.transpose(0, 1)

            Y_hat = self.model.forward(X_cur)
            # Take argmax because classification
            # print("Output from model rnn is: ", Y_hat.size())
            # Y_hat = torch.argmax(Y_hat, len(Y_hat.size())-1)
            Y_hat = Y_hat.transpose(1, -1) # TODO: Fix this thing of transposing randomly! Define the input dimension and feed it in like that
            Y_hat = Y_hat.transpose(2, -1)

            Y_cur = Y_cur.transpose(1, -1)  # TODO: Fix this thing of transposing randomly! Define the input dimension and feed it in like that
            Y_cur = Y_cur.transpose(2, -1)
            print("Shape of real Y and found Y: ", Y_hat.size(), Y_cur.size())
            Y_cur = Y_cur.squeeze()
            # print("Two inputs to the criterion: ", Y_hat.size(), Y_cur.size())
            # print("Input types are: ", Y_hat.type(), Y_cur.type())
            loss = self.criterion(Y_hat, Y_cur)
            # print("Loss: ", loss)
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

    X = torch.LongTensor(401, 4, 1).random_(0, 10000)
    print(X.size())

    dag_description = "0 0 0 1 1 2 1 2 0 2 0 5 1 1 0 6 1 8 1 8 1 8 1"
    dag_list = [int(x) for x in dag_description.split()]
    print(dag_list)
    import src.child.networks.rnn.dag_rnn as dag_rnn

    # model = dlxExampleRNNModule()
    model = dag_rnn.dlxDAGRNNModule()
    model.overwrite_dag(dag_list)

    trainer = DAGTrainWrapper(model)
    # Example forward pass
    # X = torch.randint((401, 4, 50))
    # trainer.train(X[:400,:], X[1:,:])

    trainer.predict(X[:400, :])
