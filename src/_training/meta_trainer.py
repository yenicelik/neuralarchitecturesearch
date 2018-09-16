"""
    This file includes the training, where we iteratively train
    1. the child model
    2. the controller (based on the loss of the child model)
"""
import torch
import random
from torch.autograd import Variable

from src.child.networks.rnn.dag_rnn import dlxDAGRNNModule
from src.child.training.dag_train_wrapper import DAGTrainWrapper
from src._training.debug_utils.rnn_debug import print_batches, load_dataset


class MetaTrainer:

    def __init__(self, X_train, Y_train, X_val, Y_val, X_test, Y_test):
        """
            All the X's have to be of the following shape:
                X.size() <- (total_data_size, time_length, **data_size )

        :param X_train:
        :param Y_train:
        :param X_val:
        :param Y_val:
        :param X_test:
        :param Y_test:
        """
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val
        self.X_test = X_test
        self.Y_test = Y_test

        # Spawn one child model
        self.child_model = dlxDAGRNNModule(12)
        self.child_trainer = DAGTrainWrapper(self.child_model)

    def train_child(self, dag):
        # This should be replaced by batch getters
        # Spawn child trainer and model
        self.child_model.overwrite_dag(dag)

        # Prepare the sequence data (to be shifter by one on the time-axis)
        self.child_trainer.train(self.X_train, self.Y_train)

    def get_child_validation_loss(self):
        # TODO: Implement
        return random.random()

    def train_controller_and_child(self):
        # Setting up the trainers
        for current_epoch in range(10):
            dag_description = "0 0 0 1 1 2 1 2 0 2 0 5 1 1 0 6 1 8 1 8 1 8 1"
            dag_list = [int(x) for x in dag_description.split()]
            self.train_child(dag_list)

            loss = self.get_child_validation_loss()
            print("Loss: ", loss)


if __name__ == "__main__":
    print("Starting to train the meta model")
    # meta_trainer.train()

    data, target = load_dataset(dev=True)

    # print_batches(data, target)

    # print("Input to the meta trainer is: ", data.size(), target.size())

    meta_trainer = MetaTrainer(
        X_train=data,
        Y_train=target,
        X_val=data,
        Y_val=target,
        X_test=data,
        Y_test=target
    )

    meta_trainer.train_controller_and_child()

    # As a test, run the train_child function with the batchloader)

    # print(to_word)

    # for i in data
