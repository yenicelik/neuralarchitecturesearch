"""
    This file includes the training, where we iteratively train
    1. the child model
    2. the controller (based on the loss of the child model)
"""
import torch
import random
from torch.autograd import Variable

import src.child.networks.rnn.dag_rnn as dag_rnn #.dlxDAGRNNModule
import src.child.training.dag_train_wrapper as dag_train_wrapper
from src._training.debug_utils.rnn_debug import print_batches, load_dataset
from src.model_config import ARG

from src.utils.debug_utils.tensorboard_tools import tx_writer


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
        self.child_model = dag_rnn.dlxDAGRNNModule()
        self.child_trainer = dag_train_wrapper.DAGTrainWrapper(self.child_model)

    def get_child_validation_loss(self):
        # TODO: Implement
        return random.random()

    def train_controller_and_child(self):
        # Setting up the trainers
        dag_description = "0 0 0 1 1 2 1 2 0 2 0 5 1 1 0 6 1 8 1 8 1 8 1"
        dag_list = [int(x) for x in dag_description.split()]

        self.child_model.overwrite_dag(dag_list)

        loss = self.child_trainer.get_loss(self.X_val, self.Y_val)
        print("Validation loss: ", loss)

        for current_epoch in range(ARG.max_epoch):

            # TODO: Do we create a new model for every epoch, or for each "max steps"?
            for minibatch_offset in range(0, self.X_train.size(1), ARG.shared_max_step):

                dag_description = "0 0 0 1 1 2 1 2 0 2 0 5 1 1 0 6 1 8 1 8 1 8 1"
                dag_list = [int(x) for x in dag_description.split()]

                self.child_model.overwrite_dag(dag_list)

                X_minibatch = self.X_train[
                                  minibatch_offset:
                                  minibatch_offset+ARG.shared_max_step
                              ]
                Y_minibatch = self.Y_train[
                                  minibatch_offset:
                                  minibatch_offset + ARG.shared_max_step
                              ]

                self.child_trainer.train(
                    X=X_minibatch,
                    Y=Y_minibatch
                )

                loss = self.child_trainer.get_loss(self.X_val, self.Y_val)
                print("Validation loss: ", loss[0])

                eval_idx = minibatch_offset // ARG.shared_max_step
                tx_writer.add_scalar('loss/child_val_loss', loss, eval_idx)

            if current_epoch > ARG.shared_decay_after:
                new_lr = ARG.shared_lr * ( ARG.shared_decay**(current_epoch-ARG.shared_decay_after) )
                print("Updating learning rate to ", new_lr)
                self.child_trainer.update_lr(new_lr)

if __name__ == "__main__":
    print("Starting to train the meta model")
    # meta_trainer.train()

    train_off = 600

    data, target = load_dataset(dev=True, dev_size=1500)
    X_train = data[:train_off]
    Y_train = data[:train_off]
    X_val = data[train_off:]
    Y_val = data[train_off:]


    # print_batches(data, target)

    # print("Input to the meta trainer is: ", data.size(), target.size())

    meta_trainer = MetaTrainer(
        X_train=X_train,
        Y_train=Y_train,
        X_val=X_val,
        Y_val=Y_val,
        X_test=data,
        Y_test=target
    )

    dag_description = "0 0 0 1 1 2 1 2 0 2 0 5 1 1 0 6 1 8 1 8 1 8 1"
    dag_list = [int(x) for x in dag_description.split()]

    # meta_trainer.train_controller_and_child()
    meta_trainer.train_controller_and_child()

    # As a test, run the train_child function with the batchloader)

    # print(to_word)

    # for i in data
