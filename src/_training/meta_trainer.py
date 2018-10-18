"""
    This file includes the training, where we iteratively train
    1. the child model
    2. the controller (based on the loss of the child model)

    Some possible DAG descriptions include:
        dag_description = "0 0 2 1 1 0 3 3 1 4 0 0 2"
        dag_description = "1 0 3 0 1 1 2 3 0"
"""
import gc
import torch
import numpy as np
from torch.autograd import Variable

from src.MetaTrainerBase import MetaTrainerBase
from src.child.rnn import dag_rnn# .dlxDAGRNNModule
import src.child.child_wrapper as dag_train_wrapper
from src.config import config, C_DEVICE
from src._training.debug_utils.rnn_debug import load_dataset, _print_batches
from src.controller.controller_network import ControllerLSTM
from src.controller.controller_wrapper import ControllerWrapper
from src.model_config import ARG
from src.utils.debug_utils.exploding_gradients import _check_abs_max_grad
from src.utils.debug_utils.size_network import memory_usage_resource
from src.utils.debug_utils.tensorboard_tools import tx_writer


def _print_to_std_memory_logs(identifier):
    """
        Print memory usage to screen
    :return:
    """
    if config['debug_memory']:
        print("Memory usage P{}: ".format(identifier), memory_usage_resource())

class MetaTrainer(MetaTrainerBase):
    """
        The main trainer object.
        This script covers the logic to iteratively
            train the child model
            and train the controller model
    """

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
        super(MetaTrainer, self).__init__()

        self.X_train = Variable(X_train)
        self.Y_train = Variable(Y_train)
        self.X_val = Variable(X_val)
        self.Y_val = Variable(Y_val)
        self.X_test = Variable(X_test)
        self.Y_test = Variable(Y_test)

        # Spawn one controller model
        self.controller_model = ControllerLSTM()
        self.controller_model.to(C_DEVICE)

        self.controller_wrapper = ControllerWrapper(self.controller_model)

        # Spawn one child model
        self.child_model = dag_rnn.dlxDAGRNNModule()
        self.child_model.to(C_DEVICE)

        self.child_wrapper = dag_train_wrapper.ChildWrapper(self.child_model)

    ############################################
    # Anything related to the controller model
    ############################################

    ############################################
    # Anything related to the child model
    ############################################
    def get_child_validation_loss(self):
        """
            Wrapper around the 'get_loss' function in the child model.
            This can take random indices, just because we apply it quite frequently!
        :return:
        """
        # Choose random indices to feed in to validation loss getter
        ranomd_indices = np.random.choice(
            np.arange(self.X_val.size(0)),
            ARG.shared_max_step * ARG.batch_size // 3
        )

        with torch.no_grad():
            loss = self.child_wrapper.get_loss(
                self.X_val[ranomd_indices].detach(),
                self.Y_val[ranomd_indices].detach(),
                'validation'
            )

        return loss

    def get_child_test_loss(self):
        """
            Wrapper around the 'get_loss' function in the child model.
            This has to go through the entire test loss, as this has to be fast!
        :return:
        """
        return self.child_wrapper.get_loss(self.X_test, self.Y_test, 'test')

    ############################################
    # Anything related to the joint algorithms
    ############################################
    def train_controller_and_child(self):
        """
            The main "king" function.
            Calling this function should find the best model configuration,
            and also have trained the weights for good immediate re-use
        :return:
        """

        # List all local variables we're gonna be using
        best_val_loss = np.inf
        biggest_gradient = 0.
        dag_description = "" # Initial dag description is empty!
        dag_list = [] # Initial dag_list is empty

        m = self.X_train.size(0)

        self._write_to_log_histograms()

        for current_epoch in range(ARG.max_epoch):

            # TODO: Do we create a new model for every epoch, or for each "max steps"?
            for minibatch_offset in range(0, m, ARG.shared_max_step * ARG.batch_size):

                _print_to_std_memory_logs(identifier="P0")

                ##############################
                # SAMPLE DAG FROM CONTROLLER #
                ##############################

                dag_description = "0 0 0 1 1 2 1 2 0 2 0 5 1 1 0 6 1 8 1 8 1 8 1"
                dag_list = [int(x) for x in dag_description.split()]

                _print_to_std_memory_logs(identifier="P1")

                # TODO: write a wrapper function for this:
                self.child_model.overwrite_dag(dag_list)

                X_minibatch = self.X_train[
                    minibatch_offset:
                    minibatch_offset + ARG.shared_max_step * ARG.batch_size
                ].detach()
                Y_minibatch = self.Y_train[
                    minibatch_offset:
                    minibatch_offset + ARG.shared_max_step * ARG.batch_size
                ].detach()

                if config['debug_memory']:
                    print("Training size is: ", X_minibatch.size(), " from ", self.X_train.size())

                self.child_wrapper.train(
                    X=X_minibatch.detach(),
                    Y=Y_minibatch.detach()
                )

                _print_to_std_memory_logs(identifier="P2")

                loss = self.get_child_validation_loss()

                # Some ugly stuff on logging
                print("Validation loss: ", loss)
                logging_epoch = max(current_epoch, current_epoch * (self.X_train.size(0) // ARG.shared_max_step))
                eval_idx = (minibatch_offset // ARG.shared_max_step) + logging_epoch
                print("Eval idx is: ",
                      eval_idx,
                      minibatch_offset,
                      ARG.shared_max_step,
                      self.X_train.size(1)
                      )
                tx_writer.add_scalar('loss/child_val_loss', loss, eval_idx)

                biggest_gradient = _check_abs_max_grad(biggest_gradient, self.child_model)
                tx_writer.add_scalar('misc/max_gradient', biggest_gradient, current_epoch)

                if config['debug_print_max_gradient']:
                    print("Biggest gradient is:", biggest_gradient)

                self._write_to_log_histograms()

            if current_epoch > ARG.shared_decay_after:
                new_lr = ARG.shared_lr * (
                        ARG.shared_decay ** (current_epoch - ARG.shared_decay_after)
                )
                print("Updating learning rate to ", new_lr)
                self.child_wrapper.update_lr(new_lr)

            is_best = loss < best_val_loss

            self._save_child_model(
                is_best=is_best,
                loss=loss,
                epoch=current_epoch,
                dag=dag_description,
                filename=dag_description
            )
            # self.load_child_model(model_path="0_0_2_1_1_0_3_3_1_4_0_0_2_n927.torchsave")


if __name__ == "__main__":
    # TODO: Put these functions into a testing suite?
    print("Starting to train the meta model")

    DATA, TARGET = load_dataset(dev=False, dev_size=5000)

    M = DATA.size(0)
    print("So many samples!")

    # TRAIN_OFF = round(M * (11. / 12))
    # VAL_OFF = round(M * (1. / 12))
    TRAIN_OFF = round(M * (1. / 24))
    VAL_OFF = round(M * (1. / 25))

    # The second size element should represent the embedding
    # This is needed for the crossentropy loss function
    X_train = DATA[:TRAIN_OFF]
    Y_train = TARGET[:TRAIN_OFF]
    X_val = DATA[TRAIN_OFF:TRAIN_OFF + VAL_OFF]
    Y_val = TARGET[TRAIN_OFF:TRAIN_OFF + VAL_OFF]
    X_test = DATA[TRAIN_OFF + VAL_OFF:]
    Y_test = TARGET[TRAIN_OFF + VAL_OFF:]

    del DATA
    del TARGET
    gc.collect()
    torch.cuda.empty_cache()

    if config['debug_printbatch']:
        _print_batches(X_train, Y_train)

    if config['debug_printbatch']:
        _print_batches(X_val, Y_val)

    if config['debug_printbatch']:
        _print_batches(X_test, Y_test)

    # print_batches(data, target)

    # print("Input to the meta trainer is: ", data.size(), target.size())

    META_TRAINER = MetaTrainer(
        X_train=X_train,
        Y_train=Y_train,
        X_val=X_val,
        Y_val=Y_val,
        X_test=None,
        Y_test=None
    )

    DAG_DESCRIPTION = "0 0 0 1 1 2 1 2 0 2 0 5 1 1 0 6 1 8 1 8 1 8 1"
    DAG_LIST = [int(x) for x in DAG_DESCRIPTION.split()]

    print("Before creating the train controller and child!")

    # meta_trainer.train_controller_and_child()
    META_TRAINER.train_controller_and_child()