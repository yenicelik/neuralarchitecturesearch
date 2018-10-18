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
from src.child.rnn import dag_rnn  # .dlxDAGRNNModule
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

        # Other misc variables
        self.current_epoch = 0
        self.current_shared_step = 0

    ############################################
    # Anything related to the controller model
    ############################################
    def train_controller_network(self):
        """
            Trains the controller by modifying the architecture for a number of steps.
        :return:
        """

        # 1. Creates a function which take
        def current_reward_function(dag):
            # First, overwrite the dag
            self.child_model.overwrite_dag(dag)
            # Then, calculate the validation loss on this newly dag'ed structure
            reward = ARG.reward_c / self.get_child_validation_loss(fast_calc=True) # TODO: should this be a fast_calculation?
            return reward

        # 2. Train the controller on the given weights
        # (for the given amount of timesteps)
        self.controller_wrapper.train_controller(current_reward_function)

    ############################################
    # Anything related to the child model
    ############################################
    def _overwrite_dag(self, dag):
        """
            Overwrites the dag of the child network
        :return:
        """
        self.child_model.overwrite_dag(dag)

    def train_child_network(self):
        """
            Trains the child network for the specified number of steps
        :dag: The dag which was sampled from the network before
        :return:
        """

        biggest_gradient = 0.

        print("Overwriting the dag")

        m = self.X_train.size(0)

        self._write_to_log_histograms()

        # Update the learning rate of the child network if it is time
        if self.current_epoch > ARG.shared_decay_after:
            new_lr = ARG.shared_lr * (
                    ARG.shared_decay ** (self.current_epoch - ARG.shared_decay_after)
            )
            print("Updating learning rate to ", new_lr)
            self.child_wrapper.update_lr(new_lr)

        # Train the child wrapper for the given number of steps
        # TODO: change the initial and last by a class variable which checks how many steps we have worked through already
        for minibatch_offset in range(0, m, ARG.shared_max_step * ARG.batch_size):

            # For each minibatch, sample a new dag
            dag = self.controller_wrapper.sample_dag()
            self._overwrite_dag(dag)

            _print_to_std_memory_logs(identifier="P0")

            # Sample the respective dataset parts from the training data
            X_minibatch = self.X_train[
                          minibatch_offset:
                          minibatch_offset + ARG.shared_max_step * ARG.batch_size
                          ].detach()
            Y_minibatch = self.Y_train[
                          minibatch_offset:
                          minibatch_offset + ARG.shared_max_step * ARG.batch_size
                          ].detach()

            _print_to_std_memory_logs(identifier="P1")

            self.child_wrapper.train(
                X_minibatch,
                Y_minibatch
            )

            _print_to_std_memory_logs(identifier="P2")

            # Some logging stuff
            loss = self.get_child_validation_loss(fast_calc=True)
            print("Validation loss is: ", loss)
            # TODO: There should be a global logging counter!
            logging_epoch = max(self.current_epoch, self.current_epoch * (m // ARG.shared_max_step))
            eval_idx = (minibatch_offset // ARG.shared_max_step) + logging_epoch
            print("Eval idx is: ",
                  eval_idx,
                  minibatch_offset,
                  ARG.shared_max_step,
                  m
                  )
            tx_writer.add_scalar('loss/child_val_loss', loss, eval_idx)

            biggest_gradient = _check_abs_max_grad(biggest_gradient, self.child_model)
            tx_writer.add_scalar('misc/max_gradient', biggest_gradient, self.current_epoch)

            if config['debug_print_max_gradient']:
                print("Biggest gradient is:", biggest_gradient)

            self._write_to_log_histograms()

    def get_child_validation_loss(self, fast_calc=False):
        """
            Wrapper around the 'get_loss' function in the child model.
            This can take random indices, just because we apply it quite frequently!
        :return:
        """
        rand_length = 3 if fast_calc else 1
        # Choose random indices to feed in to validation loss getter
        ranomd_indices = np.random.choice(
            np.arange(self.X_val.size(0)),
            ARG.shared_max_step * ARG.batch_size // rand_length
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

    def train_joint(self):
        """
            The main function.
            This function jointly trains
                the controller network, and
                the child network iteratively

            Calling this function should find the best model configuration,
            and also have trained the weights for good immediate re-use
        :return:
        """

        best_val_loss = np.inf # Will be iteratively enhanced

        # Sample one initial dag
        self.train_child_network()

        self.train_controller_network()

        # First, train the child model
        for current_epoch in range(ARG.max_epoch):

            self.train_child_network()
            # Calculate the validation accuracy of the model now (to check if training child reduces it)
            loss = self.get_child_validation_loss()
            print("Loss after training child for one epoch is: ", loss)

            # TODO: add this loss to two different graphs:
            # The joint optimization graph,
            # And a child validation accuracy graph

            # Get the validation loss, and see if it is best.
            # If it is best, get the new best loss

            # Second, train the controller
            self.train_controller_network()
            # Calculate the validation accuracy of the model now (to check if training controller reduces it)
            loss = self.get_child_validation_loss()
            print("Loss after training controller for max_timesteps is: ", loss)

            is_best = loss < self.best_val_loss

            # Save the model once the epoch is done
            self._save_child_model(
                is_best=is_best,
                loss=loss,
                epoch=current_epoch,
                dag=self.child_model.dag,
                filename=" ".join(self.child_model.dag)
            )
            # self.load_child_model(model_path="0_0_2_1_1_0_3_3_1_4_0_0_2_n927.torchsave")

        # Finally, return the test accuracy
        test_loss = self.get_child_test_loss()
        print("Final test loss is: {}".format(test_loss))

if __name__ == "__main__":
    # TODO: Put these functions into a testing suite?
    print("Starting to train the meta model")

    DATA, TARGET = load_dataset(dev=False, dev_size=5000)

    M = DATA.size(0)
    print("So many samples!")

    # TRAIN_OFF = round(M * (11. / 12))
    # VAL_OFF = round(M * (1. / 12))
    TRAIN_OFF = round(M * (1. / 124))
    VAL_OFF = round(M * (1. / 125))

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
    META_TRAINER.train_joint()
