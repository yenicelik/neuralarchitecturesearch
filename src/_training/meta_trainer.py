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

        assert X_train.size(0) == Y_train.size(0)
        assert X_train.size(0) % ARG.batch_size == 0, ("Not batch size compatible")

        if X_test is not None:
            assert X_test.size(0) == Y_test.size(0)
            assert X_test.size(0) % ARG.batch_size == 0, ("Not batch size compatible")

        if X_val is not None:
            assert X_val.size(0) == Y_val.size(0)
            assert X_val.size(0) % ARG.batch_size == 0, ("Not batch size compatible")

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
            self._overwrite_dag(dag)
            # Then, calculate the validation loss on this newly dag'ed structure
            reward = ARG.reward_c / self.get_child_validation_loss(fast_calc=True, reward_calc=True)
            return reward

        print("Training controller!")

        # 2. Train the controller on the given weights
        # (for the given amount of timesteps)
        self.controller_wrapper.train_controller(current_reward_function)

    ############################################
    # Anything related to the child model
    ############################################
    def _overwrite_dag(self, dag, manual=False):
        """
            Overwrites the dag of the child network
        :return:
        """
        if manual:
            new_dag = dag
        else:
            new_dag = [x.item() for x in dag]
        self.child_model.overwrite_dag(new_dag)

    def train_child_network(self, dag_list=None):
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
        # TODO: Do we pass through the entire dataset here?
        # TODO: Should pass through the entire dataset instead!
        for minibatch_offset in range(0, m, ARG.batch_size):

            # For each minibatch, sample a new dag
            if dag_list is None:
                dag = self.controller_wrapper.sample_dag()
                self._overwrite_dag(dag, manual=False)
                print("Current dag is: ", " ".join([str(x.item()) for x in dag]))
            else:
                dag = dag_list
                self._overwrite_dag(dag, manual=True)
                print("Current dag is: ", " ".join([str(x) for x in dag]))

            _print_to_std_memory_logs(identifier="P0")

            # Sample the respective dataset parts from the training data
            X_batch = self.X_train[
                          minibatch_offset:
                          minibatch_offset + ARG.batch_size
                          ].detach()
            Y_batch = self.Y_train[
                          minibatch_offset:
                          minibatch_offset + ARG.batch_size
                          ].detach()

            print("Size of batch: ", X_batch.size())

            _print_to_std_memory_logs(identifier="P1")

            self.child_wrapper.train(
                X_batch,
                Y_batch
            )

            print("Out of training batch..")
            _print_to_std_memory_logs(identifier="P2")

            # Some logging stuff
            loss = self.get_child_validation_loss(fast_calc=True)
            print("Validation loss is: ", loss)
            # TODO: There should be a global logging counter!
            eval_idx = (minibatch_offset // m) + self.current_epoch
            print("Eval idx is: ",
                  eval_idx,
                  minibatch_offset,
                  m
                  )
            tx_writer.add_scalar('child/dag_aftertraining_validation_loss', loss, eval_idx)

            biggest_gradient = _check_abs_max_grad(biggest_gradient, self.child_model)
            tx_writer.add_scalar('misc/max_gradient', biggest_gradient, self.current_epoch)

            if config['debug_print_max_gradient']:
                print("Biggest gradient is:", biggest_gradient)

            self._write_to_log_histograms()

    def get_child_validation_loss(self, fast_calc=False, reward_calc=False):
        """
            Wrapper around the 'get_loss' function in the child model.
            This can take random indices, just because we apply it quite frequently!
        :return:
        """
        print("Inside child validation loss")
        rand_length = 10 if (fast_calc and not config['dummy_debug']) else 1
        # Choose random indices to feed in to validation loss getter
        random_indices = np.random.choice(
            np.arange(self.X_val.size(0)), self.X_val.size(0) // rand_length
        )
        if reward_calc:
            # The paper mentions "In our language model experiment, the reward function is c/valid_ppl,
            # where the perplexity is computed on a minibatch of validation data.
            # I assume that this minibatch can be chosen freely
            random_indices = np.random.choice(
                np.arange(self.X_val.size(0)), ARG.batch_size
            )

        with torch.no_grad():
            loss = self.child_wrapper.get_loss(
                self.X_val[random_indices].detach(),
                self.Y_val[random_indices].detach(),
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

        # First, train the child model
        for current_epoch in range(ARG.max_epoch):

            print("Epoch: ", current_epoch)

            print("Training child network...")
            self.train_child_network()
            # Calculate the validation accuracy of the model now (to check if training child reduces it)
            loss = self.get_child_validation_loss()
            print("Loss after training child for one epoch is: ", loss)

            tx_writer.add_scalar('joint/child_controller', 1, self.current_epoch)

            # The joint optimization graph,
            # And a child validation accuracy graph

            # Get the validation loss, and see if it is best.
            # If it is best, get the new best loss

            print("Training controller network...")
            # Second, train the controller
            self.train_controller_network()
            # Calculate the validation accuracy of the model now (to check if training controller reduces it)
            loss = self.get_child_validation_loss()
            print("Loss after training controller for max_timesteps is: ", loss)

            tx_writer.add_scalar('joint/child_controller', -1, self.current_epoch)

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
            self.current_epoch += 1

        # Finally, return the test accuracy
        test_loss = self.get_child_test_loss()
        print("Final test loss is: {}".format(test_loss))

if __name__ == "__main__":
    # TODO: Put these functions into a testing suite?
    print("Starting to train the meta model")

    DATA, TARGET = load_dataset(dev=False, dev_size=5000)

    M = DATA.size(0)
    print("So many samples!")

    if config['dummy_debug']:
        TRAIN_OFF = (round(M * (1. / 224)) // ARG.batch_size) * ARG.batch_size
        VAL_OFF = (round(M * (1. / 224)) // ARG.batch_size) * ARG.batch_size
    else:
        TRAIN_OFF = (round(M * (11. / 12)) // ARG.batch_size) * ARG.batch_size
        VAL_OFF = (round(M * (1. / 12)) // ARG.batch_size) * ARG.batch_size

    # The second size element should represent the embedding
    # This is needed for the crossentropy loss function
    X_train = DATA[:TRAIN_OFF]
    Y_train = TARGET[:TRAIN_OFF]
    X_val = DATA[TRAIN_OFF:TRAIN_OFF + VAL_OFF]
    Y_val = TARGET[TRAIN_OFF:TRAIN_OFF + VAL_OFF]
    X_test = DATA[TRAIN_OFF + VAL_OFF:]
    Y_test = TARGET[TRAIN_OFF + VAL_OFF:]


    print("Total data size is: ")
    print(X_train.size())
    print(X_val.size())


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
    # META_TRAINER.train_joint()

    ##############################
    # TRAINING CHILD NETWORK
    ##############################
    for current_epoch in range(ARG.max_epoch):
        print("Epoch is: ", current_epoch)
        META_TRAINER.current_epoch = current_epoch
        META_TRAINER.train_child_network()
        # META_TRAINER._overwrite_dag(DAG_LIST, manual=True)
        loss = META_TRAINER.get_child_validation_loss(fast_calc=True)
        print("Loss is: ", META_TRAINER)

