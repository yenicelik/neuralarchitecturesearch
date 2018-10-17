"""
    This file includes the training, where we iteratively train
    1. the child model
    2. the controller (based on the loss of the child model)

    Some possible DAG descriptions include:
        dag_description = "0 0 2 1 1 0 3 3 1 4 0 0 2"
        dag_description = "1 0 3 0 1 1 2 3 0"
"""
import os
import random
import gc
import shutil
import torch
import numpy as np
from torch.autograd import Variable

from src.child.rnn import dag_rnn# .dlxDAGRNNModule
import src.child.train_wrapper as dag_train_wrapper
from src.config import config, C_DEVICE
from src._training.debug_utils.rnn_debug import load_dataset
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

class MetaTrainer:

    def _save_child_model(self, is_best, loss, epoch, dag, filename):

        full_path = config['model_savepath'] + filename.replace(" ", "_") + \
                    "_n" + self.nonce + ".torchsave"

        print("Saving child model!", full_path)

        save_checkpoint = {
            'epoch': epoch,
            'dag': dag,
            'state_dict': self.child_model.state_dict(),
            'loss': loss,
            'optimizer': self.child_trainer.optimizer.state_dict(),
            'is_best': is_best
        }

        print("Child model saved to: ", full_path)

        torch.save(save_checkpoint, full_path)

        if is_best:
            print("Copying model to the best copy")
            new_path = full_path + "_n" + str(random.randint(1, 7)) + "pth.tar"
            shutil.copyfile(full_path, new_path)

    def _load_child_model(self, model_path):

        full_path = config['model_savepath'] + model_path

        if os.path.isfile(full_path):
            print("=> loading checkpoint ", full_path)
            checkpoint = torch.load(full_path)
            self.child_model.load_state_dict(checkpoint['state_dict'])
            self.child_trainer.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loading checkpoint successful! ")
        else:
            print("=> no checkpoint found at ", full_path)

    def _write_to_log_histograms(self):
        """
            Write the weights as a histogram to the tx_writer
            (which is the tensorboard writer)
        :return:
        """
        if config['debug_weights_histogram']:
            tx_writer.add_histogram('hist/embedding_encoder',
                                    self.child_model.word_embedding_module_encoder
                                    .weight.detach().numpy(), -1)
            tx_writer.add_histogram('hist/embedding_decoder',
                                    self.child_model.word_embedding_module_decoder
                                    .weight.detach().numpy(), -1)
            tx_writer.add_histogram('hist/w_input_to_c',
                                    self.child_model.w_input_to_c
                                    .weight.detach().numpy(), -1)
            tx_writer.add_histogram('hist/w_hidden_to_hidden',
                                    self.child_model.w_previous_hidden_to_h
                                    .weight.detach().numpy(), -1)
            tx_writer.add_histogram('hist/sample_h_weight_sample',
                                    self.child_model.h_weight_block2block[0][1]
                                    .weight.detach().numpy(), -1)
            tx_writer.add_histogram('hist/sample_h_weight_sample_identical',
                                    self.child_model._h_weight_block2block[1]
                                    .weight.detach().numpy(), -1)

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
        self.nonce = str(random.randint(1, 10000))

        self.X_train = Variable(X_train)
        self.Y_train = Variable(Y_train)
        self.X_val = Variable(X_val)
        self.Y_val = Variable(Y_val)
        self.X_test = Variable(X_test)
        self.Y_test = Variable(Y_test)

        # Spawn one child model
        self.child_model = dag_rnn.dlxDAGRNNModule()
        self.child_model.to(C_DEVICE)

        self.child_trainer = dag_train_wrapper.DAGTrainWrapper(self.child_model)

    # Functions about accessing the controller
    # (such that we have well handled setters and getters)

    # Functions about accessing the child model
    # (such that we have well handled setters and getters)

    def get_child_validation_loss(self):
        """
            Gets the validation loss of the child network.
            Will be moved to the child wrapper soon.
        :return:
        """
        # TODO: Implement this in the child wrapper!
        return random.random()


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

        self._write_to_log_histograms()

        for current_epoch in range(ARG.max_epoch):

            # TODO: Do we create a new model for every epoch, or for each "max steps"?
            for minibatch_offset in range(
                    0, self.X_train.size(0), ARG.shared_max_step * ARG.batch_size):

                _print_to_std_memory_logs(identifier="P0")

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

                self.child_trainer.train(
                    X=X_minibatch.detach(),
                    Y=Y_minibatch.detach()
                )

                _print_to_std_memory_logs(identifier="P2")

                # Choose random indices to feed in to validation loss getter
                ranomd_indices = np.random.choice(
                    np.arange(self.X_val.size(0)),
                    ARG.shared_max_step * ARG.batch_size // 3
                )

                with torch.no_grad():
                    loss = self.child_trainer.get_loss(
                        self.X_val[ranomd_indices].detach(),
                        self.Y_val[ranomd_indices].detach()
                    )

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
                self.child_trainer.update_lr(new_lr)

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

    train_off = round(M * (11. / 12))
    val_off = round(M * (1. / 12))

    # The second size element should represent the embedding
    # This is needed for the crossentropy loss function
    X_train = DATA[:train_off]
    Y_train = TARGET[:train_off]
    X_val = DATA[train_off:train_off + val_off]
    Y_val = TARGET[train_off:train_off + val_off]
    X_test = DATA[train_off + val_off:]
    Y_test = TARGET[train_off + val_off:]

    del DATA
    del TARGET
    gc.collect()
    torch.cuda.empty_cache()

    # if config['debug_printbatch']:
    #     print_batches(X_train, Y_train)
    #
    # if config['debug_printbatch']:
    #     print_batches(X_val, Y_val)
    #
    # if config['debug_printbatch']:
    #     print_batches(X_test, Y_test)

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