"""
    This file includes the training, where we iteratively train
    1. the child model
    2. the controller (based on the loss of the child model)
"""
import os
import numpy as np

import torch
import random
import gc
import shutil
from torch.autograd import Variable

import src.child.networks.rnn.dag_rnn as dag_rnn #.dlxDAGRNNModule
import src.child.training.dag_train_wrapper as dag_train_wrapper
from src.config import config, C_DEVICE
from src._training.debug_utils.rnn_debug import print_batches, load_dataset
from src.model_config import ARG
from src.utils.debug_utils.exploding_gradients import _check_abs_max_grad
from src.utils.debug_utils.size_network import memory_usage_resource

from src.utils.debug_utils.tensorboard_tools import tx_writer

# random.seed(a=2)

class MetaTrainer:

    def save_child_model(self, is_best, loss, epoch, dag, filename):



        full_path = config['model_savepath'] + filename.replace(" ", "_") + "_n" + self.nonce + ".torchsave"

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

    def load_child_model(self, model_path):

        full_path = config['model_savepath'] + model_path

        if os.path.isfile(full_path):
            print("=> loading checkpoint ", full_path)
            checkpoint = torch.load(full_path)
            self.child_model.load_state_dict(checkpoint['state_dict'])
            self.child_trainer.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loading checkpoint successful! ")
        else:
            print("=> no checkpoint found at ", full_path)

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

        # del X_train
        # del Y_train
        # del X_val
        # del Y_val
        # del X_test
        # del Y_test

        # Spawn one child model
        self.child_model = dag_rnn.dlxDAGRNNModule()
        self.child_model.to(C_DEVICE)

        self.child_trainer = dag_train_wrapper.DAGTrainWrapper(self.child_model)

    def get_child_validation_loss(self):
        # TODO: Implement
        return random.random()

    def train_controller_and_child(self):
        # Setting up the trainers
        dag_description = "0 0 0 1 1 2 1 2 0 2 0 5 1 1 0 6 1 8 1 8 1 8 1"
        # dag_description = "1 0 3 0 1 1 2 3 0"
        # dag_description = "0 0 2 1 1 0 3 3 1 4 0 0 2"

        print(" Skipping initial validation ")
        # dag_list = [int(x) for x in dag_description.split()]
        # self.child_model.overwrite_dag(dag_list)
        #
        # loss = self.child_trainer.get_loss(self.X_val, self.Y_val)
        # print("Validation loss: ", loss)

        best_val_loss = np.inf
        biggest_gradient = 0.

        for current_epoch in range(ARG.max_epoch):

            # TODO: Do we create a new model for every epoch, or for each "max steps"?
            for minibatch_offset in range(0, self.X_train.size(0), ARG.shared_max_step):

                print("Total memory used (MB): ", memory_usage_resource())

                for obj in gc.get_objects():
                    try:
                        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                            print(type(obj), obj.size())
                    except:
                        pass

                dag_description = "0 0 0 1 1 2 1 2 0 2 0 5 1 1 0 6 1 8 1 8 1 8 1"
                # dag_description = "1 0 3 0 1 1 2 3 0"
                # dag_description = "0 0 2 1 1 0 3 3 1 4 0 0 2"
                dag_list = [int(x) for x in dag_description.split()]

                print("Memory usage P1: ", memory_usage_resource())

                self.child_model.overwrite_dag(dag_list)

                X_minibatch = Variable(self.X_train[
                                  minibatch_offset:
                                  minibatch_offset+ARG.shared_max_step
                              ]).to(C_DEVICE)
                Y_minibatch = Variable(self.Y_train[
                                  minibatch_offset:
                                  minibatch_offset + ARG.shared_max_step
                              ]).to(C_DEVICE)

                print("Training size is: ", X_minibatch.size(), " from ", self.X_train.size())

                print("Memory usage P1: ", memory_usage_resource())

                self.child_trainer.train(
                    X=X_minibatch,
                    Y=Y_minibatch
                )

                print("Memory usage P2: ", memory_usage_resource())

                self.child_model.set_train(is_train=False)

                print("Memory usage P3: ", memory_usage_resource())

                del X_minibatch
                del Y_minibatch
                gc.collect()
                torch.cuda.empty_cache()

                print("Skipping validation loss")
                # loss = self.child_trainer.get_loss(self.X_val, self.Y_val)
                # print("Validation loss: ", loss)
                loss = 0.0

                eval_idx = (minibatch_offset // ARG.shared_max_step) \
                           + max(current_epoch, current_epoch * (self.X_train.size(0) // ARG.shared_max_step))
                print("Eval idx is: ", eval_idx, minibatch_offset, ARG.shared_max_step, self.X_train.size(1))
                tx_writer.add_scalar('loss/child_val_loss', loss, eval_idx)

            if current_epoch > ARG.shared_decay_after:
                new_lr = ARG.shared_lr * ( ARG.shared_decay**(current_epoch-ARG.shared_decay_after) )
                print("Updating learning rate to ", new_lr)
                self.child_trainer.update_lr(new_lr)

            is_best = loss < best_val_loss

            biggest_gradient = _check_abs_max_grad(biggest_gradient, self.child_model)
            tx_writer.add_scalar('misc/max_gradient', biggest_gradient, current_epoch)

            self.save_child_model(is_best=is_best, loss=loss, epoch=current_epoch, dag=dag_description, filename=dag_description)
            # self.load_child_model(model_path="0_0_2_1_1_0_3_3_1_4_0_0_2_n927.torchsave")


if __name__ == "__main__":
    print("Starting to train the meta model")
    # meta_trainer.train()

    # train_off = 9000
    # val_off = 1000

    data, target = load_dataset(dev=False, dev_size=10000)

    m = data.size(0)
    print("So many samples!")

    train_off = round(m * (4/6))
    val_off = round(m * (1/6))

    X_train = data[:train_off]
    Y_train = target[:train_off]
    X_val = data[train_off:train_off+val_off]
    Y_val = target[train_off:train_off+val_off]
    X_test = data[train_off+val_off:]
    Y_test = target[train_off+val_off:]

    del data
    del target
    gc.collect()
    torch.cuda.empty_cache()

    # print_batches(data, target)

    # print("Input to the meta trainer is: ", data.size(), target.size())

    meta_trainer = MetaTrainer(
        X_train=X_train,
        Y_train=Y_train,
        X_val=X_val,
        Y_val=Y_val,
        X_test=None,
        Y_test=None
    )

    dag_description = "0 0 0 1 1 2 1 2 0 2 0 5 1 1 0 6 1 8 1 8 1 8 1"
    dag_list = [int(x) for x in dag_description.split()]

    print("Before creating the train controller and child!")

    # meta_trainer.train_controller_and_child()
    meta_trainer.train_controller_and_child()

    # As a test, run the train_child function with the batchloader)

    # print(to_word)

    # for i in data
