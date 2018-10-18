"""
    Some base functionality that just obscurs the main functionality of the code
    within the meta_trainer.py.
    The meta_trainer.py script should still handle all the logic
"""

import os
import random
import shutil

import torch

from src.config import config
from src.utils.debug_utils.tensorboard_tools import tx_writer


class MetaTrainerBase:
    """
        This module includes some funcationality which obscurs the code
        contained in the meta_trainer.py script.
        This functionality includes
            printing (debug),
            saving the child model,
            loading the child model,
            saving the controller model,
            loading the controller model,
            writing the weights to the histograms
    """

    def __init__(self):
        print("Initializing Meta Trainer Base")
        self.nonce = str(random.randint(1, 10000))

        # Child variables
        self.child_model = None
        self.child_wrapper = None

        # Controller variables
        self.controller_model = None
        self.controller_trainer = None

    #################################
    # Anything related to the controller model
    #################################
    def _save_controller_model(self, is_best, loss, epoch, dag, filename):
        """
            Save the child model to the pre-defined directory
        :param is_best:
        :param loss:
        :param epoch:
        :param dag:
        :param filename:
        :return:
        """

        full_path = config['model_savepath'] + "_controller_" + filename.replace(" ", "_") + \
                    "_n" + self.nonce + ".torchsave"

        print("Saving child model!", full_path)

        save_checkpoint = {
            'epoch': epoch,
            'dag': dag,
            'state_dict': self.controller_model.state_dict(),
            'loss': loss,
            'optimizer': self.controller_trainer.optimizer.state_dict(),
            'is_best': is_best
        }

        print("Child model saved to: ", full_path)

        torch.save(save_checkpoint, full_path)

        if is_best:
            print("Copying model to the best copy")
            new_path = full_path + "_controller_" + "_n" + str(random.randint(1, 7)) + "pth.tar"
            shutil.copyfile(full_path, new_path)

    def _load_controller_model(self, model_path):
        """
            Load the child model from the pre-defined directory
        :param model_path:
        :return:
        """

        full_path = config['model_savepath'] + "_controller_" + model_path

        if os.path.isfile(full_path):
            print("=> loading checkpoint ", full_path)
            checkpoint = torch.load(full_path)
            self.controller_model.load_state_dict(checkpoint['state_dict'])
            self.controller_trainer.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loading checkpoint successful! ")
        else:
            print("=> no checkpoint found at ", full_path)

    ##################################
    # Anything related to the child model
    ############################
    def _save_child_model(self, is_best, loss, epoch, dag, filename):
        """
            Save the child model to the pre-defined directory
        :param is_best:
        :param loss:
        :param epoch:
        :param dag:
        :param filename:
        :return:
        """

        full_path = config['model_savepath'] + "_child_" + filename.replace(" ", "_") + \
                    "_n" + self.nonce + ".torchsave"

        print("Saving child model!", full_path)

        save_checkpoint = {
            'epoch': epoch,
            'dag': dag,
            'state_dict': self.child_model.state_dict(),
            'loss': loss,
            'optimizer': self.child_wrapper.optimizer.state_dict(),
            'is_best': is_best
        }

        print("Child model saved to: ", full_path)

        torch.save(save_checkpoint, full_path)

        if is_best:
            print("Copying model to the best copy")
            new_path = full_path + "_child_" + "_n" + str(random.randint(1, 7)) + "pth.tar"
            shutil.copyfile(full_path, new_path)

    def _load_child_model(self, model_path):
        """
            Load the child model from the pre-defined directory
        :param model_path:
        :return:
        """

        full_path = config['model_savepath'] + "_child_"  + model_path

        if os.path.isfile(full_path):
            print("=> loading checkpoint ", full_path)
            checkpoint = torch.load(full_path)
            self.child_model.load_state_dict(checkpoint['state_dict'])
            self.child_wrapper.optimizer.load_state_dict(checkpoint['optimizer'])
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

