"""
    This file handles the entire logic that weights are saved an restored
"""
import os
import pickle

# savepath = '/Users/david/neuralarchitecturesearch/tmp/network_weights.pkl'

# def save_weights(weight_dict):
#     with open(WEIGHT_SAVE_PATH, 'wb') as fileptr:
#         pickle.dump(weight_dict, fileptr)
#
# def load_weights(keys_arr):
#     with open(WEIGHT_SAVE_PATH, 'rb') as fileptr:
#         print(fileptr)
#         b = pickle.load(fileptr)
#         hidden2block, block2block = b['hidden2block'], b['hidden2block']
#     return hidden2block, block2block

# def get_weights(hidden_size, num_blocks):
#     """
#         A wrapper that
#         1. Loads the weights, if the weights are available, or
#         2. Generates the weights, if they are not available
#     :param hidde_size:
#     :param num_blocks:
#     :return:
#     """
#     exists = os.path.isfile(WEIGHT_SAVE_PATH)
#     hidden2block, block2block = None, None
#     if exists:
#         print("Loading existing weights! ...")
#         # Load weights from pickle
#         hidden2block, block2block = load_weights()
#
#     if (hidden2block is not None) and (block2block[1].weight.size(0) == hidden_size):
#         weight_are_correct_size = True
#     else:
#         weight_are_correct_size = False
#
#     if (not exists) or (not weight_are_correct_size):
#         print("Generating new weights! ...")
#         hidden2block, block2block = generate_weights(hidden_size, num_blocks)
#         save_weights(hidden2block, block2block)
#
#     print(hidden2block)
#
#     assert hidden2block[1].weight.size(0) == hidden_size
#     assert len(hidden2block) == num_blocks
#
#     return hidden2block, block2block
