"""
    This file allows us to retrieve the activation function
"""

import torch
from torch import nn

def get_activation_function(digit, inp):
    """
        In the dag, each odd element refers to the activation function.
        The following enumeration is the code:
    :param digit:
    :param inp:
    :return:
    """
    if digit == 0:
        return nn.Tanh()(inp)
    elif digit == 1:
        return nn.ReLU()(inp)
    elif digit == 2:
        return inp
    elif digit == 3:
        return nn.Sigmoid()(inp)
    else:
        raise Exception("The activation digit is not between 0 and 3! ", digit)

def _get_activation_function_name(digit):
    if digit == 0:
        return "tanh"
    elif digit == 1:
        return "relu"
    elif digit == 2:
        return "id"
    elif digit == 3:
        return "sigmoid"
    else:
        raise Exception("The activation digit is not between 0 and 3! ", digit)



