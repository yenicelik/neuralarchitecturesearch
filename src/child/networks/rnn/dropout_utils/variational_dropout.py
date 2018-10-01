"""
    This file implements all functions that use the
    variational dropout module (which is also called Locked Dropout)
"""

import torch
from torch.autograd import Variable
import torch.nn.functional as F


class VariationalDropout(torch.nn.Module):
    # code from https://github.com/salesforce/awd-lstm-lm/blob/master/locked_dropout.py
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m, requires_grad=False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x

def _get_dropped_weights(w_raw, dropout_p, is_training):
    """Drops out weights to implement DropConnect.

    Args:
        w_raw: Full, pre-dropout, weights to be dropped out.
        dropout_p: Proportion of weights to drop out.
        is_training: True iff _shared_ model is training.

    Returns:
        The dropped weights.

    TODO(brendan): Why does torch.nn.functional.dropout() return:
    1. `torch.autograd.Variable()` on the training loop
    2. `torch.nn.Parameter()` on the controller or eval loop, when
    training = False...

    Even though the call to `_setweights` in the Smerity repo's
    `weight_drop.py` does not have this behaviour, and `F.dropout` always
    returns `torch.autograd.Variable` there, even when `training=False`?

    The above TODO is the reason for the hacky check for `torch.nn.Parameter`.
    """
    dropped_w = F.dropout(w_raw, p=dropout_p, training=is_training)

    if isinstance(dropped_w, torch.nn.Parameter):
        dropped_w = dropped_w.clone()

    return dropped_w