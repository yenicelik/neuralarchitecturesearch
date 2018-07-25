import torch
from torch import optim
from torch import nn

# Assume we have one rnn defined as global
from torch.nn import RNN

losses = []

# Infrastructure to load datafiles and put them in an appropriate structure for the trainer to be read in

# Training hyperparameters are defined here # Copy these values from the paper proposed to replicate the PennTreeDatabank example
TRAIN_PARA = {

    # Hyperparameters
    'lr': 0.0005,
    'epochs': 100,

    # Optimizers and functions
    'optimizer': optim.SGD,
    'criterion': nn.NLLLoss(),
}

# Defining peripheral functions, such as loss function etc.

#
def train(cell, input_line_tensor, target_line_tensor):
    """
        We manually unroll the RNN depending on the input.
    :return:
    """

    # Reshape targets etc. # TODO: is it correct that we always only have 1D input?
    target_line_tensor.unsqueeze(-1)


    # Initilaize matrices and reset the gradients
    hidden = cell.Hidden()
    cell.zero_grad()

    # Initialize other values
    loss = 0.

    # Apply the training steps
    for i in range(input_line_tensor.size(0)):
        output, hidden = cell(category_tensor, input_line_tensor[i], hidden)
        l = TRAIN_PARA['criterion'](output, target_line_tensor[i])
        loss += l

    loss.backward()

    for p in cell.parameters():
        p.data.add_(-TRAIN_PARA['lr'], p.grad.data)

    return output, loss.item() / input_line_tensor.size(0)

if __name__ == "__main__":
    print("Passing the spawned child network with the test data")

    # Spawn random tensors as input

    # Spawn a vanilla RNN cell for inputs
    rnn = RNN(n_letters, 128, n_letters)

    # Do some training steps just to check syntactic integrity
    train()