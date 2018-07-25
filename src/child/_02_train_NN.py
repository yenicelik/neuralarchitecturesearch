import torch
from torch import optim
from torch import nn

# Assume we have one rnn defined as global
from torch.nn import RNN

from src.child._00_DAG_to_NN import PARAMETERS

losses = []

# Infrastructure to load datafiles and put them in an appropriate structure for the trainer to be read in

# Training hyperparameters are defined here # Copy these values from the paper proposed to replicate the PennTreeDatabank example
TRAIN_PARA = {

    # Hyperparameters
    'lr': 0.0005,
    'epochs': 100,
    'batch_size': 8,

    # Optimizers and functions
    'optimizer': optim.SGD,
    'criterion': nn.NLLLoss(),
}

# Define a class that wraps the cell and other properties
class RecurrentNN(nn.Module):

    def __init__(self):
        super(RecurrentNN, self).__init__()

        # hyperparameters
        self.hidden_size = PARAMETERS['hidden_size']
        self.input_size = PARAMETERS['input_size']
        self.output_size = self.input_size

        # TODO: should save these values as weight-saveable values
        self.rnn = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def step(self, input, hidden=None):
        """
            This is one step, for a single timestep (batch and hidden size stays the same)
        :param input:
        :param hidden:
        :return:
        """
        input = input.view(1, -1).unsqueeze(1) # TODO: need to reformat this accordingly
        output, hidden = self.rnn(input, hidden)
        output = self.out(output.squeeze(1))
        return output, hidden

    def forward(self, input, hidden=None, force=True, steps=0):

# Defining peripheral functions, such as loss function etc.

# TODO: enclose this in another module, and call the below steps "forward" with an option to activate backwards aswell!
def train(cell, input_line_tensor, target_line_tensor):
    """
        We manually unroll the RNN depending on the input.
    :return:
    """

    # Reshape targets etc. # TODO: is it correct that we always only have 1D input?
    target_line_tensor.unsqueeze(-1)


    # Initilaize matrices and reset the gradients
    hidden = torch.Tensor(1, TRAIN_PARA['batch_size'], PARAMETERS['hidden_size']) # Has size (BATCH, TIMESTEP, SIZE)
    print("Hidden state has size: ", hidden.shape)

    torch.nn.init.uniform_(hidden, a=-0.025, b=0.025)
    cell.zero_grad()

    # Initialize other values
    loss = 0.

    # Apply the training steps
    for i in range(input_line_tensor.size(0)):
        reshaped_input = input_line_tensor[i, :, :].view(1, -1, PARAMETERS['input_size'])
        reshaped_target = target_line_tensor[i, :, :].view(1 , -1, PARAMETERS['input_size'])

        # Input and target should have same shapes:
        assert reshaped_target.shape == reshaped_input.shape
        print("Target and inputs have shape: ", reshaped_target.shape)

        output, hidden = cell(reshaped_input, hidden)

        print("Outputs have shape: ", output.shape)

        # Multiply the output by a linearity (so we can take the loss
        out = nn.Linear(
            in_features=PARAMETERS['hidden_size'],
            out_features=PARAMETERS['input_size']
        )

        l = TRAIN_PARA['criterion'](out, reshaped_target)
        loss += l

    loss.backward()

    for p in cell.parameters():
        p.data.add_(-TRAIN_PARA['lr'], p.grad.data)

    return output, loss.item() / input_line_tensor.size(0)

if __name__ == "__main__":
    print("Passing the spawned child network with the test data")

    # Spawn random tensors as input
    # Generate random matrices:
    timesteps = 10

    inp_x = torch.Tensor(timesteps, TRAIN_PARA['batch_size'], PARAMETERS['input_size']) # Has size (BATCH, TIMESTEP, SIZE)
    torch.nn.init.uniform_(inp_x, a=-0.025, b=0.025)

    # Spawn a vanilla RNN cell for inputs
    rnn = RNN(input_size=PARAMETERS['input_size'], hidden_size=PARAMETERS['hidden_size'])

    # Do some training steps just to check syntactic integrity
    inp = inp_x[:-1, :, :]
    tar = inp_x[1:, :, :]
    print("Input and target shapes are: ", inp.shape, tar.shape)
    assert inp.shape == tar.shape
    train(rnn, inp, tar)