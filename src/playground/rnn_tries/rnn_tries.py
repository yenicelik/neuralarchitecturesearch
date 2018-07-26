import numpy as np
import torch
from torch import nn
from torch.autograd import Variable


class CustomRNNCell(nn.Module):
    """
        This class just wraps an actual RNN
    """

    def __init__(self, input_size=8, hidden_size=32, batch_size=8):

        super(CustomRNNCell, self).__init__()

        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size

        # For now, we sample the cell to be a standard RNN
        self.cell = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size)

    def forward(self, input, hidden):
        # TODO: does this take into consideraion the batch size?
        output, new_hidden = self.cell.forward(input, hidden)
        return output, new_hidden

if __name__ == "__main__":
    print("Starting to test the RNN tries")

    batch_size = 3
    input_size = 5
    hidden_size = 7

    # Do some example forward passes
    # Generate random matrices for input:
    # The first element denotes the sequence length. This is necessary for internal pytorch purposes
    hidden = Variable(torch.zeros(1, batch_size, hidden_size), requires_grad=True)
    inp_x = torch.Tensor(1, batch_size, input_size)
    y = torch.Tensor(1, batch_size, input_size)

    torch.nn.init.uniform_(hidden, a=-0.025, b=0.025)
    torch.nn.init.uniform_(inp_x, a=-0.025, b=0.025)
    torch.nn.init.uniform_(y, a=-0.025, b=0.025)

    assert inp_x.shape == y.shape, ("Shapes do not conform!", inp_x.shape, y.shape)

    # From here, we apply the forward function
    cell = CustomRNNCell(input_size, hidden_size, batch_size)
    logits, new_hidden = cell.forward(inp_x, hidden)

    # Need to add a linear layer that takes the hidden shape to the output (input) shape
    output_weights = Variable(torch.zeros(hidden_size, input_size), requires_grad=True)
    torch.nn.init.uniform_(output_weights, a=-0.025, b=0.025)

    y_hat = torch.mm(logits.squeeze(), output_weights)
    y_hat = y_hat.reshape(1, batch_size, input_size)

    # Some more variables
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD([hidden,], lr=0.01)

    # Simulate what training would look like:
    optimizer.zero_grad()
    print("Predicted outputs")
    print(y_hat)
    print("Real outputs")
    print(y)
    loss = criterion(y_hat, y)
    loss.backward()
    optimizer.step()

    print("Loss value is: ")
    print(loss)