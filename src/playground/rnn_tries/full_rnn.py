import torch
from torch import nn
from torch.autograd import Variable

from src.playground.rnn_tries.rnn_tries import CustomRNNCell


class RNNBasedOnCell(nn.Module):

    def __init__(self, cell, input_size=2, batch_size=3, hidden_size=5):
        super(RNNBasedOnCell, self).__init__()

        self.cell = cell
        self.batch_size = batch_size
        self.hidden_size = hidden_size

        # Define the parameters that will be used by the cell
        self.hidden = Variable(torch.zeros(1, batch_size, hidden_size), requires_grad=True)    # Need to add a linear layer that takes the hidden shape to the output (input) shape
        self.output_weights = Variable(torch.zeros(hidden_size, input_size), requires_grad=True)
        torch.nn.init.uniform_(self.output_weights, a=-0.025, b=0.025)
        # y_hat = torch.mm(logits.squeeze(), output_weights)
        # Define the output weights

    def cell_forward(self, input, reset_hidden=False):
        """
            Makes once forward move with the cell,
            given that the hidden weights do not reset
        :param input:
        :return:
        """
        pass

    def forward(self, input):
        """

        :param input:
        :return:
        """
        # Go through the sequence, record the output, and write out the output
        # Pass the hidden element through the elements
        pass

if __name__ == "__main__":
    print("Starting to train the full rnn...")

    # Generate hyperparameters
    batch_size = 3
    input_size = 5
    hidden_size = 7
    seq_length = 11

    # Do some example forward passes
    # Generate random matrices for input:
    # The first element denotes the sequence length. This is necessary for internal pytorch purposes

    # Generate training set
    X = torch.Tensor(seq_length, batch_size, input_size)
    Y = torch.Tensor(seq_length, batch_size, input_size)

    # Initialize the inputs
    torch.nn.init.uniform_(X, a=-0.025, b=0.025)
    torch.nn.init.uniform_(Y, a=-0.025, b=0.025)

    # Make sure the shapes confirm
    assert X.shape == Y.shape, ("Shapes do not conform!", X.shape, Y.shape)

    # Define additional components
    cell = CustomRNNCell(input_size, hidden_size, batch_size)

    # Create the model to take in the cell
    model = RNNBasedOnCell(cell, batch_size, hidden_size)

    # Make a forward pass through the model
    # Y_hat = model.forward(X)








    # logits, new_hidden = cell.forward(inp_x, hidden)
    #
    # y_hat = y_hat.reshape(1, batch_size, input_size)
    #
    # # Some more variables
    # criterion = nn.MSELoss()
    # optimizer = torch.optim.SGD([hidden,], lr=0.01)
    #
    # # Simulate what training would look like:
    # optimizer.zero_grad()
    # print("Predicted outputs")
    # print(y_hat)
    # print("Real outputs")
    # print(y)
    # loss = criterion(y_hat, y)
    # loss.backward()
    # optimizer.step()
    #
    # print("Loss value is: ")
    # print(loss)