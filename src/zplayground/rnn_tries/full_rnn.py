import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import RNN, LSTM

from src.playground.rnn_tries.rnn_tries import CustomRNNCell


class RNNBasedOnCell(nn.Module):

    def __init__(self, cell, input_size, batch_size, hidden_size):
        super(RNNBasedOnCell, self).__init__()

        self.cell = cell
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.input_size = input_size

        # Define the parameters that will be used by the cell
        self.hidden = Variable(torch.zeros(1, batch_size, hidden_size), requires_grad=True)    # Need to add a linear layer that takes the hidden shape to the output (input) shape
        self.output_weights = Variable(torch.zeros(hidden_size, input_size), requires_grad=True)
        torch.nn.init.uniform_(self.output_weights, a=-0.025, b=0.025)

    def reset_hidden(self):
        torch.nn.init.uniform_(self.hidden, a=-0.025, b=0.025)
        return True

    def cell_forward(self, input):
        """
            Makes once forward move with the cell,
            given that the hidden weights do not reset
        :param input:
        :return:
        """
        assert input.shape == (self.batch_size, self.input_size), ("Sizes do not conform!", input.shape, (self.batch_size, self.input_size))
        output, hidden = self.cell.forward(input.view(1, self.batch_size, self.input_size), self.hidden)
        assert list(output.size()) == [1, self.batch_size, self.hidden_size], ("Sizes do not conform!", output.shape, (1, self.batch_size, self.hidden_size))
        return output, hidden

    def forward(self, input, reset_hidden=False):
        """

        :param input:
        :return:
        """
        # Go through the sequence, record the output, and write out the output
        # Pass the hidden element through the elements
        sequence_length = input.shape[0]

        # Reset the hidden layer:
        if reset_hidden:
            self.reset_hidden()

        all_outputs = torch.Tensor(sequence_length, self.batch_size, self.input_size)

        for i in range(sequence_length):
            # Pass input and last state to the RNN
            cur_input = input[i, :, :]
            print("Passing the following input: ", cur_input.shape)
            hidden_output, self.hidden = self.cell_forward(cur_input)

            # Pass out such that output size equals input size
            output = torch.matmul(hidden_output, self.output_weights)
            print("Shape of output is: ", output.shape)
            all_outputs[i, :, :] = output

        print("Outputs have shape: ", all_outputs.shape)

        return all_outputs

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
    # cell = RNN(input_size=input_size, hidden_size=hidden_size)
    # cell = LSTM(input_size=input_size, hidden_size=hidden_size)

    # Create the model to take in the cell
    model = RNNBasedOnCell(
        cell,
        input_size=input_size,
        batch_size=batch_size,
        hidden_size=hidden_size
    )

    # Make a forward pass through the model
    Y_hat = model.forward(X)

    criterion = nn.MSELoss()

    old_weights_hidden = model.hidden.clone()
    old_weights_output = model.hidden.clone()

    optimizer = torch.optim.SGD(
        model.parameters()
        # [model.hidden, model.output_weights, model.cell.parameters()]
        , lr=10)

    print("Model parameters are: ")
    for para in model.parameters():
        print(para)

    optimizer.zero_grad()

    loss = criterion(Y_hat, Y)
    loss.backward()
    optimizer.step()

    print("Model parameters are: ")
    for para in model.parameters():
        print(para)


