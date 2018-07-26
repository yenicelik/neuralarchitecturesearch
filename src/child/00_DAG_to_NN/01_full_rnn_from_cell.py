import torch
from torch import nn
from torch.autograd import Variable

class RNNBasedOnCell(nn.Module):
    """
        Assuming we are given a custom cell,
        we can easily generate a RNN wrapper that uses this specifc cell
    """

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

