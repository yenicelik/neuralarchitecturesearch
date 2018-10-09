import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

from src.model_config import ARG


class ControllerLSTM(nn.Module):

    def __init__(self):
        print("Starting the LSTM Controller")

        super(ControllerLSTM, self).__init__()

        # Initialize the embedding (from the tokens)
        self.embedding_block = nn.Embedding(
            4,  # TODO: change this to something appropriate
            ARG.controller_hid
        )  # The embedding which specifies the block to which this is connected to
        self.embedding_activation = nn.Embedding(
            ARG.num_blocks,
            ARG.controller_hid
        )
        # TODO: don't quite understand what this embedding is doing

        # Initialize the decoding (to the tokens)

        # Initialize the internal cell used for this controller LSTM
        self.lstm = torch.nn.LSTMCell(
            ARG.controller_hid,
            ARG.controller_hid
        )

        # We have an individual decoder for
        # each individual token

        # Simply have one encoder for each individual sequence item
        self.encoders = {}
        # We don't have an encoder for the very first item!
        for block_idx in range(1, ARG.num_blocks):

            # Embedding from activation function to encoder
            self.encoders[2*block_idx-1] = nn.Embedding(
                4,
                ARG.controller_hid
            )

            # Embedding from previous block to the encoder
            self.econders[2*block_idx] = nn.Embedding(
                block_idx,
                ARG.controller_hid
            )

        self.decoders = {}
        # The first decoder is a linear connection to the activation
        # (this should be shared actually?)
        self.decoders[0] = nn.Linear(
            ARG.controller_hid,
            4,
            bias=False
        )
        # We have one decoder for each block, and one for each activate
        # These can probably be tied together in an efficient manner
        for block_idx in range(1, ARG.num_blocks):

            # Decoder from the hidden state to the activation function
            self.decoders[2*block_idx-1] = torch.nn.Linear(
                ARG.controller_hid,
                4,
                bias=False
            )

            # Decoder from the hdiden state to the block deciding function
            self.decoders[2*block_idx] = nn.Linear(
                ARG.controller_hid,
                block_idx,
                bias=False
            )

        # Tying weights?

    def _reset_parameters(self):
        """
            Resets all the parameters to be optimized
            by the controller. Will need to call this before
            "_reset_initial_states"
        :return:
        """
        init_range = 0.1
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)
        for decoder in self.decoders:
            decoder.bias.data.fill_(0)

    def _reset_initial_states(self):
        """
            The initial input,
            and the initial hidden state
            Hidden state
        :return:
        """

        # The hidden state consists of two items
        self.initial_cell = torch.nn.Parameter(
            torch.randn((ARG.controller_batch_size, ARG.controller_hid,)),
            requires_grad=False
        )
        self.initial_hidden = torch.nn.Parameter(
            torch.randn((ARG.controller_batch_size, ARG.controller_hid,)),
            requires_grad=False
        )
        self.initial_input = torch.nn.Parameter(
            torch.randn((ARG.controller_batch_size, ARG.controller_hid)),
            requires_grad=False
        )

        torch.nn.init.zeros_(self.initial_cell)
        torch.nn.init.zeros_(self.initial_hidden)
        torch.nn.init.zeros_(self.initial_input)

    def forward(self, inputs, hidden, block_id, is_embedded):
        """
            Do one forward pass with the sampler.
            Apply the forward and apply a decoder
        :param input:
        :param hidden:
        :return:
        """

        # Depending on is_embedding,
        # we take the raw inputs vs not the raw inputs
        if not is_embedded:
            # embed = self.encoder(inputs)
            embed = inputs
        else:
            embed = inputs

        # Pass through the lstm
        new_hidden, new_cell = self.lstm(embed, hidden)
        # logits = self.decoders[block_id](new_hidden)  # So we have different decoders for each block!
        logits = new_hidden

        return logits, (new_hidden, new_cell)

    # Sampling logic
    # def _sample_activation(self, input, hidden_h, hidden_c):
    #     logits, hidden, cell = self.lstm(input, (hidden_h, hidden_c))
    #
    #     return logits, hidden, cell
    # def _sample_previous_layer(self, input, hidden_h, hidden_c):
    #     logits, hidden, cell = self.lstm(input, (hidden_h, hidden_c))
    #
    #     # Pass the output through the embedding that samples the previous activation layer
    #     return logits, hidden, cell

    def sample(self, batch_size=1, verbose=True):
        """
            Repeatedly run the forward pass,
            until the corresponding string has been sampled.
            This function makes use of the following functions to handle the logic:
                - _sample_activation
                - _sample_previous_layer

            We iteratively first sample the activation function,
            and then the previous block id

        :return:
        """

        # Reset all outputs and inputs
        activations = []
        prev_nodes = []

        self._reset_initial_states()

        inputs = self.initial_input
        hidden = self.initial_hidden
        cell = self.initial_cell

        for block_idx in range(2 * (ARG.num_blocks - 1) + 1):
            print("Going through the following idx: ", block_idx, ARG.num_blocks)
            print("Shapes: ", inputs.size())
            print("Shapes hidden: ", hidden.size())
            print("Shapes cell: ", cell.size())

            logits, (hidden, cell) = self.forward(
                inputs=inputs,
                hidden=(hidden, cell),
                block_id=block_idx,  # This seems terribly wrong!
                is_embedded=(block_idx == 0)
            )

            print("Logits are: ", logits.size())

        # Initial element gives the activation for the activation function

        # First sample the

    # def forward(self, inputs, hidden=None, force=True, steps=0):
    #
    #     outputs = Variable(torch.zeros(steps, 1, 1))
    #     output = None
    #
    #     # Do the actual forward pass
    #     for i in range(steps):
    #         if force or i == 0 or (output is None):
    #             input = inputs[i

    # def reset_parameters(self):
    #     init_range = 0.1
    #     for param in self.parameters():
    #         param.data.uniform_(-init_range, init_range)
    #     for decoder in self.decoders:
    #         decoder.bias.data.fill_(0)


if __name__ == "__main__":
    print("Starting the controller lstm")
    controller = ControllerLSTM()

    controller.sample()

    print(controller.decoders)
