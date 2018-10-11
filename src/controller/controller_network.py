import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from src.model_config import ARG


class ControllerLSTM(nn.Module):

    def __init__(self):
        print("Starting the LSTM Controller")

        super(ControllerLSTM, self).__init__()

        # Initialize the internal cell used for this controller LSTM
        self.lstm = torch.nn.LSTMCell(
            ARG.controller_hid,
            ARG.controller_hid
        )

        # Simply have one encoder for each individual sequence item
        self.encoders_activation = {}
        self.encoders_block = {}
        # We don't have an encoder for the very first item!
        for block_idx in range(1, ARG.num_blocks):
            # Embedding from activation function to encoder
            self.encoders_activation[block_idx] = nn.Embedding(
                4,
                ARG.controller_hid
            )

            self.encoders_block[block_idx] = nn.Embedding(
                block_idx,
                ARG.controller_hid
            )

        # Now comes the decoder functions
        # The first decoder is a linear connection to the activation
        self.decoders_activation = {}
        self.decoders_block = {}
        # (this should be shared actually?)
        self.decoders_activation[0] = nn.Linear(
            ARG.controller_hid,
            4,
            bias=False
        )
        # We have one decoder for each block, and one for each activate
        # These can probably be tied together in an efficient manner
        for block_idx in range(1, ARG.num_blocks):

            # Decoder from the hidden state to the activation function
            self.decoders_activation[block_idx] = torch.nn.Linear(
                ARG.controller_hid,
                4,
                bias=False
            )

            # Decoder from the hdiden state to the block deciding function
            self.decoders_block[block_idx] = nn.Linear(
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
            torch.zeros(
                (ARG.controller_batch_size, ARG.controller_hid,)
            ),
            requires_grad=False
        )
        self.initial_hidden = torch.nn.Parameter(
            torch.zeros(
                (ARG.controller_batch_size, ARG.controller_hid,)
            ),
            requires_grad=False
        )
        self.initial_input = torch.nn.Parameter(
            torch.zeros(
                (ARG.controller_batch_size, ARG.controller_hid)
            ),
            requires_grad=False
        )

    def forward_activation(self, inputs, hidden, cell, block_id):
        """
            Do one forward pass with the sampler to get an
            activation function
        :param inputs:
        :param hidden:
        :param cell:
        :param block_id:
        :return:
        """
        # Encode (if not block_id == 0)
        if block_id == 0:
            embedded = inputs
        else:
            embedded = self.encoders_block[block_id](inputs)

        # Run through LSTM
        new_hidden, new_cell = self.lstm(embedded, (hidden, cell))

        # Decode
        logits = self.decoders_activation[block_id](new_hidden)

        return logits, new_hidden, new_cell

    def forward_block(self, inputs, hidden, cell, block_id):
        """
            Do one forward pass with the sampler to get a
            previous block to connect to
        :param inputs:
        :param hidden:
        :param cell:
        :param block_id:
        :return:
        """
        # Is always embedded
        embedded = self.encoders_activation[block_id](inputs)

        # Run through LSTM
        new_hidden, new_cell = self.lstm(embedded, (hidden, cell))

        # Decode
        logits = self.decoders_block[block_id](new_hidden)

        return logits, new_hidden, new_cell

    def get_intermediate_values_from_logits(self, logits):

        print("Size of the input is: ", logits.size())
        assert logits.size(0) == ARG.controller_batch_size, ("The logits do not have BATCH-NUMBER of entries", logits, ARG.batch_size)
        assert len(logits.size()), ("Logits are not 2-dimesnionally shaped!", logits.size())

        # TODO: is dim = the right dimension?
        probabilities = F.softmax(logits, dim=1)
        log_probabilities = F.log_softmax(logits, dim=1)

        # Calculate entropies
        entropy = -torch.sum(
            log_probabilities * probabilities,
            dim=1, keepdim=False
        )

        action = probabilities.multinomial(num_samples=1).data

        # TODO: the following line is unclear to me. What is this supposed to do?
        selected_log_probabilities = log_probabilities.gather(
            1,
            Variable(action) # TODO: requires grad = False?
        )
        # We don't apply squeeze, because in the case of batch_size = 0, it just removes all dimensions
        # These are 2-dimensional, but we need to make these 1-dimensional by default
        selected_log_probabilities = selected_log_probabilities[:, 0]
        action = action[:, 0]

        print("Action is: ", action)
        print("Action size is: ", action.size())

        # TODO: a bunch of assert statements on the sizes of the values


        # We return the entropy, and the selected log probabilities
        return entropy, selected_log_probabilities, action

    def forward(self, inputs):
        """
            Do one forward pass with the sampler.
            Apply the forward and apply a decoder
        :param input:
        :param hidden:
        :return:
        """

        # Have arrays for individual actions (to generate the string from this later)
        activations = []
        previous_blocks = []
        entropy_history = []
        reward_history = []

        dag_ops = []

        # Inputs and hidden should be sampled from zero
        self._reset_initial_states()

        inputs = self.initial_input
        hidden = self.initial_hidden
        cell = self.initial_cell

        # Do one pass through the first activation
        # print("Outside initial cell")
        inputs, hidden, cell = self.forward_activation(inputs, hidden, cell, 0)
        inputs = torch.argmax(inputs, dim=1, keepdim=False)
        # print("Activation outputs are: ", inputs)

        dag_ops.append(inputs.item())

        for block_id in range(1, ARG.num_blocks):

            # print("Block idx is: (block input) ", block_id)
            # First get the previous layer
            inputs, hidden, cell = self.forward_block(inputs, hidden, cell, block_id)

            # Pass the logits through the probability-passing function
            # self.get_intermediate_values_from_logits(logits=inputs)

            # Need to pass logits through the embedding, so take argmax, right?
            inputs = torch.argmax(inputs, dim=1, keepdim=False)

            # print("Block outputs are: ", inputs)
            dag_ops.append(inputs.item())

            # print("Block idx is: (activation input) ", block_id)
            # Second get the activation
            inputs, hidden, cell = self.forward_activation(inputs, hidden, cell, block_id)

            # Pass the logits through the finder function
            self.get_intermediate_values_from_logits(logits=inputs)

            # Need to pass logits through the embedding again
            inputs = torch.argmax(inputs, dim=1, keepdim=False)
            # print("Activation outputs are: ", inputs)
            dag_ops.append(inputs.item())

        return dag_ops

    # def reset_parameters(self):
    #     init_range = 0.1
    #     for param in self.parameters():
    #         param.data.uniform_(-init_range, init_range)
    #     for decoder in self.decoders:
    #         decoder.bias.data.fill_(0)


if __name__ == "__main__":
    print("Starting the controller lstm")
    controller = ControllerLSTM()

    for i in range(10):
        dag = controller.forward(None)
        print(dag)

