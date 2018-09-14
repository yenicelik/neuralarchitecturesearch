"""
    This is one example RNN which uses the interface given by the Base class.
    We will use the RNN cell as this provides the right interface for the NAS we will later implement
"""

import numpy as np
import torch
from torch import nn
from torchviz import make_dot
from torch.autograd import Variable

from src.child.networks.rnn.Base import dlxRNNModelBase
from src.child.networks.rnn.viz_utils.dag_to_graph import draw_network
from src.utils.random_tensor import random_tensor

# Import all utils functions, as we're gonna need them
from src.child.networks.rnn.dag_utils.activation_function import get_activation_function, _get_activation_function_name
from src.child.networks.rnn.dag_utils.generate_weights import generate_weights
from src.child.networks.rnn.dag_utils.identify_loose_ends import identify_loose_ends

GEN_GRAPH = True
if GEN_GRAPH:
    import pygraphviz as pgv
    import matplotlib
    import matplotlib.pyplot as plt

class dlxDAGRNNModule(dlxRNNModelBase):
    """
        We don't need a backward pass, as this is implicitly computed by the forward pass
        -- Write tests if the backward pass actually optimizes the parameters
        The embedding functions are not embeddings per se.
        -> These functions just allow for a linear transformation of the input shape to the hidden shape.
    """

    def _name(self):
        return "dlxDAGRNNModule"

    def build_cell(self, inputx, hidden, dag):
        """
            This function will be used as the cell right away, as pytorch has a dynamical computation graph

        :param description_string: The string which defines the cell of the unrolled RNN in the ENAS paper
        :return:
        """
        # This is not needed for this example network (which uses LSTMs)
        assert isinstance(dag, list), ("DAG is not in the form of a list! ", dag)

        if GEN_GRAPH:
            graph = pgv.AGraph(directed=True, strict=True,
                               fontname='Helvetica', arrowtype='open')  # not work?
            for i in range(0, self.number_of_blocks):
                graph.add_node("Block " + str(i), color='black', fillcolor='pink')

        # print("Building cell")

        # The following dictionary saves the partial of the individual blocks, so we can easily refer to these individual blocks
        partial_outputs = {}

        # The first operation is an activation function
        print("Cell inputs to current block: ", 1)
        first_input = self.embedding_encoder(inputx) + self.weight_hidden2block[1](hidden)
        print("Activation: ", _get_activation_function_name(dag[0]))
        partial_outputs['1'] = get_activation_function(digit=dag[0], inp=first_input)

        # Now apply the ongoing operations
        for i in range(1, self.number_of_blocks): # We start array-indexing with 1, because block 0 refers to the input!
            current_block = i + 1
            previous_block = dag[2*i - 1]
            activation_op = dag[2*i]

            if previous_block == 0:
                print("Cell inputs to current block: ", current_block)
                tmp = self.embedding_encoder(inputx) + self.weight_hidden2block[current_block]( hidden )
                partial_outputs[str(current_block)] = get_activation_function(
                    digit=activation_op,
                    inp=tmp
                )
                print("Activation: ", _get_activation_function_name(activation_op))
                assert partial_outputs[str(current_block)].size() == tmp.size(), ("Not the case!")

                if GEN_GRAPH:
                    graph.add_edge("Block " + str(0), "Block " + str(current_block), label=_get_activation_function_name(activation_op))

            else:
                print("Previous block to current block: ", previous_block, current_block)
                previous_output = partial_outputs[str(previous_block)] # Check if this indexing adds up
                previous_output = self.weight_block2block[previous_block][current_block]( previous_output )
                assert partial_outputs[str(previous_block)].size() == previous_output.size(), ("Not the case!")
                print("Activation: ", _get_activation_function_name(activation_op))
                partial_outputs[str(current_block)] = get_activation_function(
                    digit=activation_op,
                    inp=previous_output
                )

                if GEN_GRAPH:
                    graph.add_edge("Block " + str(previous_block), "Block " + str(current_block), label=_get_activation_function_name(activation_op))

        # Identify the loose ends:
        loose_ends = identify_loose_ends(dag)

        # Return the average of all the loose ends
        outputs = []
        for i in loose_ends:
            outputs.append(partial_outputs[str(i)][None, :])
        averaged_output = torch.cat(outputs, 0)

        # The averaged outputs are the new hidden state now, and we get the logits by decoding it to the dimension of the input
        hidden = torch.mean(averaged_output, dim=0)
        logits = self.embedding_decoder(hidden)

        if GEN_GRAPH:
            print("Printing graph...")
            graph.layout(prog='dot')
            graph.draw('./tmp/cell_viz.png')
            # plt.show()

        # g = make_dot(hidden, params=dict(self.get_all_parameters_with_names()))
        # g.view()
        # exit(0)
        # draw_network(" ".join(dag), "./tmp/")
        # print(self)
        exit(0)

        return logits, hidden

    def __init__(self, dag):
        super(dlxDAGRNNModule, self).__init__()

        assert isinstance(dag, list), ("DAG is not in the form of a list! ", dag)

        self.dag = dag

        # Used probably for every application
        self.embedding_module_encoder = nn.Linear(50, 8)  # 2 words in vocab, 5 dimensional embeddings
        self.embedding_module_decoder = nn.Linear(8, 50)

        self.number_of_blocks = (len(dag) // 2) + 1 # Includes "0" as the first block

        # Spawn all weights here (as these weights will be shared)
        self.weight_hidden2block, self.weight_block2block = generate_weights(8, num_blocks=self.number_of_blocks)

    def embedding_encoder(self, inputx):
        """
            Pass a tensor through an embeddings, such that the shape is appropriate for the LSTM
        :param X:
        :return:
        """
        return self.embedding_module_encoder(inputx)[None, :]

    def embedding_decoder(self, inputx):
        """
                    Pass a tensor through an embeddings, such that the shape is appropriate for the LSTM
                :param X:
                :return:
                """
        return self.embedding_module_decoder(inputx)

    def cell(self, inputx, hidden):
        """
            Use an LSTM as an example cell
        :return:
        """
        if hidden is None: # If hidden is none, then spawn a hidden cell
            hidden = Variable(torch.randn((8,)), requires_grad=True)  # Has size (BATCH, TIMESTEP, SIZE)
        return self.build_cell(inputx, hidden, self.dag)

    def forward(self, X):
        assert len(X.size()) > 2, ("Not enough dimensions! Expected more than 2 dimensions, but have ", X.size())

        time_steps = X.size(0)
        batch_size = X.size(1)

        outputs = []

        # First input to cell
        current_X = X[0, :]
        logit, hidden = self.cell(inputx=current_X, hidden=None)
        outputs.append(logit)

        # Dynamic unrolling of the cell for the rest of the timesteps
        for i in range(1, time_steps):

            current_X = X[i, :]
            logit, hidden = self.cell(inputx=current_X, hidden=hidden)
            outputs.append(logit)

        output = torch.cat(outputs, 0)

        return output


if __name__ == "__main__":
    print("Do a bunch of forward passes: ")

    "0 0 0 1 1 2 1 2 0 2 0 5 1 1 0 6 1 8 1 8 1 8 1"
    "1   2   3   4   5   6   7   8   9   10  11  12 "

    # Example forward pass
    dag_description = "0 0 0 1 1 2 1 2 0 2 0 5 1 1 0 6 1 8 1 8 1 8 1"
    dag_list = [int(x) for x in dag_description.split()]
    print(dag_list)

    # X = random_tensor((5, 4, 50))

    model = dlxDAGRNNModule(dag=dag_list)

    # Test running the cell only:
    # Has shape :: (BATCH, TIMESTEP, SIZE)
    X = torch.randn((5, 4, 50))
    hidden = torch.randn((8,))

    # X = X[0, :]
    # print("X and hidden shapes are: ", X.size(), hidden.size())
    # logit, hidden = model.cell(X, hidden)
    # print("Logit and hidden have shapes: ", logit.size(), hidden.size())


    # Test running the entire forward pass
    model = dlxDAGRNNModule(dag=dag_list)
    Y_hat = model.forward(X)
    print(Y_hat.size())

    # for i in range(100):
    #     model.build_cell(inputx=X, hidden=hidden, dag=dag_list)
