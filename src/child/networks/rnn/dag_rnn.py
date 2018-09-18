"""
    This is one example RNN which uses the interface given by the Base class.
    We will use the RNN cell as this provides the right interface for the NAS we will later implement
"""

import numpy as np
import torch
from torch import nn
from torchviz import make_dot
from torch.autograd import Variable

from src._training.debug_utils.rnn_debug import load_dataset
from src.child.networks.rnn.Base import dlxRNNModelBase
from src.child.networks.rnn.viz_utils.dag_to_graph import draw_network

# Import all utils functions, as we're gonna need them
from src.child.networks.rnn.dag_utils.activation_function import get_activation_function, _get_activation_function_name
from src.child.networks.rnn.dag_utils.generate_weights import generate_weights
from src.child.networks.rnn.dag_utils.identify_loose_ends import identify_loose_ends

from src.model_config import ARG

class dlxDAGRNNModule(dlxRNNModelBase):
    """
        We don't need a backward pass, as this is implicitly computed by the forward pass
        -- Write tests if the backward pass actually optimizes the parameters
        The embedding functions are not embeddings per se.
        -> These functions just allow for a linear transformation of the input shape to the hidden shape.
    """

    def _name(self):
        return "dlxDAGRNNModule"

    # The following are functions, that are used within build_cell, and build_cell only
    def _connect_input(self, inputx, hidden, act_fun):
        """
                # TODO: modify this equation!
                h_j = c_j*f_{ij}{(W^h_{ij}*h_i)} + (1 - c_j)*h_i,
                    where c_j = \sigmoid{(W^c_{ij}*h_i)}
        :param inputx:
        :param hidden:
        :return:
        """
        # Calculate the c's
        tmp = self.w_input_to_c(inputx) + self.w_previous_hidden_to_c(hidden)
        c = torch.nn.Sigmoid()(tmp)

        tmp = self.w_input_to_h(inputx) + self.w_previous_hidden_to_h(hidden)
        tmp = act_fun(tmp)

        # Calculate the hidden block output
        t1 = torch.mul(c, tmp)
        t2 = torch.mul(1.-c, hidden)
        return t1 + t2

    def _connect_block(self, internal_hidden, i, j, act_fun):
        """
            Connects block i to block j (i->j)
            The smallest block we can connect from is block 0! # TODO: fix the indecies (as 0 is the one with the first output!
                h_j = c_j*f_{ij}{(W^h_{ij}*h_i)} + (1 - c_j)*h_i,
                    where c_j = \sigmoid{(W^c_{ij}*h_i)}
        :param block_output:
        :param i: is the block number that we are connecting from
        :param j: is the block number that we are connecting to
        :return:
        """
        # Calculate the c's
        tmp = self.c_weight_block2block[i][j](internal_hidden)
        c = torch.nn.Sigmoid()(tmp)

        # Calculate the hidden block output
        tmp = self.h_weight_block2block[i][j](internal_hidden)
        tmp = act_fun(tmp)
        t1 = torch.mul(c, tmp)
        t2 = torch.mul(1.-c, internal_hidden)

        return t1 + t2

    def build_cell(self, inputx, hidden, dag, GEN_GRAPH=False):
        """
            This function will be used as the cell right away, as pytorch has a dynamical computation graph

        :param description_string: The string which defines the cell of the unrolled RNN in the ENAS paper
        :return:
        """
        # This is not needed for this example network (which uses LSTMs)
        assert isinstance(dag, list), ("DAG is not in the form of a list! ", dag)

        if GEN_GRAPH:
            import pygraphviz as pgv
            graph = pgv.AGraph(directed=True, strict=True,
                               fontname='Helvetica', arrowtype='open')  # not work?
            for i in range(0, ARG.num_blocks):
                graph.add_node("Block " + str(i), color='black', shape='box', style='filled', fillcolor='pink')

        # print("Building cell")

        # The following dictionary saves the partial of the individual blocks, so we can easily refer to these individual blocks
        partial_outputs = {}

        # The first operation is an activation function
        # print("Cell inputs to current block: ", 1)

        def act_f(x):
            return get_activation_function(digit=dag[0], inp=x)

        # TODO: fix the indices

        #######
        partial_outputs['0'] = self._connect_input(
            inputx=inputx,
            hidden=hidden,
            act_fun=act_f
        )

        if GEN_GRAPH:
            print(dag)
            print(1, "Previous block: ", 0, ":: Activation: ", dag[0])
            print("Activation: ", _get_activation_function_name(dag[0]))

            graph.add_edge("Block " + str(0), "Block " + str(1),
                           label=_get_activation_function_name(dag[0]))

        # Now apply the ongoing operations
        # We start array-indexing with 1, because block 0 refers to the input!
        for i in range(1, ARG.num_blocks):
            current_block = i
            previous_block = dag[2 * i - 1]
            activation_op = dag[2 * i]

            if GEN_GRAPH:
                print(current_block, "Previous block: ", previous_block, " (", 2 * i - 1, ")", ":: Activation: ", activation_op)

            def act_f(x):
                return get_activation_function(digit=activation_op, inp=x)

            previous_output = partial_outputs[str(previous_block)]
            partial_outputs[str(current_block)] = self._connect_block(
                internal_hidden=previous_output,
                i=previous_block,
                j=current_block,
                act_fun=act_f
            )

            # print("Previous block to current block: ", previous_block, current_block)

            assert partial_outputs[str(previous_block)].size() == previous_output.size(), ("Not the case!")
            # print("Activation: ", _get_activation_function_name(activation_op))

            if GEN_GRAPH:
                graph.add_edge("Block " + str(previous_block), "Block " + str(current_block),
                    label=_get_activation_function_name(activation_op))

        # Identify the loose ends:
        loose_ends = identify_loose_ends(dag, ARG.num_blocks)

        # Return the average of all the loose ends
        outputs = []
        for i in loose_ends:
            outputs.append(partial_outputs[str(i)][None, :])
            if GEN_GRAPH:
                graph.add_edge("Block " + str(i), "avg")

        averaged_output = torch.cat(outputs, 0)

        # The averaged outputs are the new hidden state now, and we get the logits by decoding it to the dimension of the input
        last_hidden = partial_outputs[str(ARG.num_blocks-1)]
        output = torch.mean(averaged_output, dim=0)

        if GEN_GRAPH:
            print("Printing graph...")
            graph.layout(prog='dot')
            graph.draw('./tmp/cell_viz.png')

        return output, last_hidden

    def cell(self, inputx, hidden):
        """
            Use an LSTM as an example cell
        :return:
        """
        if hidden is None:  # If hidden is none, then spawn a hidden cell
            hidden = Variable(torch.randn((ARG.shared_hidden,)), requires_grad=True)  # Has size (BATCH, TIMESTEP, SIZE)
        return self.build_cell(inputx, hidden, self.dag)

    def overwrite_dag(self, new_dag):
        assert isinstance(new_dag, list), ("DAG is not in the form of a list! ", new_dag)
        self.dag = new_dag

    def __init__(self,):
        super(dlxDAGRNNModule, self).__init__()

        # Used probably for every application
        self.word_embedding_module_encoder = torch.nn.Embedding(10000, ARG.shared_embed)
        self.word_embedding_module_decoder = nn.Linear(ARG.shared_hidden, 10000)

        # if ARG.shared_tie_weights:
            # print("Tying weights!")
            # Ties the weights, if this is possible
            # self.word_embedding_module_decoder.weight = self.word_embedding_module_encoder.weight

        # Spawn all weights here (as these weights will be shared)
        self.h_weight_hidden2block, self.h_weight_block2block = generate_weights(
            input_size=ARG.shared_embed,
            hidden_size=ARG.shared_hidden,
            num_blocks=ARG.num_blocks
        )
        self.c_weight_hidden2block, self.c_weight_block2block = generate_weights(
            input_size=ARG.shared_embed,
            hidden_size=ARG.shared_hidden,
            num_blocks=ARG.num_blocks
        )

        # These weights are only for the very first block
        self.w_input_to_c = nn.Linear(ARG.shared_embed, ARG.shared_hidden)
        self.w_input_to_h = nn.Linear(ARG.shared_embed, ARG.shared_hidden)
        self.w_previous_hidden_to_c = nn.Linear(ARG.shared_hidden, ARG.shared_hidden)
        self.w_previous_hidden_to_h = nn.Linear(ARG.shared_hidden, ARG.shared_hidden)

    def word_embedding_encoder(self, inputx):
        return self.word_embedding_module_encoder(inputx)

    def word_embedding_decoder(self, inputx):
        # Need to pass through softmax first
        return self.word_embedding_module_decoder(inputx)

    def forward(self, X):
        """
            X must be of the following shape:
                -> (batch_size, time_steps, **data_size)
        :param X:
        :return:
        """
        assert len(X.size()) > 2, ("Not enough dimensions! Expected more than 2 dimensions, but have ", X.size())

        batch_size = X.size(0)
        time_steps = X.size(1)

        outputs = []

        # First input to cell
        current_X = X[:, 0]
        current_X = self.word_embedding_encoder(current_X)
        logit, hidden = self.cell(inputx=current_X, hidden=None)
        logit = self.word_embedding_decoder(logit)
        outputs.append(logit)

        # Dynamic unrolling of the cell for the rest of the timesteps
        for i in range(1, time_steps):
            current_X = X[:, i]
            current_X = self.word_embedding_encoder(current_X)
            logit, hidden = self.cell(inputx=current_X, hidden=hidden)
            logit = self.word_embedding_decoder(logit)
            outputs.append(logit)

        output = torch.cat(outputs, 1) # Concatenate along the time axis
        # Take argmax amongst last axis,

        return output


if __name__ == "__main__":

    import src.child.training.dag_train_wrapper

    print("Do a bunch of forward passes: ")

    # "0 0 0 1 1 2 1 2 0 2 0 5 1 1 0 6 1 8 1 8 1 8 1"
    # "1   2   3   4   5   6   7   8   9   10  11  12 "
    #
    # # Example forward pass
    # dag_description = "0 0 0 1 1 2 1 2 0 2 0 5 1 1 0 6 1 8 1 8 1 8 1"
    # dag_list = [int(x) for x in dag_description.split()]
    # print(dag_list)
    #
    # model = dlxDAGRNNModule(dag=dag_list)
    #
    # # Test running the cell only:
    # # Has shape :: (BATCH, TIMESTEP, SIZE)
    X = torch.randn((5, 4, 50))
    # hidden = torch.randn((8,))
    #
    # # X = X[0, :]
    # # print("X and hidden shapes are: ", X.size(), hidden.size())
    # # logit, hidden = model.cell(X, hidden)
    # # print("Logit and hidden have shapes: ", logit.size(), hidden.size())
    #
    # # Test running the entire forward pass
    # model = dlxDAGRNNModule(dag=dag_list)
    # Y_hat = model.forward(X)
    # print(Y_hat.size())
    #
    # # for i in range(100):
    # #     model.build_cell(inputx=X, hidden=hidden, dag=dag_list)
    child_model = dlxDAGRNNModule()
    child_trainer = src.child.training.dag_train_wrapper.DAGTrainWrapper(child_model)

    dag_description = "0 0 0 1 1 2 1 2 0 2 0 5 1 1 0 6 1 8 1 8 1 8 1"
    dag_list = [int(x) for x in dag_description.split()]

    data, target = load_dataset(dev=True)
    data = data[:10]
    target = target[:10]

    print("Whats the size of the data", data[:20].size())
    # exit(0)

    X = data
    Y = target

    child_model.overwrite_dag(dag_list)
    Y_hat = child_model.forward(X)
    print("Y_hat size is: ", Y_hat.size())
    print("Y size is: ", Y.size())
    assert Y_hat.size(0) == X.size(0), ("Sizes dont conform: ", Y_hat.size(), X.size())
    assert Y_hat.size(1) == X.size(1), ("Sizes dont conform: ", Y_hat.size(), X.size())
    # self.child_trainer.train(self.X_train, self.Y_train)



