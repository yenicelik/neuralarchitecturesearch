"""
    This is one example RNN which uses the interface given by the Base class
    We will use the RNN cell as this provides the right interface
    for the NAS we will later implement
"""

import torch
from torch import nn

from src._training.debug_utils.rnn_debug import load_dataset
from src.child.rnn.Base import dlxRNNModelBase
from src.child.rnn.dropout_utils.embedding_dropout import EmbeddingDropout
from src.child.rnn.dropout_utils.variational_dropout import VariationalDropout

# Import all utils functions, as we're gonna need them
from src.child.rnn.dag_utils.activation_function import get_activation_function
from src.child.rnn.dag_utils.generate_weights import generate_weights
from src.child.rnn.viz_utils.viz_utils import _graph_setup, _graph_print_blocks, \
    _graph_add_edge_activation, _graph_add_edge_initial, _graph_add_final_block, _graph_save_to_png
from src.child.rnn.dag_utils.identify_loose_ends import identify_loose_ends

from src.model_config import ARG


class dlxDAGRNNModule(dlxRNNModelBase):
    """
        We don't need a backward pass, as this is implicitly computed by the forward pass
        -- Write tests if the backward pass actually optimizes the parameters
        The embedding functions are not embeddings per se.
        -> These functions just allow for a linear transformation
          of the input shape to the hidden shape.
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

        c_t1 = self.w_input_to_c(inputx)
        # TODO: Is this correctly applied? TODO: should probably be dropconnect
        c_t2 = self.w_dropout(hidden)
        c_t2 = self.w_previous_hidden_to_c(c_t2)
        tmp = c_t1 + c_t2
        c = self.sigmoid(tmp)

        h_t1 = self.w_input_to_h(inputx)
        # TODO: Is this correctly applied? TODO: should probably be dropconnect
        h_t2 = self.w_dropout(hidden)
        h_t2 = self.w_previous_hidden_to_h(h_t2)
        tmp = h_t1 + h_t2
        tmp = act_fun(tmp)

        # Calculate the hidden block output
        t1 = torch.mul(c, tmp)
        t2 = torch.mul(1. - c, hidden)
        return t1 + t2

    def _connect_block(self, internal_hidden, i, j, act_fun):
        """
            # TODO: fix the indecies (as 0 is the one with the first output!
            Connects block i to block j (i->j)
            The smallest block we can connect from is block 0!
                h_j = c_j*f_{ij}{(W^h_{ij}*h_i)} + (1 - c_j)*h_i,
                    where c_j = \sigmoid{(W^c_{ij}*h_i)}
        :param block_output:
        :param i: is the block number that we are connecting from
        :param j: is the block number that we are connecting to
        :return:
        """
        # Calculate the c's
        tmp = self.c_weight_block2block[i][j](internal_hidden)
        c = self.sigmoid(tmp)

        # Calculate the hidden block output
        tmp = self.h_weight_block2block[i][j](internal_hidden)
        tmp = act_fun(tmp)

        t1 = torch.mul(c, tmp)
        t2 = torch.mul(1. - c, internal_hidden)

        return t1 + t2

    def _reset_parameters(self):
        """
            This function resets the parameters to the initial weights.
            This modifies the initial hidden variables, and all the other
            network parameters.
        :return:
        """
        print("Resetting parameters to correct weights")
        init_range = ARG.shared_init_weight_range_train \
            if self.is_train else ARG.shared_init_weight_range_real_train
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)

    def _generate_block_weights(self):
        """
            This function generates the weights, and assigns the
            weights as the network parameters
        :return:
        """

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

        # Registering the parameters as variables
        self._h_weight_block2block = nn.ModuleList([self.h_weight_block2block[idx][jdx]
                                                    for idx in self.h_weight_block2block
                                                    for jdx in self.h_weight_block2block[idx]])
        self._c_weight_block2block = nn.ModuleList([self.c_weight_block2block[idx][jdx]
                                                    for idx in self.c_weight_block2block
                                                    for jdx in self.c_weight_block2block[idx]])

        self._h_weight_hidden2block = torch.nn.Parameter(self.h_weight_hidden2block.weight)
        self._c_weight_hidden2block = torch.nn.Parameter(self.c_weight_hidden2block.weight)

        # The following hidden parameters have size (BATCH, TIMESTEP, SIZE)
        self.hidden = torch.nn.Parameter(
            torch.randn((ARG.shared_hidden,)),
            requires_grad=True
        )

    def build_cell(self, inputx, hidden, dag, GEN_GRAPH=False):
        """
            This function will be used as the cell right away,
            as pytorch has a dynamical computation graph

        :param description_string:
            The string which defines the cell of the unrolled RNN in the ENAS paper
        :return:
        """
        # This is not needed for this example network (which uses LSTMs)
        assert isinstance(dag, list), ("DAG is not in the form of a list! ", dag)
        graph = _graph_setup(GEN_GRAPH)

        # The following dictionary saves the partial of the individual blocks, so we can easily refer to these individual blocks
        partial_outputs = {}

        # We manually unroll the first iteratoin of the loop
        def act_f(x):
            return get_activation_function(digit=dag[0], inp=x)

        #######
        partial_outputs['0'] = self._connect_input(
            inputx=inputx,
            hidden=hidden,
            act_fun=act_f
        )

        _graph_add_edge_initial(GEN_GRAPH, dag, graph)

        # Now apply the ongoing operations
        # We start array-indexing with 1, because block 0 refers to the input!
        for i in range(1, ARG.num_blocks):
            current_block = i
            previous_block = dag[2 * i - 1]
            activation_op = dag[2 * i]
            _graph_print_blocks(GEN_GRAPH, current_block, previous_block, activation_op, i)

            def act_f(x):
                return get_activation_function(digit=activation_op, inp=x)

            previous_output = partial_outputs[str(previous_block)]
            partial_outputs[str(current_block)] = self._connect_block(
                internal_hidden=previous_output,
                i=previous_block,
                j=current_block,
                act_fun=act_f
            )

            assert partial_outputs[str(previous_block)].size() == previous_output.size(), ("Not the case!")

            _graph_add_edge_activation(GEN_GRAPH, current_block, previous_block, activation_op, graph)

        # Identify the loose ends :: Return the average of all the loose endgs
        loose_ends = identify_loose_ends(dag, ARG.num_blocks)
        outputs = []
        for i in loose_ends:
            outputs.append(partial_outputs[str(i)][None, :])
            _graph_add_final_block(GEN_GRAPH, graph, i)

        # The averaged outputs are the new hidden state now,
        # and we get the logits by decoding it to the dimension of the input
        averaged_output = torch.cat(outputs, 0)
        output = torch.mean(averaged_output, dim=0)  # TODO: is it the average, or partial_outputs[str(ARG.num_blocks - 1)]

        # TODO: need to implement batch normalization
        # if self.batch_norm is not None and ARG.use_batch_norm:
        #     output = output.transpose(-1, -2)
        #     output = self.batch_norm(output)
        #     output = output.transpose(-1, -2)

        _graph_save_to_png(GEN_GRAPH, graph)

        return output

    def cell(self, inputx, hidden):
        """
            Use an LSTM as an example cell
        :return:
        """
        if hidden is None:  # If hidden is none, then spawn a hidden cell
            torch.nn.init.uniform_(self.hidden)
            return self.build_cell(inputx, self.hidden, self.dag)
        else:
            return self.build_cell(inputx, hidden, self.dag)

    def overwrite_dag(self, new_dag):
        """
            This operations overwrites the internal dag of the model
            by the given dag description.
        :param new_dag:
        :return:
        """
        assert isinstance(new_dag, list), ("DAG is not in the form of a list! ", new_dag)
        assert len(new_dag) > 0, ("DAG list cannot be empty!", new_dag)
        assert isinstance(new_dag[0], int), ("DAG must have correct input types", new_dag)
        self.dag = new_dag

    def set_train(self, is_train=True):
        """
            This operation sets the model if it is in training mode
            or in testing mode.
            Statistics, such as dropout and batchnormalization
            are affected by this
        :param is_train:
        :return:
        """
        self.is_train = is_train

        # Set pytorch specific train and eval function
        if is_train:
            self.train()
        else:
            self.eval()

        # TODO: apply batchnmorm for training
        # Apply BatchNorm
        # if is_train:
        #     self.batch_norm = nn.BatchNorm1d(ARG.shared_hidden)
        #     self.batch_norm.to(C_DEVICE)
        # else:
        #     self.batch_norm = None

    def __init__(self, ):
        super(dlxDAGRNNModule, self).__init__()

        # Used probably for every application
        self.word_embedding_module_encoder = EmbeddingDropout(
            10000,
            ARG.shared_embed,
            dropout=ARG.shared_dropoute
        )
        self.word_embedding_module_decoder = nn.Linear(ARG.shared_hidden, 10000, bias=False)

        if ARG.shared_tie_weights:
            # Ties the weights, if this is possible
            print("Tying weights!")
            assert ARG.shared_embed == ARG.shared_hidden, (
                "Sizes of hidden and shared weights must be the same!", ARG.shared_embed, ARG.shared_hidden)
            self.word_embedding_module_decoder.weight = self.word_embedding_module_encoder.weight

        self.var_dropout = VariationalDropout()
        self.w_dropout = torch.nn.Dropout(ARG.shared_wdrop) # TODO: should be drop-connect
        self.sigmoid = torch.nn.Sigmoid()

        self._generate_block_weights()

        # These weights are only for the very first block
        self.w_input_to_c = nn.Linear(ARG.shared_embed, ARG.shared_hidden, bias=False)
        self.w_input_to_h = nn.Linear(ARG.shared_embed, ARG.shared_hidden, bias=False)

        # These are the ones that we apply dropconnect (wdrop) to!
        self.w_previous_hidden_to_c = nn.Linear(ARG.shared_hidden, ARG.shared_hidden, bias=False)
        self.w_previous_hidden_to_h = nn.Linear(ARG.shared_hidden, ARG.shared_hidden, bias=False)

        # Register these weights as parameters
        self._w_input_to_c = nn.Parameter(self.w_input_to_c.weight)
        self._w_input_to_h = nn.Parameter(self.w_input_to_h.weight)
        self._w_previous_hidden_to_c = nn.Parameter(self.w_previous_hidden_to_c.weight)
        self._w_previous_hidden_to_h = nn.Parameter(self.w_previous_hidden_to_h.weight)

        # Additional parameters
        self.set_train(is_train=True)
        self._reset_parameters()

        # print("Printing all parameters from this model: ", self.parameters())
        # print(self.h_weight_block2block[0][2].weight)
        # print(torch.sum(self.h_weight_block2block[0][2].weight))
        # exit(0)

    def word_embedding_encoder(self, inputx):
        """
            Applies the word embedding to the ordinal input vector (to the continuous vector)
        :param inputx:
        :return:
        """
        return self.word_embedding_module_encoder(inputx)

    def word_embedding_decoder(self, inputx):
        """
            Applies the word embedding decoder, from the constinuous, to the ordinal input vector
        :param inputx:
        :return:
        """
        # Need to pass through softmax first
        return self.word_embedding_module_decoder(inputx)

    def forward(self, X):
        """
            X must be of the following shape:
                -> (batch_size, time_steps, **data_size)
        :param X:
        :return:
        """

        # TODO: One way to apply drop-connect (drop the individual weight fields)
        # self.h_weight_hidden2block.weight = _get_dropped_weights(
        #     self.h_weight_hidden2block.weight,
        #     ARG.shared_wdrop,
        #     self.training
        # )
        # self.c_weight_hidden2block.weight = _get_dropped_weights(
        #     self.c_weight_hidden2block.weight,
        #     self.training
        # )

        assert len(X.size()) > 2, ("Not enough dimensions! Expected more than 2 dimensions, but have ", X.size())
        # TODO: also include possibility to equal to test-batch_size
        assert X.size(0) == ARG.batch_size, ("Batch size does not match first dimension", X.size(), ARG.batch_size)

        time_steps = X.size(1)

        outputs = []

        # First input to cell
        current_X = X[:, 0]
        embed = self.word_embedding_encoder(current_X)
        embed = self.var_dropout(embed, ARG.shared_dropouti if self.is_train else 0)

        hidden = self.cell(inputx=embed, hidden=None)
        logit = self.word_embedding_decoder(hidden)
        logit = self.var_dropout(logit, ARG.shared_dropout if self.is_train else 0)

        outputs.append(logit)

        # Dynamic unrolling of the cell for the rest of the timesteps
        for i in range(1, time_steps):
            current_X = X[:, i]
            embed = self.word_embedding_encoder(current_X)
            embed = self.var_dropout(embed, ARG.shared_dropouti if self.is_train else 0)

            hidden = self.cell(inputx=embed, hidden=hidden)
            logit = hidden
            logit = self.var_dropout(logit, ARG.shared_dropout if self.is_train else 0)

            logit = self.word_embedding_decoder(logit)
            outputs.append(logit)

        output = torch.cat(outputs, 1)  # Concatenate along the time axis (last axis)

        return output


if __name__ == "__main__":
    import src.child.dag_train_wrapper

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
    CHILD_MODEL = dlxDAGRNNModule()
    CHILD_TRAINER = src.child.dag_train_wrapper.DAGTrainWrapper(CHILD_MODEL)

    DAG_DESCRIPTION = "0 0 0 1 1 2 1 2 0 2 0 5 1 1 0 6 1 8 1 8 1 8 1"
    DAG_LIST = [int(x) for x in DAG_DESCRIPTION.split()]

    DATA, TARGET = load_dataset(dev=True)
    DATA = DATA[:10]
    target = TARGET[:10]

    print("Whats the size of the data", DATA[:20].size())
    # exit(0)

    X = DATA
    Y = TARGET

    CHILD_MODEL.overwrite_dag(DAG_LIST)
    Y_hat = CHILD_MODEL.forward(X)
    print("Y_hat size is: ", Y_hat.size())
    print("Y size is: ", Y.size())
    assert Y_hat.size(0) == X.size(0), ("Sizes dont conform: ", Y_hat.size(), X.size())
    assert Y_hat.size(1) == X.size(1), ("Sizes dont conform: ", Y_hat.size(), X.size())
    # self.child_wrapper.train(self.X_train, self.Y_train)
