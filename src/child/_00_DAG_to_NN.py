"""
    This file handles all the logic, where we convert the description of a the NN (as a DAG) into a neural network.

    Every even element refers to the node that the current cell connects to.

    We save the weights at a local directory (pickle?), and retrieve them every time we spawn a new child network

    Be sure to distinguish between nodes and hidden states when we design the cell!
"""

# Import activation functions
import torch.nn as nn

# Processes intended for preprocessing
def spawn_all_weights_if_non_existent(total_number_of_nodes):
    """
        We create all the weights that we want to use over time, and later just load these weights.
            W_{ to_node, from_node }
            W_{ 4, 1 }
        denotes the weight that transforms the output of node 1 to the input of node 4
    :param total_number_of_nodes:
    :return:
    """
    pass

# Processes for runtime
def load_weights(to_node, from_node):
    """
        Each weight can be described by the node that the weight originates from, and the node that it destines to. i.e.
            $$ W_{2, 1}^{(h)} $$
        describes the weight that goes from node 2, to node 1
    :param from_node:
    :param to_node:
    :return:
    """

def recognize_loose_ends(net_descr):
    pass

def create_layer_dependent_on_input_x(node, x, hidden_state):
    pass

def create_layer_dependent_on_hidden_layer_hi(node, prev_node):
    pass

def get_activation_function(digit):
    """
        Where every odd element refers to the activation function, with the following coding:
        if self.sample_arc[0] == 0:
          h = tf.tanh(h)
        elif self.sample_arc[0] == 1:
          h = tf.nn.relu(h)
        elif self.sample_arc[0] == 2:
          h = tf.identity(h)
        elif self.sample_arc[0] == 3:
          h = tf.sigmoid(h)

    :param digit:
    :return:
    """
    
    if digit == 0:
        return nn.Tanh
    elif digit == 1:
        return nn.ReLU
    elif digit == 2:
        return (lambda x: x)
    elif digit == 3:
        return nn.Sigmoid
    else:
        raise Exception("The activation digit is not 0, 1, 2, 3! ", digit)


def create_cell_from_string(net_descr):
    """
        Given a digit string, we spawn a cell that the RNN will unroll in the next step

        The description may look as follows:
            0 0 0 1 1 2 1 2 0 2 0 5 1 1 0 6 1 8 1 8 1 8 1
            0 | 0 0 | 1 1 | 2 1 | 2 0 | 2 0 | 5 1 | 1 0 | 6 1 | 8 1 | 8 1 | 8 1

        The first digit describes the activation function applied on the input x and the last state h_{t-1} (e.g.)
            h1 = tanh( x_t * W^{x} + h_{t-1} * W_1^{h} )

        After that, every other pair of digits describes
            1.) the last hidden state to refer to (e.g.)
                h2 = h_1 * W_{2, 1}^{h}
            2.) the activation function to use
                h2 = ReLU( h2 )
            h2 = ReLU( h_1 * W_{2, 1}^{h} )

    :param net_descr:
    :return:
    """

    # Create special operation for first digit

    # Create same operatoins for every other pair of strings
    pass

def create_model_from_cell():
    pass