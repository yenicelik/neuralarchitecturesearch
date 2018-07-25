"""
    This file handles all the logic, where we convert the description of a the NN (as a DAG) into a neural network.

    Every even element refers to the node that the current cell connects to.

    We save the weights at a local directory (pickle?), and retrieve them every time we spawn a new child network

    Be sure to distinguish between nodes and hidden states when we design the cell!
"""

# Local Library imports

from src.config import config
# from src.child.utils import make_dot

# Python imports
import os
import numpy
import pickle

# Torch imports
import torch
import torch.nn as nn
import torch.legacy.nn as legacy_nn
from torch.autograd import Variable
from torchviz import make_dot

PARAMETERS = {
    'input_size': 32,
    'hidden_size': 64
}


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
    W = {}
    a = -0.025
    b = 0.025

    # Create weights for direct input
    W['x'] = torch.Tensor(
        PARAMETERS['input_size'],
        PARAMETERS['hidden_size']
    )
    torch.nn.init.uniform_(W['x'], a=a, b=b)
    W['x'] = Variable(W['x'], requires_grad=True)

    # Create weights for hidden states transitions
    for i in range(1, total_number_of_nodes):
        print("Creating weight from last hidden state to current node " + str(i))
        # TODO: Is this initialization correct?
        W[str(i)] = torch.Tensor(
            PARAMETERS['hidden_size'],
            PARAMETERS['hidden_size']
        )
        torch.nn.init.uniform_(W[str(i)], a=a, b=b)
        W[str(i)] = Variable(W[str(i)], requires_grad=True)

    # For all input nodes
    for i in range(1, total_number_of_nodes):
        # For all output nodes
        for o in range(1, i):
            name = str(i) + "_from_" + str(o)
            print("Creating weight from " + str(o) + " to " + str(i))
            W[name] = torch.Tensor(
                PARAMETERS['hidden_size'],
                PARAMETERS['hidden_size']
            )
            torch.nn.init.uniform_(W[name], a=a, b=b)
            W[name] = Variable(W[name], requires_grad=True)

    with open(config['datapath_save_weights'] + config['filename_weights'], 'wb') as handle:
        pickle.dump(W, handle)

    return W


def load_weights_from_memory():
    with open(config['datapath_save_weights'] + config['filename_weights'], 'rb') as handle:
        return pickle.load(handle)


# Processes for runtime
def get_weights(to_node, from_node):
    """
        Loads the respective weight from memory
        Each weight can be described by the node that the weight originates from, and the node that it destines to. i.e.
            $$ W_{2, 1}^{(h)} $$
        describes the weight that goes from node 2, to node 1
    :param from_node:
    :param to_node:
    :return:
    """
    name = str(to_node) + "_from_" + str(from_node)
    W = load_weights_from_memory()
    return W[name]


class CustomCell(nn.Module):

    def __init__(self, cell_descr, weights):
        super(CustomCell, self).__init__()

        # TODO: Do some assertion, that the numbers conform with the shape of the network as given by the bitstring
        self.weights = weights

        # Parse string
        ops = cell_descr.split()
        self.ops = [int(x) for x in ops]

        print("The following cell will be used by the spawned child network")
        print(self.ops)

        self.number_of_ops = (len(self.ops) // 2) + 1
        print("Number of operations / nodes found")

    def multiply_weight_dependent_on_input_x(self, current_node, x, hidden_state):
        """
            These multiplications are of type
                h1 = x_t * W^{x} + h_{t-1} * W_1^{h}
                # Later on: h1 = tanh( x_t * W^{x} + h_{t-1} * W_1^{h} )
        :param current_node:
        :param x:
        :param hidden_state:
        :return:
        """
        # Get weight of x input and respective hidden state transition matrix
        W_x = self.weights['x']
        W_h_i = self.weights[str(current_node)]

        out_1 = torch.mm(x, W_x)
        out_2 = torch.mm(hidden_state, W_h_i)

        out = torch.add(out_1, out_2)

        return out

    def multiply_weight_dependent_on_node_i(self, current_node, prev_node_output, prev_node):
        """
            These multiplications are of type
                h2 = h_1 * W_{2, 1}^{h}
                # Later on: h2 = ReLU( h_1 * W_{2, 1}^{h} )
        :param current_node:
        :param prev_node_output:
        :param prev_node:
        :return:
        """
        name = str(current_node) + "_from_" + str(prev_node)
        W_node = self.weights[name]
        out = torch.add(prev_node_output, W_node)
        return out

    def get_activation_function(self, digit, inp):
        """
            Where every odd element refers to the activation function, with the following coding:
            0: Tanh
            1: ReLU
            2: Identity
            3: Sigmoid
        :param digit:
        :return:
        """

        if digit == 0:
            return nn.Tanh()(inp)
        elif digit == 1:
            return nn.ReLU()(inp)
        elif digit == 2:
            # m = legacy_nn.Identity()
            # return m.updateOutput(inp)
            # return (lambda x: x)
            return inp
        elif digit == 3:
            return nn.Sigmoid()(inp)
        else:
            raise Exception("The activation digit is not 0, 1, 2, 3! ", digit)

    def recognize_loose_ends(self):
        ops = self.ops[:]
        used_nodes = [ops[0]]
        used_nodes += ops[1::2]  # Every second number (offset first) denotes which node is used

        # check which ones don't appear anywhere
        all_nodes = list(range(self.number_of_ops))

        unused_nodes = list(filter(lambda x: x not in used_nodes, all_nodes))

        return unused_nodes

    def forward(self, x, h):  # input type previously was (self, *input)
        # Create the forward operation by using the string!
        ops = self.ops[:]  # Copy to make it pass by value

        # a is the dictionary that contains all the operations
        # TOOD: make this a class-wide variable, so we can call it somewhere else aswell!
        a = {}

        # The first operations always acts on the input!
        a['1'] = self.multiply_weight_dependent_on_input_x(1, x, h)
        a['1'] = self.get_activation_function(ops[0], a['1'])

        # print(a['1'])
        # g = make_dot(a['1'])
        # g.view()
        input("Enter...")

        # Iterate over all ops, and add the last layer, and activation function to it
        for node in range(1, self.number_of_ops):
            assert isinstance(node, int), ("Node is not of type int!", node, type(node))
            prev_node = ops[2 * node - 1]
            activation = ops[2 * node]
            print("Currently at node: ", node, prev_node, activation)
            if prev_node == 0:
                a[str(node)] = self.multiply_weight_dependent_on_input_x(node, x, h)
                a[str(node)] = self.get_activation_function(activation, a[str(node)])
            else:
                a[str(node)] = self.multiply_weight_dependent_on_node_i(node, a[str(prev_node)], prev_node)
                a[str(node)] = self.get_activation_function(activation, a[str(node)])

            # print(a[str(node)])
            # g = make_dot(a[str(node)])
            # g.view()
            # input("Press Enter to continue...")

        # Identify loose ends
        loose_ends = self.recognize_loose_ends()

        # Return the Average of all the loose ends
        outputs = []
        for i in loose_ends:
            outputs.append(a[str(i)])
        concatenated_unconnected_tensors = torch.stack(outputs,
                                                       dim=0)  # TODO: does this mean that the new tensor will be "long" amongst the ifrst dimension

        # Concatenate all loose ends, and take the mean amongst the concatenated dimension
        mean_of_loose_ends = torch.mean(concatenated_unconnected_tensors, dim=0, keepdim=False)

        return mean_of_loose_ends


def create_cell_from_string(cell_descr):
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

    :param cell_descr: The digit-string that describes the composition of the network
    :return:
    """
    ops = cell_descr.split()
    ops = [int(x) for x in ops]
    print(ops)

    # Create special operation for first digit

    # Create same operatoins for every other pair of strings
    pass


def create_model_from_cell():
    """
        Looks like we manually unroll the RNN depending on the input
    :return:
    """
    pass


if __name__ == "__main__":
    digit_string = "0 0 0 1 1 2 1 2 0 2 0 5 1 1 0 6 1 8 1 8 1 8 1"
    print("Starting to create the test network for digit-string: ", digit_string)

    # create_cell_from_string(digit_string)
    #
    total_nodes = (len(digit_string) // 4) + 1
    # W0 = spawn_all_weights_if_non_existent(total_nodes)
    # W1 = load_weights_from_memory()
    #
    # x = list(W0.keys())
    # y = list(W1.keys())
    #
    # assert set(x) == set(y)
    #
    # print("Success1")
    #
    # for i in range(1, total_nodes):
    #     # For all output nodes
    #     for o in range(1, i):
    #         get_weights(i, o)
    # print("Success 2")

    # TODO: Move up "loading weights" by one layer, because we cannot save the weights within this cell!
    # Spawn or Load all weights
    if not os.path.exists(config['datapath_save_weights']):
        os.makedirs(config['datapath_save_weights'])
        weights = spawn_all_weights_if_non_existent(total_nodes)
    else:
        # Maybe assert that there are "number of ops" keys (proportionally)?
        print("Loading weights from memory...")
        weights = load_weights_from_memory()

    # Save the weights here maybe?

    child_network_cell = CustomCell(digit_string, weights)

    print("Success Generating")

    # Generate random matrices:
    last_hidden = torch.Tensor(
            1,
            PARAMETERS['hidden_size']
        )
    inp_x = torch.Tensor(
        1,
        PARAMETERS['input_size']
    )
    torch.nn.init.uniform_(last_hidden, a=-0.025, b=0.025)
    torch.nn.init.uniform_(inp_x, a=-0.025, b=0.025)

    y = child_network_cell(Variable(inp_x, requires_grad=True), Variable(last_hidden, requires_grad=True))

    print("Success forward run")

    print("Creating the graph...")

    print(y)
    g = make_dot(y)
    g.view()
