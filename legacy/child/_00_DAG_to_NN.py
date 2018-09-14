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

class CustomCell(nn.Module):

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
        print("Entering...")
        # input("Enter...")

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

# From the custom cell, we can create a custom NN

if __name__ == "__main__":
    # Spawn the cell by the cell description:
    digit_string = "0 0 0 1 1 2 1 2 0 2 0 5 1 1 0 6 1 8 1 8 1 8 1"
    print("Starting to create the test network for digit-string: ", digit_string)

    total_nodes = (len(digit_string) // 4) + 1

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

    # Generate random matrices for input:
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

    # From here, we apply the new functions

    y = child_network_cell(Variable(inp_x, requires_grad=True), Variable(last_hidden, requires_grad=True))