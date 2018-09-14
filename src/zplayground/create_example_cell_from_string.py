import os
import torch
from torch.autograd import Variable
from torchviz import make_dot

from src.config import config
from src.child._00_DAG_to_NN import spawn_all_weights_if_non_existent, load_weights_from_memory, CustomCell, PARAMETERS


def test_string_to_cell():
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

if __name__ == "__main__":
    test_string_to_cell()
