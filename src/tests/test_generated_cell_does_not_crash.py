# Test framework imports
from hypothesis import given, example, settings
import hypothesis.strategies as st

import os
import torch
from torch.autograd import Variable
from torchviz import make_dot

from src.config import config
from src.child._00_DAG_to_NN import spawn_all_weights_if_non_existent, load_weights_from_memory, CustomCell, PARAMETERS

def test_generated_string_does_not_crash():

    dag_descriptions = [
        "0 0 0 1 1 2 1 2 0 2 0 5 1 1 0 6 1 8 1 8 1 8 1",
        # "0 0 2 1 0 1 3 2 2 3 2 4 3 5 3 5 2 7 2 7 3 8 3"
    ]

    for dag_description in dag_descriptions:
        print("Testing", dag_description)
        total_nodes = (len(dag_description) // 4) + 1

        # Generate or Load weights from Memory

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

        child_network_cell = CustomCell(dag_description, weights)

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

        print(y)
        g = make_dot(y)
        g.view()