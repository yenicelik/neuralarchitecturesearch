"""
    This file spawns the graph which will later be used for the visualization
"""
from graphviz import Digraph

def spawn_digraph(dag_string, num_blocks):
    dot = Digraph(comment=dag_string)
    for i in range(num_blocks):
        dot.node('input2hidden')
    # for i in range(num_blocks):


def get_all_parameters_with_names(self):
    out = [('input2hidden', self.embedding_module_encoder)]
    out += [('hidden2input', self.embedding_module_decoder)]
    for idx in range(self.number_of_blocks):
        out += [('hidden2block', self.weight_hidden2block[idx])]
    for idx in range(self.number_of_blocks):
        for jdx in range(idx + 1, self.number_of_blocks):
            out += [('block2block', self.weight_block2block[idx][jdx])]
    return out

class GraphWarpper:

    def __init__(self):
        nodes = {}

