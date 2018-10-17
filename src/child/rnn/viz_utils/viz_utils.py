"""
    Includes the entire code needed to
    initiate, add nodes, add edges and save the graph to .png
"""
from src.child.rnn.dag_utils.activation_function import _get_activation_function_name
from src.model_config import ARG

def _graph_setup(GEN_GRAPH):
    """
        Sets up that we can finally write to the graph
    :return:
    """
    if GEN_GRAPH:
        import pygraphviz as pgv
        graph = pgv.AGraph(directed=True, strict=True,
                           fontname='Helvetica', arrowtype='open')  # not work?
        for i in range(0, ARG.num_blocks):
            graph.add_node("Block " + str(i), color='black', shape='box', style='filled', fillcolor='pink')

        return graph
    return None


def _graph_add_edge_block(GEN_GRAPH, dag, graph):
    """
        Adds an edge to the graph that we can print as output (to a .png file)
    :return:
    """
    if GEN_GRAPH:
        print(dag)
        print(1, "Previous block: ", 0, ":: Activation: ", dag[0])
        print("Activation: ", _get_activation_function_name(dag[0]))

        graph.add_edge("Block " + str(0), "Block " + str(1),
                       label=_get_activation_function_name(dag[0]))


def _graph_add_edge_activation(GEN_GRAPH, current_block, previous_block, activation_op, graph):
    """
        Adds an edge to the graph that we can print as the output (to a .png file).
        This function handles the activations
    """
    if GEN_GRAPH:
        graph.add_edge("Block " + str(previous_block), "Block " + str(current_block),
                       label=_get_activation_function_name(activation_op))


def _graph_print_blocks(GEN_GRAPH, current_block, previous_block, activation_op, i):
    """
        Print which previous block we are currently connecting to
    :return:
    """
    if GEN_GRAPH:
        print(current_block, "Previous block: ", previous_block, " (", 2 * i - 1, ")", ":: Activation: ",
              activation_op)


def _graph_add_final_block(GEN_GRAPH, graph, i):
    """
        Print the final loose ends of the blocks
    :return:
    """
    if GEN_GRAPH:
        graph.add_edge("Block " + str(i), "avg")

def _graph_save_to_png(GEN_GRAPH, graph):
    """
        Save the graph to a .png file
    :param self:
    :param GEN_GRAPH:
    :param graph:
    :param i:
    :return:
    """
    if GEN_GRAPH:
        print("Printing graph...")
        graph.layout(prog='dot')
        graph.draw('./tmp/cell_viz.png')