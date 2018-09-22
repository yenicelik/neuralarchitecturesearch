import os
import pygraphviz as pgv

def add_node(graph, node_id, label, shape='box', style='filled'):
    if label.startswith('x'):
        color = 'white'
    elif label.startswith('h'):
        color = 'skyblue'
    elif label == 'tanh':
        color = 'yellow'
    elif label == 'ReLU':
        color = 'pink'
    elif label == 'identity':
        color = 'orange'
    elif label == 'sigmoid':
        color = 'greenyellow'
    elif label == 'avg':
        color = 'seagreen3'
    else:
        color = 'white'

    if not any(label.startswith(word) for word in  ['x', 'avg', 'h']):
        label = str(label) + "\n(" + str(node_id) + ")"

    graph.add_node(
            node_id, label=label, color='black', fillcolor=color,
            shape=shape, style=style,
    )

def draw_network(dag, path):
    makedirs(os.path.dirname(path))
    graph = pgv.AGraph(directed=True, strict=True,
                       fontname='Helvetica', arrowtype='open') # not work?

    checked_ids = [-2, -1, 0]

    # if -1 in dag:
    #     add_node(graph, -1, 'x[t]')
    # if -2 in dag:
    #     add_node(graph, -2, 'h[t-1]')
    #
    # add_node(graph, 0, dag[-1][0].name)

    for idx in dag:
        for node in dag[idx]:
            if node.id not in checked_ids:
                add_node(graph, node.id, node.name)
                checked_ids.append(node.id)
            graph.add_edge(idx, node.id)

    graph.layout(prog='dot')
    graph.draw(path)

def makedirs(path):
    if not os.path.exists(path):
        print("[*] Make directories : {}".format(path))
        os.makedirs(path)