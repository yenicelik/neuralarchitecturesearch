"""
    This file implements an example loss function
    (which is smaller the further we get to an example string)
"""
import numpy as np
import random

from src.model_config import ARG

def example_loss(inp_dag):
    """
        Is an example loss function which allows us to judge
        if the controller minimizes the loss
    :param dag:
    :return:
    """

    best_dag_description = "0 0 0 1 1 2 1 2 0 2 0 5 1 1 0 6 1 8 1 8 1 8 1"
    best_dag_list = [int(x) for x in best_dag_description.split()]

    assert len(inp_dag) == 2*ARG.num_blocks-1, ("Length not as specified in the config module")
    assert len(inp_dag) == len(best_dag_list)

    print("Getting loss")
    best_dag = np.asarray(best_dag_list)
    inp_dag = np.asarray(inp_dag)

    return np.sum(np.abs(inp_dag - best_dag)) / len(inp_dag)

if __name__ == "__main__":
    print("Starting to get loss")

    dag = [random.randint(1,10) for i in range(2*ARG.num_blocks -1)]

    loss = example_loss(dag)
    print(loss)

