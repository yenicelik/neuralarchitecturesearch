"""
    This function identifies all loose ends, given the string description of the network.
    A loose end is defined as a block, which is not part of any previous connections.
"""

import numpy as np

def identify_loose_ends(dag, number_of_blocks):

    assert isinstance(dag, list), ("DAG is not in the form of a list! ", dag)

    # First create a list of all blocks
    all_blocks = np.arange(number_of_blocks+1) # We add + 1, because the blocks are not inclusive!
    used_blocks = [dag[i] for i in range(1, len(dag), 2)]

    # Loose blocks:
    loose_blocks = [x for x in all_blocks if x not in used_blocks]

    return loose_blocks

if __name__ == "__main__":
    print("Starting to identify the blocks")
    dag_description = "0 0 0 1 1 2 1 2 0 2 0 5 1 1 0 6 1 8 1 8 1 8 1"

    dag_list = [int(x) for x in dag_description.split()]
    print(dag_list)

    identify_loose_ends(dag_list, len(dag_list) // 2 + 1)

    # TODO: write tests cases for this thing

