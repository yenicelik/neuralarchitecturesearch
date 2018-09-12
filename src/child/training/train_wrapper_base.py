"""
    The training wrapper summarises all functions and members that are needed to train a given network on some training data.
    This includes splitting up the data into 1. individual time-sequences and 2. batches
"""

class TrainWrapperBase:

    def __init__(self, model):
        pass

    def train(self, X, Y):
        """
            Trains the model on a certain dataset
        :param X: The data
        :param Y: The shape
        :return:
        """
        raise NotImplementedError
