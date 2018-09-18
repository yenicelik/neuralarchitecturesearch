"""
    The training wrapper summarises all functions and members that are needed to train a given network on some training data.
    This includes splitting up the data into 1. individual time-sequences and 2. batches
"""

class TrainWrapperBase:

    def __init__(self, model):
        pass

    def predict(self, X, n=1):
        """
            Predicts the next n tokens for a given sequence X.
            By default, it only predicts the very next token
        :param X:
        :param n:
        :return:
        """
        raise NotImplementedError

    def loss(self, X, Y):
        """
            Calculates the loss, which will be fed into
            the controller as a reward
        :param X:
        :param Y:
        :return:
        """
        raise NotImplementedError


    def train(self, X, Y):
        """
            Trains the model on a certain dataset.
            We plan to train the model for a sequence (i.e. sequence to sequence) task.
            X should be of size
        :param X: The data
        :param Y: The shape
        :return:
        """
        raise NotImplementedError
