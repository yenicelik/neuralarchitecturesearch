"""
    This file includes the training, where we iteratively train
    1. the child model
    2. the controller (based on the loss of the child model)
"""
import torch
from torch.autograd import Variable

from src.child.networks.rnn.dag_rnn import dlxDAGRNNModule
from src.child.training.dag_train_wrapper import DAGTrainWrapper


class MetaTrainer:

    def __init__(self):
        self.max_length = 30
        # self.child = dlxDAGRNNModule()

    def _get_batch(self, source, idx, batch_size=None, volatile=False):
        """
            From the batchifies source, which has shape
                (total_samples, length, **data_size)
            we extract a exactly 'batch_size' many elements
        :param source:
        :param idx:
        :param length:
        :param volatile:
        :return:
        """
        if batch_size is None:
            batch_size = 30

        length = source.size(1) - 1

        data = Variable(source[idx:idx + batch_size, :length])
        target = Variable(source[idx:idx + batch_size, 1:length+1])

        return data, target

    def train_child(self, dag):
        # This should be replaced by batch getters
        # Spawn child trainer and model
        model = dlxDAGRNNModule(dag)
        self.child_trainer = DAGTrainWrapper(model)

        X = torch.randn((401, 4, 50))
        self.child_trainer.train(X[:400, :], X[1:, :])

    def get_child_validation_loss(self):
        pass

    def train_controller_and_child(self):
        # Setting up the trainers
        for current_epoch in range(3):
            dag_description = "0 0 0 1 1 2 1 2 0 2 0 5 1 1 0 6 1 8 1 8 1 8 1"
            dag_list = [int(x) for x in dag_description.split()]
            self.train_child(dag_list)


if __name__ == "__main__":
    print("Starting to train the meta model")
    meta_trainer = MetaTrainer()
    # meta_trainer.train()

    from src.preprocessor.text import Corpus, batchify

    print("Starting corpus: ")
    corpus = Corpus("/Users/david/neuralarchitecturesearch/data/ptb/")
    print(corpus.test)

    batch = batchify(corpus.train, 31, use_cuda=False)
    print(batch)
    print(batch.size())

    data, target = meta_trainer._get_batch(batch, 10)
    print(data.size())
    print(target.size())


    def to_word(x):
        return corpus.dictionary.idx2word[x]


    for idx in range(data.size(0)):
        print([to_word(data[idx][jdx]) for jdx in range(data.size(1))])
        print([to_word(target[idx][jdx]) for jdx in range(target.size(1))])
        print("\n")

    # print(to_word)

    # for i in data
