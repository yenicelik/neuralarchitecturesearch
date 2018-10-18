"""
    Includes tools that allow to semantically debug the program
"""
import gc
from src.config import config
from src.model_config import ARG
from src.preprocessor.text import Corpus, batchify

print("Loading corpus...")

CORPUS = Corpus(config['basepath'] + "data/ptb/")

def to_word(x):
    """
        Converts a single id representation to the word string
    :param x:
    :return:
    """
    return CORPUS.dictionary.idx2word[x]

# TODO: change this to also load the training, development and test set
def load_dataset(dev=False, dev_size=500, verbose=True):
    """
        Loads the dataset (given a CORPUS), and loads a batch-ifyable version to it
    :param dev:
    :param dev_size:
    :param verbose:
    :return:
    """
    batch = batchify(CORPUS.train, ARG.shared_rnn_max_length + 1, use_cuda=False)

    if dev:
        data = batch[:dev_size, 0:ARG.shared_rnn_max_length, None]
        target = batch[:dev_size, 1:1+ARG.shared_rnn_max_length, None]
    else:
        data = batch[:, 0:ARG.shared_rnn_max_length, None]
        target = batch[:, 1:1+ARG.shared_rnn_max_length, None]

    del CORPUS.train
    del CORPUS.test
    del CORPUS.valid
    gc.collect()

    if config['debug_printbatch']:
        _print_batches(data, target)

    if config['debug_printbatch'] and verbose:
        print(batch)
        print(batch.size())

    return data, target

def _print_batches(X, Y, counter_max=10):
    """
        Prints the batches to check if the data is still correct).
        This is human readable (the output are word strings, not word-ids!)
    :param X:
    :param Y:
    :param c_max:
    :return:
    """

    if config['debug_printbatch']:
        counter = 0
        print("\n\n\n\n###############################")
        print("PRINTING EXAMPLES")

        for idx in range(X.size(0)):
            print([to_word(X[idx][jdx]) for jdx in range(X.size(1))])
            print([to_word(Y[idx][jdx]) for jdx in range(Y.size(1))])
            print("\n")
            counter += 1
            if counter > counter_max:
                break

        print("SIZES ARE: ", X.size(), Y.size())
