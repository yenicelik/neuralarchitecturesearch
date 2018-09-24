"""
    Includes tools that allow to semantically debug the program
"""
from src.config import config
from src.model_config import ARG
from src.preprocessor.text import Corpus, batchify

corpus = Corpus(config['basepath'] + "data/ptb/")

print(corpus.test)

def to_word(x):
    return corpus.dictionary.idx2word[x]

def load_dataset(dev=False, dev_size=500):
    batch = batchify(corpus.train, ARG.shared_rnn_max_length+1, use_cuda=config['cuda'])
    print(batch)
    print(batch.size())

    # data, target = meta_trainer._get_batch(batch, 10)
    # print(data.size())
    # print(target.size())
    data = batch[:dev_size, 0:ARG.shared_rnn_max_length, None]
    target = batch[:dev_size, 1:1+ARG.shared_rnn_max_length, None]

    return data, target

def print_batches(X, Y, c_max = 50):
    c = 0
    for idx in range(X.size(0)):
        print([to_word(X[idx][jdx]) for jdx in range(X.size(1))])
        print([to_word(Y[idx][jdx]) for jdx in range(Y.size(1))])
        print("\n")
        c += 1
        if c > c_max:
            break