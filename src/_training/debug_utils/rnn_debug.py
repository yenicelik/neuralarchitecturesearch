"""
    Includes tools that allow to semantically debug the program
"""

from src.preprocessor.text import Corpus, batchify

corpus = Corpus("/Users/david/neuralarchitecturesearch/data/ptb/")

print(corpus.test)

def to_word(x):
    return corpus.dictionary.idx2word[x]

def load_dataset(dev=False):
    batch = batchify(corpus.train, 11, use_cuda=False)
    print(batch)
    print(batch.size())

    # data, target = meta_trainer._get_batch(batch, 10)
    # print(data.size())
    # print(target.size())
    data = batch[:1000, 0:10, None]
    target = batch[:1000, 1:11, None]

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