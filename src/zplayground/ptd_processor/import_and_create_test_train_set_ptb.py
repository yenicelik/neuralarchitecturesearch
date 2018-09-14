################################
# HERE WE WRITE THE CONVERTERS #
################################
import os
import torch as t

import collections


class _Dictionary(object):
    """
        NOTICE: This specific class is only used from within the `Corpus` class and nowhere else.
            As such, we can make this class private.
    """

    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = collections.Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1

        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1

        return token_id

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = _Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))
        self.num_tokens = len(self.dictionary)

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = t.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids


if __name__ == "__main__":
    print("Starting application")

    corpus = Corpus("/Users/david/neuralarchitecturesearch/data/ptb")

    counter_obj = corpus.dictionary.counter

    print(counter_obj)

    # for key, val in counter_obj:
    #     print(
    #         counter_obj.dictionary.idx2word[key], " has ", val, " occurences."
    #     )

    print(corpus.dictionary.idx2word)
    print("\n\n\n NEXT PRINT \n\n\n")
    print(corpus.dictionary.word2idx)
    print("We have so many elements in total: ", len(corpus.dictionary.idx2word))

    # for x in corpus.dictionary.counter:
    #     print(x)

    print("Successfully created the corpus")
