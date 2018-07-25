import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.autograd import Variable
from torchviz import make_dot

# Helper imports
import math, random


# Helper functions (to synthesize data)
def sine_2(X, signal_freq=60.):
    return (np.sin(2 * np.pi * (X) / signal_freq) + np.sin(4 * np.pi * (X) / signal_freq)) / 2.0

def noisy(Y, noise_range=(-0.05, 0.05)):
    noise = np.random.uniform(noise_range[0], noise_range[1], size=Y.shape)
    return Y + noise

def sample(sample_size):
    random_offset = random.randint(0, sample_size)
    X = np.arange(sample_size)
    Y = noisy(sine_2(X + random_offset))
    return Y


# Definition of the RNN
class RNNNet(nn.Module):

    def __init__(self, hidden_size):
        super(RNNNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution

        # hyperparameters
        self.hidden_size = hidden_size

        self.inp = nn.Linear(1, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, 2, dropout=0.05) # TODO: why is this correct?
        self.out = nn.Linear(hidden_size, 1)

    def step(self, input, hidden=None):
        input = self.inp(input.view(1, -1)).unsqueeze(1)
        output, hidden = self.rnn(input, hidden)
        output = self.out(output.squeeze(1))
        return output, hidden

    def forward(self, inputs, hidden=None, force=True, steps=0):
        if force or steps == 0: steps = len(inputs)
        outputs = Variable(torch.zeros(steps, 1, 1))
        output = None

        # Do the actual forward pass
        for i in range(steps):
            if force or i == 0 or (output is None):
                input = inputs[i]
            else:
                input = output # Not define, but will soon be

            output, hidden = self.step(input, hidden)
            outputs[i] = output

        return outputs, hidden

# Writing a function to train the entire thing above
class Trainer:

    def __init__(self):
        self.epochs = 100
        self.iters = 1
        self.hidden_size = 50

        self.model = RNNNet(self.hidden_size)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

        self.losses = np.zeros(self.epochs)

    def train(self):

        for epoch in range(self.epochs):

            for iter in range(self.iters):

                _inputs = sample(50) ## Some random inputs to sample
                inputs = Variable(torch.from_numpy(_inputs[:-1]).float())
                targets = Variable(torch.from_numpy(_inputs[1:]).float())

                # Use teacher forcing 50% of the time
                force = random.random() < 0.5
                outputs, hidden = self.model(inputs, None, force)

                outputs = outputs.squeeze()

                # Optimizer functions
                self.optimizer.zero_grad()
                loss = self.criterion(outputs, targets)
                loss.backward() # How does this affect the optimizer?
                self.optimizer.step()

                self.losses[epoch] += loss.data[0]

            if epoch > 0:
                print(epoch, loss.data[0])



if __name__ == "__main__":

    trainer = Trainer()

    # Train a few times
    for i in range(100):

        trainer.train()