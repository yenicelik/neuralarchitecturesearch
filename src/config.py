import os
import torch
from sys import platform


config = {}

# Determine if we're on a cluster, on on mac
if platform == "linux" or platform == "linux2":
    config['basepath'] = "/home/david/neuralarchitecturesearch/"
elif platform == "darwin":
    config['basepath'] = "/Users/david/neuralarchitecturesearch/"

config['datapath_save_weights'] = config['basepath'] + "model/weights/"
config['model_savepath'] = config['basepath'] + "model_saves/"
config['tensorboard_savepath'] = config['basepath'] + "model_saves/tensorboard/"

# Create paths if not existent
if not os.path.exists(os.path.dirname(config['datapath_save_weights'])):
    os.makedirs(os.path.dirname(config['datapath_save_weights']))

if not os.path.exists(os.path.dirname(config['model_savepath'])):
    os.makedirs(os.path.dirname(config['model_savepath']))

if not os.path.exists(os.path.dirname(config['tensorboard_savepath'])):
    os.makedirs(os.path.dirname(config['tensorboard_savepath']))

# Constants (used often, and as a variable)

# Checking if the memory leak is only because of cuda!
if torch.cuda.is_available():
    print("Using cuda!")
    C_DEVICE = torch.device('cuda')
    config['cuda'] = True
else:
    print("Using cpu!")
    C_DEVICE = torch.device('cpu')
    config['cuda'] = False

