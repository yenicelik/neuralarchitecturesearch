import os
from sys import platform


config = {}
config['DEV'] = True

# Determine if we're on a cluster, on on mac
if platform == "linux" or platform == "linux2":
    config['basepath'] = "/home/yedavid/BachelorThesis/"
    config['dev'] = False
elif platform == "darwin":
    config['basepath'] = "/Users/david/neuralarchitecturesearch/"
    config['dev'] = True

config['datapath_save_weights'] = config['basepath'] + "model/weights/"
config['filename_weights'] = "weights_dictionary.pkl"
config['model_savepath'] = config['basepath'] + "model_saves/"

# Create paths if not existent
if not os.path.exists(os.path.dirname(config['datapath_save_weights'])):
    os.makedirs(os.path.dirname(config['datapath_save_weights']))

if not os.path.exists(os.path.dirname(config['model_savepath'])):
    os.makedirs(os.path.dirname(config['model_savepath']))
