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