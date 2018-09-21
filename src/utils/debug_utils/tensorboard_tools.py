from tensorboardX import SummaryWriter

tx_writer = SummaryWriter(log_dir="/Users/david/neuralarchitecturesearch/tmp/runs/")
tx_counter = [0] # A small hack, where we have a global variable