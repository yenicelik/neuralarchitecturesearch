from tensorboardX import SummaryWriter

from src.config import config

tx_writer = SummaryWriter(log_dir=config['tensorboard_savepath'])
tx_counter = [0] # A small hack, where we have a global variable