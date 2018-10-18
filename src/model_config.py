import argparse

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

C_CHILD_HIDDEN_SIZE = 0

# NETWORK
net_arg = add_argument_group('Network')
learn_arg = add_argument_group('Learning') # Learning

# Controller configuration
net_arg.add_argument('--num_blocks', type=int, default=12) # 12 # TODO: implement one assert, if this is ok
net_arg.add_argument('--controller_hid', type=int, default=100) #
net_arg.add_argument('--controller_batch_size', type=int, default=1) #
net_arg.add_argument('--num_tokens', type=int, default=4) #

# Paramters for the controller
net_arg.add_argument('--controller_max_step', type=int, default=1000) # TODO: actually 2000
learn_arg.add_argument('--ema_baseline_decay', type=int, default=0.95)
learn_arg.add_argument('--discount', type=float, default=1.0)
learn_arg.add_argument('--controller_lr', type=float, default=3.5e-4, help="will be ignored if --controller_lr_cosine=True")
learn_arg.add_argument('--reward_c', type=float, default=80., help="")

# TODO:
learn_arg.add_argument('--tanh_c', type=float, default=2.5) # such that the controller explores more
learn_arg.add_argument('--softmax_temperature', type=float, default=5.0) # such that


# Shared parameters for the child controller
net_arg.add_argument('--shared_wdrop', type=float, default=0.5)
net_arg.add_argument('--shared_dropout', type=float, default=0.4) # used (output dropout)
net_arg.add_argument('--shared_dropoute', type=float, default=0.1) # used (embedding dropout)
net_arg.add_argument('--shared_dropouti', type=float, default=0.65) # used (input dropout)

net_arg.add_argument('--shared_embed', type=int, default=1000) # TODO: 200, 500, 1000
# net_arg.add_argument('--shared_embed', type=int, default=100) # TODO: 200, 500, 1000
net_arg.add_argument('--shared_hidden', type=int, default=1000)
# net_arg.add_argument('--shared_hidden', type=int, default=100)
net_arg.add_argument('--shared_rnn_max_length', type=int, default=30)

net_arg.add_argument('--shared_tie_weights', type=int, default=1, help="Non-zero value means we tie the weights")
net_arg.add_argument('--shared_grad_clip', type=float, default=0.25)

# TRAINING / TEST
learn_arg.add_argument('--mode', type=str, default='train',
                       choices=['train', 'derive', 'test'],
                       help='train: Training ENAS, derive: Deriving Architectures')
learn_arg.add_argument('--batch_size', type=int, default=64)
learn_arg.add_argument('--test_batch_size', type=int, default=1)
learn_arg.add_argument('--max_epoch', type=int, default=150)
# learn_arg.add_argument('--entropy_mode', type=str, default='reward', choices=['reward', 'regularizer']) # TODO: what is this?

# Shared parameters for the child controller (training)
learn_arg.add_argument('--shared_max_step', type=int, default=50, # 400
                       help='step for shared parameters')
learn_arg.add_argument('--shared_optim', type=str, default='sgd')
learn_arg.add_argument('--shared_lr', type=float, default=20.0)
learn_arg.add_argument('--shared_decay', type=float, default=0.96)
learn_arg.add_argument('--shared_decay_after', type=float, default=15)
learn_arg.add_argument('--shared_l2_reg', type=float, default=1e-7)
learn_arg.add_argument('--shared_init_weight_range_train', type=float, default=0.04) # 0.025
learn_arg.add_argument('--shared_init_weight_range_real_train', type=float, default=0.04)
learn_arg.add_argument('--use_batch_norm', type=int, default=1, help="Whether or not to use batchnorm right after averaging outputs")

# Deriving architectures
learn_arg.add_argument('--derive_num_sample', type=int, default=100)

# Return the arguments
def get_args():
    """Parses all of the arguments above, which mostly correspond to the
    hyperparameters mentioned in the paper.
    """
    args, unparsed = parser.parse_known_args()
    return args, unparsed

ARG, unparsed = get_args()

if __name__ == "__main__":
    print("Printing arguments")
    print(ARG.batch_size)