import torch

def random_tensor(shape):
    inp_x = torch.Tensor(shape[0], shape[1], shape[2]) # Has size (BATCH, TIMESTEP, SIZE)
    torch.nn.init.uniform_(inp_x, a=-0.025, b=0.025)
    return inp_x