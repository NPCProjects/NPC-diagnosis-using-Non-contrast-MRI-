import random
import pickle
import numpy as np
import torch

M = 2**32 - 1

def init_fn(worker):
    seed = torch.LongTensor(1).random_().item()
    seed = (seed + worker) % M
    np.random.seed(seed)
    random.seed(seed)


def sample(x, size):
    i = random.sample(range(x.shape[0]), size)
    return torch.tensor(x[i], dtype=torch.int16)


_shape  = (240, 240, 155)


_zero = torch.tensor([0])




