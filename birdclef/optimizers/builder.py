from functools import partial

from torch.optim import *


def build_optimizer(config):
    optimizer_type = eval(config.type)
    del config.type
    optimizer_fn = partial(optimizer_type, **config)
    return optimizer_fn
