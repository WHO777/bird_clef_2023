from functools import partial

from torch.optim.lr_scheduler import *


def build_scheduler(config):
    scheduler_type = eval(config.type)
    del config.type
    scheduler_fn = partial(scheduler_type, **config)
    return scheduler_fn
