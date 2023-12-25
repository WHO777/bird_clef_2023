import contextlib
import functools
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from termcolor import colored


class AverageMeter:
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_checkpoint(module, checkpoint_path, checkpoint_key=None):
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    module.load_state_dict(state_dict[checkpoint_key]
                           if checkpoint_key is not None else state_dict)
    return module


def is_model_parallel(model):
    return type(model) in (nn.parallel.DataParallel,
                           nn.parallel.DistributedDataParallel)


def de_parallel(model):
    return model.module if is_model_parallel(model) else model


@contextlib.contextmanager
def torch_distributed_zero_first(local_rank: int):
    # Decorator to make all processes in distributed training wait for each local_master to do something
    if local_rank not in [-1, 0]:
        dist.barrier(device_ids=[local_rank])
    yield
    if local_rank == 0:
        dist.barrier(device_ids=[0])


def cuda_to_host(data):
    if isinstance(data, dict):
        for k in data:
            data[k] = data[k].detach().cpu().numpy()
    elif isinstance(data, (list, tuple)):
        for i in range(len(data)):
            data[i] = data[i].detach().cpu().numpy()
    else:
        data = data.detach().cpu().numpy()
    return data


def merge_dicts_with_arrays(dicts):
    outputs_history_concat = {}
    for d in dicts:
        for k, v in d.items():
            if k not in outputs_history_concat:
                outputs_history_concat[k] = []
            outputs_history_concat[k].append(v)
    outputs = {k: torch.concat(v) for k, v in outputs_history_concat.items()}
    return outputs


def select_device(device='', batch_size=1, logger=None):
    log_msg = 'used devices: '
    device = device.strip().lower()
    cpu = device == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif device:
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        assert torch.cuda.is_available() and torch.cuda.device_count() == len(device.replace(',', '')), \
            f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"

    if not cpu and torch.cuda.is_available():
        devices = device.split(',') if device else '0'
        n = len(devices)
        if n > 1:
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        arg = 'cuda:0'
        log_msg += 'cuda: ' + device if device else arg
    else:
        arg = 'cpu'
        log_msg += 'cpu'

    if logger is not None:
        logger.info(log_msg)

    return torch.device(arg)


@functools.lru_cache()
def create_logger(output_dir, dist_rank=0, name=''):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)

    # create file handlers
    file_handler = logging.FileHandler(os.path.join(
        output_dir, f'log_rank{dist_rank}.txt'),
                                       mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger
