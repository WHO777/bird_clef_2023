import argparse
import importlib
import os
import sys
import shutil
import copy
from pathlib import Path

import addict
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from birdclef import utils
from birdclef.datasets import builder as datasets_builder
from birdclef.losses import builder as losses_builder
from birdclef.metrics import builder as metrics_builder
from birdclef.models import builder as models_builder
from birdclef.optimizers import builder as optimizers_builder
from birdclef.schedulers import builder as schedulers_builder
from birdclef.callbacks import builder as callbacks_builder
from lib import train_lib

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

OUTPUT_DIR = Path(__file__).parents[1] / 'runs' / 'train'

if LOCAL_RANK not in [-1, 0]:
    null_f = open(os.devnull, 'w')
    sys.stdout = null_f
    sys.stderr = null_f


def train(orig_cfg: addict.Dict):
    cfg = copy.deepcopy(orig_cfg)

    utils.set_seed(cfg.seed)

    output_dir = OUTPUT_DIR / cfg.name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = utils.create_logger(str(output_dir), name='train')

    device = utils.select_device(cfg.device, cfg.batch_size, logger=logger)
    cuda = device.type == 'cuda'
    if LOCAL_RANK != -1:
        assert torch.cuda.device_count(
        ) > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(
            backend="nccl" if dist.is_nccl_available() else "gloo")

    train_ds, val_ds = datasets_builder.get_data(cfg.data)
    train_dataloader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    val_dataloader = torch.utils.data.DataLoader(val_ds,
                                                 batch_size=cfg.batch_size,
                                                 shuffle=False,
                                                 num_workers=cfg.num_workers)

    model = models_builder.build_model(cfg.model)
    model = model.to(device)
    if cfg.half:
        model = model.half()
    if cuda and RANK != -1:
        model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)

    metrics = metrics_builder.build_metrics(cfg.metrics)

    optimizer_fn = optimizers_builder.build_optimizer(cfg.optimizer)
    optimizer = optimizer_fn(model.parameters())

    loss = losses_builder.build_loss(config.loss).to(device)

    scheduler_fn = schedulers_builder.build_scheduler(config.scheduler)
    scheduler = scheduler_fn(optimizer)

    start_epoch = 1
    if orig_cfg.resume:
        if orig_cfg.model.pretrained:
            state_dict = torch.load(orig_cfg.model.pretrained, map_location='cpu')
            if 'optimizer' in state_dict:
                optimizer.load_state_dict(state_dict['optimizer'])
            if 'scheduler' in state_dict:
                scheduler.load_state_dict(state_dict['scheduler'])
            if 'epoch' in state_dict:
                start_epoch = state_dict['epoch'] + 1

    for callback_config in cfg.callbacks:
        if callback_config.type == callbacks_builder.CSVLoggerCallback.__name__:
            callback_config.output_dir = str(output_dir)
        if callback_config.type == callbacks_builder.TensorBoardCallback.__name__:
            log_dir = output_dir / 'tensorboard_logs'
            if not log_dir.is_dir():
                log_dir.mkdir(parents=True, exist_ok=True)
            callback_config.output_dir = str(log_dir)

    callbacks = callbacks_builder.build_callbacks(cfg.callbacks)

    train_lib.train(model,
                    train_dataloader,
                    optimizer,
                    loss,
                    device,
                    output_dir=output_dir,
                    metrics=metrics,
                    callbacks=callbacks,
                    start_epoch=start_epoch,
                    epochs=cfg.epochs,
                    scheduler=scheduler,
                    val_dataloader=val_dataloader,
                    logger=logger)


def gen_exp_name(root_dir):
    exp_name = 'exp'
    output_dir = root_dir / exp_name
    i = 1
    while output_dir.is_dir():
        exp_name = 'exp' + str(i)
        output_dir = root_dir / exp_name
        i += 1
    return exp_name


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--device', default='')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--name')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    config_path = Path(args.config).absolute()
    config_module_path = '.'.join([config_path.parent.name, config_path.stem])
    config = importlib.import_module(str(config_module_path))

    config = {k: v for k, v in vars(config).items() if '_' not in k}

    args.batch_size = args.batch_size or config['data']['global_batch_size']
    args.name = args.name or gen_exp_name(OUTPUT_DIR)

    config.update(vars(args))
    config = addict.Dict(config)

    output_dir = OUTPUT_DIR / args.name
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True)

    shutil.copyfile(str(config_path), OUTPUT_DIR / args.name / 'config.py')

    train(config)
