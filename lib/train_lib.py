import copy
import os
import sys
import psutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from birdclef import utils


def train_one_epoch(epoch,
                    model,
                    dataloader,
                    optimizer,
                    loss_fn,
                    device,
                    metrics=None,
                    scheduler=None):
    rank = int(os.environ.get('RANK', -1))
    world_size = int(os.environ.get('WORLD_SIZE', -1))

    model.train()

    bar = tqdm.tqdm(enumerate(dataloader),
                    total=len(dataloader),
                    bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

    dataset_size = 0

    history = {'loss': 0}
    loss_meter = utils.AverageMeter()
    labels_history, outputs_history = [], []

    for i, data in bar:
        images = data['image'].to(device, non_blocking=True)
        labels = data['label_onehot'].to(device, non_blocking=True)

        batch_size = images.shape[0]

        optimizer.zero_grad()

        outputs = model(images)

        loss = loss_fn(outputs, labels)

        if rank != -1:
            loss *= world_size

        loss.backward()

        optimizer.step()

        dataset_size += batch_size

        loss_meter.update(loss.item(), n=batch_size)
        history['loss'] = loss_meter.avg

        if metrics is not None:
            labels_history.append(data['label_onehot'])
            outputs_history.append(outputs)

        images_width = images.shape[-1]
        images_height = images.shape[-2]
        bar.set_description('{}: {}   {}: {:.3f}   {}: {}'.format(
            'epoch', epoch, 'loss', loss_meter.avg, 'images_size',
            'x'.join([str(images_width), str(images_height)])))

    if scheduler is not None:
        scheduler.step()

    if metrics is not None:
        if isinstance(outputs_history[0], dict):
            outputs = utils.merge_dicts_with_arrays(outputs_history)
        else:
            outputs = torch.concat(outputs_history)
        labels = torch.concat(labels_history)
        for metric in metrics:
            score = metric(labels, outputs)
            history[metric.name] = score

    history['learning_rate'] = optimizer.param_groups[0]['lr']

    return history


@torch.inference_mode()
def val_one_epoch(model, dataloader, loss_fn, device, metrics=None):
    model.eval()

    dataset_size = 0

    labels_history, outputs_history = [], []

    bar = tqdm.tqdm(enumerate(dataloader),
                    total=len(dataloader),
                    bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
                    desc='validating:')

    meters = {'loss': utils.AverageMeter()}
    if metrics is not None:
        metric_names = [metric.name for metric in metrics]
        meters.update({name: utils.AverageMeter() for name in metric_names})

    for i, data in bar:
        images = data['image'].to(device, non_blocking=True)
        labels_onehot = data['label_onehot'].to(device, non_blocking=True)

        batch_size = images.shape[0]

        outputs = model(images)

        loss = loss_fn(outputs, labels_onehot)

        dataset_size += batch_size

        meters['loss'].update(loss.item(), n=batch_size)
        if metrics is not None:
            labels_history.append(labels_onehot)
            outputs_history.append(outputs)

    if metrics is not None:
        if isinstance(outputs_history[0], dict):
            outputs = utils.merge_dicts_with_arrays(outputs_history)
        else:
            outputs = torch.concat(outputs_history)
        labels = torch.concat(labels_history)
        for metric in metrics:
            score = metric(labels, outputs)
            meters[metric.name].update(score, n=1)

    history = {name: meter.avg for name, meter in meters.items()}

    return history


def train(model,
          train_dataloader,
          optimizer,
          loss_fn,
          device,
          output_dir=None,
          metrics=None,
          callbacks=None,
          scheduler=None,
          val_dataloader=None,
          start_epoch=1,
          epochs=50,
          save_weight_freq=1,
          validate_freq=1,
          logger=None):
    if callbacks is not None:
        for callback in callbacks:
            callback.on_train_begin(**locals())

    for epoch in range(start_epoch, start_epoch + epochs + 1):
        if callbacks is not None:
            for callback in callbacks:
                callback.on_epoch_begin(**locals())

        train_history = train_one_epoch(epoch,
                                        model,
                                        train_dataloader,
                                        optimizer,
                                        loss_fn,
                                        device,
                                        metrics=None,
                                        scheduler=scheduler)

        if val_dataloader is not None and epoch % validate_freq == 0:
            val_history = val_one_epoch(model, val_dataloader, loss_fn, device,
                                        metrics=metrics)
        else:
            val_history = {}

        if callbacks is not None:
            for callback in callbacks:
                callback.on_epoch_end(**locals())

        if logger is not None:
            logged_names = [
                'train_' + name if name != 'learning_rate' else name
                for name in train_history.keys()
            ]
            logged_values = list(train_history.values())
            if val_history:
                logged_names += ['val_' + name for name in val_history.keys()]
                logged_values += list(val_history.values())
            max_logged_name_len = max([len(name) for name in logged_names])
            logged_names_str = ('%{}s'.format(max_logged_name_len + 2) *
                                len(logged_names)) % tuple(logged_names)
            logged_values_str = ('%{}.4f'.format(max_logged_name_len + 2) *
                                 len(logged_values)) % tuple(logged_values)
            logger.info(logged_names_str)
            logger.info(logged_values_str)

        if output_dir is not None:
            weights_dir = Path(output_dir) / 'weights'
            if not weights_dir.is_dir():
                weights_dir.mkdir(parents=True)
            if epoch % save_weight_freq == 0:
                de_paralleled_model = utils.de_parallel(model)
                ckpt = {
                    'model': copy.deepcopy(de_paralleled_model.state_dict()),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch
                }
                weights_path = weights_dir / 'last_epoch_{}.pt'.format(
                    str(epoch))
                torch.save(ckpt, weights_path)

    if callbacks is not None:
        for callback in callbacks:
            callback.on_train_end(**locals())
