import os
from pathlib import Path
from typing import Optional, Tuple

import addict
import numpy as np
import pandas as pd
import torch
from sklearn import model_selection

from birdclef.datasets import augmentations, bird_clef_dataset
from birdclef.datasets import utils as dataset_utils


def _get_df_from_root_dir(root_dir):
    metadata_name = 'train_metadata.csv'
    metadata_path = Path(root_dir) / metadata_name

    assert metadata_path.is_file(), 'file {} does not exist in {}.'.format(
        metadata_name, root_dir)

    metadata = pd.read_csv(str(metadata_path))

    class_names = bird_clef_dataset.CLASS_NAMES
    class_labels = list(range(bird_clef_dataset.NUM_CLASSES))

    metadata[
        'filename'] = root_dir + os.sep + 'train_audio' + os.sep + metadata[
            'filename']
    name_to_label = {n: l for n, l in zip(class_names, class_labels)}
    metadata['target'] = metadata.primary_label.map(name_to_label)

    return metadata


def _get_bird_clef_dataset(
    config: addict.Dict
) -> Tuple[Optional[bird_clef_dataset.BirdCLEFDataset],
           Optional[bird_clef_dataset.BirdCLEFDataset]]:
    train_cfg = config.train
    val_cfg = config.val

    train_df = _get_df_from_root_dir(
        train_cfg.root_dir) if train_cfg.root_dir else None
    val_df = _get_df_from_root_dir(
        val_cfg.root_dir) if val_cfg.root_dir else None

    df = train_df

    if config.filter_thresh:
        df = dataset_utils.filter_data(df, thresh=config.filter_thresh)
    else:
        df['cv'] = True

    if train_cfg.root_dir == val_cfg.root_dir:
        cv = bool(config.num_folds)
        if cv:
            skf = model_selection.StratifiedKFold(n_splits=config.num_folds,
                                                  shuffle=True,
                                                  random_state=config.seed)
            df.reset_index(drop=True)
            df['fold'] = -1
            for fold, (train_idx,
                       val_idx) in enumerate(skf.split(df,
                                                       df['primary_label'])):
                df.loc[val_idx, 'fold'] = fold
            train_folds, val_folds = train_cfg.folds, val_cfg.folds
            train_df = df.query(
                'fold == {} | ~cv'.format(train_folds)).reset_index(drop=True)
            val_df = df.query(
                'fold == {} & cv'.format(val_folds)).reset_index(drop=True)
        else:
            val_split = val_cfg.val_split
            assert val_split, 'you have to define "val_split" if you dont use cross validation.'
            num_val_samples = int(len(df) * val_split)
            train_df = df[num_val_samples:]
            val_df = df[:num_val_samples]

    if config.upsample_thresh:
        train_df = dataset_utils.upsample_data(train_df,
                                               thresh=config.upsample_thresh,
                                               seed=config.seed)

    rng = np.random.default_rng(config.seed)
    index = np.arange(len(train_df))
    rng.shuffle(index)
    train_df = train_df.iloc[index]

    del train_cfg.type
    del train_cfg.root_dir
    del train_cfg.folds
    if 'upsample_thresh' in train_cfg:
        del train_cfg.upsample_thresh
    if 'filter_thresh' in train_cfg:
        del train_cfg.filter_thresh
    del val_cfg.type
    del val_cfg.root_dir
    del val_cfg.folds

    train_cfg.audio_augments = augmentations.build_augments(
        train_cfg.audio_augments)
    train_cfg.spec_augments = augmentations.build_augments(
        train_cfg.spec_augments)

    train_ds = bird_clef_dataset.BirdCLEFDataset(
        train_df['filename'].values,
        labels=train_df['target'].values,
        **train_cfg) if train_df is not None else None
    val_ds = bird_clef_dataset.BirdCLEFDataset(
        val_df['filename'].values, labels=val_df['target'].values, **
        val_cfg) if val_df is not None else None

    return train_ds, val_ds


_DATASET_TYPE_TO_CREATE_FUNCTION = {
    'BirdCLEF': _get_bird_clef_dataset,
}


def get_data(
    config: addict.Dict
) -> Tuple[Optional[torch.utils.data.Dataset],
           Optional[torch.utils.data.Dataset]]:
    train_cfg = config.train
    val_cfg = config.val
    assert train_cfg or val_cfg, '"train" or "val" must be specified in data config.'

    dataset_type = train_cfg.type if train_cfg is not None else val_cfg.type
    assert dataset_type is not None, '"type" must be specified in dataset config.'
    assert dataset_type in _DATASET_TYPE_TO_CREATE_FUNCTION, 'dataset type {} doesnt found.'.format(
        dataset_type)

    return _DATASET_TYPE_TO_CREATE_FUNCTION[dataset_type](config)


if __name__ == '__main__':
    import argparse
    import importlib

    import addict

    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--save_results', action='store_true')
    args = parser.parse_args()

    config_path = Path(args.config).absolute()
    config_path = '.'.join([config_path.parent.name, config_path.stem])
    config = importlib.import_module(str(config_path))

    data = addict.Dict(config.data)
    train_ds, val_ds = get_data(data)

    for i in range(1):
        x, y = train_ds[i], val_ds[i]
        x_image = x['image']
        x_label = x['label']
        x_onehot_label = x['label_onehot']
        x_class_name = x['class_name']
        y_image = y['image']
        y_label = y['label']
        y_onehot_label = y['label_onehot']
        y_class_name = y['class_name']
        print("train image shape: ", x_image.shape)
        print("train label: ", x_label)
        print("train onehot label: ", x_onehot_label)
        print("train class name: ", x_class_name)
        print("val image shape: ", y_image.shape)
        print("val label: ", y_label)
        print("val onehot label: ", y_onehot_label)
        print("val class name: ", y_class_name)

        if args.save_results:
            import librosa.display as lid
            import matplotlib.pyplot as plt

            lid.specshow(x_image[0].numpy(),
                         sr=train_ds.sample_rate,
                         hop_length=50130,
                         fmin=train_ds.f_min,
                         fmax=train_ds.f_max,
                         x_axis='time',
                         y_axis='mel',
                         cmap='coolwarm')
            plt.savefig('train_image.png')
            plt.cla()
            lid.specshow(y_image[0].numpy(),
                         sr=val_ds.sample_rate,
                         hop_length=val_ds.hop_length,
                         fmin=val_ds.f_min,
                         fmax=val_ds.f_max,
                         x_axis='time',
                         y_axis='mel',
                         cmap='coolwarm')
            plt.savefig('val_image.png')
