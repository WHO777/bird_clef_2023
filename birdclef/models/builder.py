import copy

import torch

from birdclef import utils
from birdclef.models.classifier import Classifier
from birdclef.models.sed_model import SED
from birdclef.models.gem_pool import GeM
from birdclef.models.adaptive_avg_max_pool2d import AdaptiveAvgMaxPool2d
from birdclef.models.heads import SimpleHead
from birdclef.models.timm_model import TimmClassifier


def _get_classifier_model(cfg):
    backbone_type = eval(cfg.backbone.type)
    pool_type = eval(cfg.pool.type)
    head_type = eval(cfg.head.type)

    del cfg.backbone.type
    del cfg.pool.type
    del cfg.head.type

    backbone = backbone_type(**cfg.backbone)
    pool = pool_type(**cfg.pool)
    cfg.head.update({'input_features': backbone.num_features})
    head = head_type(**cfg.head)

    classifier = Classifier(backbone, pool, head)

    return classifier


def _get_sed_model(cfg):
    backbone_type = eval(cfg.backbone.type)

    del cfg.backbone.type

    backbone = backbone_type(**cfg.backbone)

    del cfg.type
    del cfg.pretrained
    del cfg.backbone

    model = SED(backbone, **cfg)

    return model


_MODEL_TYPE_TO_CREATE_FUNCTION = {
    'Classifier': _get_classifier_model,
    'SED': _get_sed_model,
}


def build_model(config):
    model_type = config.type
    model_builder = _MODEL_TYPE_TO_CREATE_FUNCTION[model_type]
    model = model_builder(copy.deepcopy(config))

    if config.pretrained:
        utils.load_checkpoint(model, config.pretrained, 'model')

    return model
