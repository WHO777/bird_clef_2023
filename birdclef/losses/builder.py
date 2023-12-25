from torch.nn import CrossEntropyLoss
from birdclef.losses.sed_loss import SEDLoss, SEDLossFocal


def build_loss(config):
    loss_type = eval(config.type)
    del config.type
    loss = loss_type(**config)
    return loss
