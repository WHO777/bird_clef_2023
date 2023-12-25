import numpy as np
import torch
from scipy.special import softmax


class Accuracy:

    def __init__(self,
                 categorical=True,
                 from_logits=True,
                 is_sed_output=False,
                 name='accuracy'):
        self.categorical = categorical
        self.from_logits = from_logits
        self.is_sed_output = is_sed_output
        self.name = name

    def __call__(self, y_true, y_pred):
        if self.is_sed_output:
            y_pred = y_pred['clipwise_output']
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if self.from_logits:
            y_pred = softmax(y_pred, axis=1)
        if self.categorical:
            y_true = np.argmax(y_true, axis=1)
        n = y_true.shape[0]
        max_idx_classes = y_pred.argmax(axis=1)
        acc = np.equal(max_idx_classes, y_true).sum().item() / n
        return acc
