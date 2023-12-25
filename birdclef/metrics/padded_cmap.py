import numpy as np
import torch
from sklearn import metrics
from scipy.special import softmax


class PaddedCMap:

    def __init__(self,
                 padding_factor=5,
                 from_logits=True,
                 is_sed_output=False,
                 name='padded_cmap'):
        self.padding_factor = padding_factor
        self.from_logits = from_logits
        self.id_sed_output = is_sed_output
        self.name = name

    def __call__(self, y_true, y_pred):
        if self.id_sed_output:
            y_pred = y_pred['clipwise_output']
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if self.from_logits:
            y_pred = softmax(y_pred, axis=1)
        num_classes = y_true.shape[1]
        pad_rows = np.array([[1] * num_classes] * self.padding_factor)
        y_true = np.concatenate([y_true, pad_rows])
        y_pred = np.concatenate([y_pred, pad_rows])
        score = metrics.average_precision_score(
            y_true,
            y_pred,
            average='macro',
        )
        return score
