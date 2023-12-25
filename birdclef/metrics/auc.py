import numpy as np
import sklearn.metrics as sklearn_metrics
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')  # disable gpu usage
import torch
from torchmetrics.classification import MulticlassPrecisionRecallCurve
from scipy.special import softmax


class AUC:

    def __init__(self,
                 num_classes,
                 thresholds=None,
                 categorical=True,
                 from_logits=True,
                 is_sed_output=False,
                 name='pr_auc'):
        self.pr_curve = MulticlassPrecisionRecallCurve(num_classes,
                                                       thresholds=thresholds)
        self.categorical = categorical
        self.from_logits = from_logits
        self.is_sed_output = is_sed_output
        self.name = name

    def __call__(self, y_true, y_pred):
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if self.is_sed_output:
            y_pred = y_pred['clipwise_output']

        if self.from_logits:
            y_pred = softmax(y_pred, axis=1)
        if self.categorical:
            y_true = np.argmax(y_true, axis=1)

        y_true = torch.from_numpy(y_true)
        y_pred = torch.from_numpy(y_pred)

        class_precision, class_recall, _ = self.pr_curve(y_pred, y_true)

        auc_scores = []
        for precision, recall in zip(class_precision, class_recall):
            auc = sklearn_metrics.auc(recall, precision)
            print(auc)
            auc_scores.append(auc)

        auc = np.nanmean(np.array(auc_scores))

        return auc


class AUC_TF:

    def __init__(self,
                 num_classes,
                 thresholds=None,
                 categorical=True,
                 from_logits=True,
                 is_sed_output=False,
                 name='pr_auc'):
        self.num_classes = num_classes
        self.auc = tf.keras.metrics.AUC(thresholds=thresholds)
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
        if not self.categorical:
            y_true = tf.keras.utils.to_categorical(
                y_true, num_classes=self.num_classes)
        auc = self.auc(y_true, y_pred).numpy()
        return auc
