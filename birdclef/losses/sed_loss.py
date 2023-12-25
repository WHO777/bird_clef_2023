from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class SEDLoss(nn.Module):

    def __init__(self):
        super(SEDLoss, self).__init__()
        self.bce = nn.BCELoss()

    def forward(self, y_pred, y_true):
        y_pred = y_pred['clipwise_output']
        y_pred = torch.clip(y_pred, 0, 1)
        y_pred = torch.where(torch.isnan(y_pred), torch.zeros_like(y_pred),
                             y_pred)
        y_pred = torch.where(torch.isinf(y_pred), torch.zeros_like(y_pred),
                             y_pred)
        y_true = y_true.float()
        loss = self.bce(y_pred, y_true)
        return loss


class SEDLossFocal(nn.Module):

    def __init__(self, alpha=0.25, gamma=2, reduction="none"):
        super(SEDLossFocal, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        y_pred = y_pred['clipwise_output']
        y_pred = torch.clip(y_pred, 0, 1)

        y_pred = torch.where(torch.isnan(y_pred), torch.zeros_like(y_pred),
                             y_pred)
        y_pred = torch.where(torch.isinf(y_pred), torch.zeros_like(y_pred),
                             y_pred)

        y_pred = y_pred.float()
        y_true = y_true.float()

        ce_loss = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction="none")
        p_t = y_pred * y_true + (1 - y_pred) * (1 - y_true)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * y_true + (1 - self.alpha) * (1 - y_true)
            loss = alpha_t * loss

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss
