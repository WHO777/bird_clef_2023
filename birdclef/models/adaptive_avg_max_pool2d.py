import torch.nn as nn
import torch.nn.functional as F


class AdaptiveAvgMaxPool2d(nn.Module):

    def __init__(self, output_size):
        super(AdaptiveAvgMaxPool2d, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        x_avg = F.adaptive_avg_pool2d(x, self.output_size)
        x_max = F.adaptive_max_pool2d(x, self.output_size)
        return 0.5 * (x_avg + x_max)
