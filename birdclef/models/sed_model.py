import torch
import torch.nn as nn
import torch.nn.functional as F


def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.0)


def init_weights(model):
    classname = model.__class__.__name__
    if classname.find("Conv2d") != -1:
        nn.init.xavier_uniform_(model.weight, gain=np.sqrt(2))
        model.bias.data.fill_(0)
    elif classname.find("BatchNorm") != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)
    elif classname.find("GRU") != -1:
        for weight in model.parameters():
            if len(weight.size()) > 1:
                nn.init.orghogonal_(weight.data)
    elif classname.find("Linear") != -1:
        model.weight.data.normal_(0, 0.01)
        model.bias.data.zero_()


def interpolate(x: torch.Tensor, ratio: int):
    """Interpolate data in time domain. This is used to compensate the
    resolution reduction in downsampling of a CNN.
    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate
    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def pad_framewise_output(framewise_output: torch.Tensor, frames_num: int):
    """Pad framewise_output to the same length as input frames. The pad value
    is the same as the value of the last frame.
    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad
    Outputs:
      output: (batch_size, frames_num, classes_num)
    """
    output = F.interpolate(framewise_output.unsqueeze(1),
                           size=(frames_num, framewise_output.size(2)),
                           align_corners=True,
                           mode="bilinear").squeeze(1)

    return output


class AttBlockV2(nn.Module):

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 activation="sigmoid"):
        super().__init__()

        self.activation = activation
        self.att = nn.Conv1d(in_channels=in_features,
                             out_channels=out_features,
                             kernel_size=1,
                             stride=1,
                             padding=0,
                             bias=True)
        self.cla = nn.Conv1d(in_channels=in_features,
                             out_channels=out_features,
                             kernel_size=1,
                             stride=1,
                             padding=0,
                             bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)

    def forward(self, x):
        norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        cla = self.nonlinear_transform(self.cla(x))
        x = torch.sum(norm_att * cla, dim=2)
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)


class SED(nn.Module):

    def __init__(
        self,
        backbone,
        num_classes=264,
        attn_activation='linear',
        interpolate_ratio=30,
    ):
        super().__init__()

        self.backbone = backbone
        num_features = backbone.num_features
        self.fc1 = nn.Linear(num_features, num_features, bias=True)
        self.att_block = AttBlockV2(num_features,
                                    num_classes,
                                    activation=attn_activation)
        self.interpolate_ratio = interpolate_ratio
        self.init_weight()

    def init_weight(self):
        init_layer(self.fc1)

    def forward(self, x):
        frames_num = x.size(3)

        # (batch_size, channels, freq, frames)
        x = self.backbone(x)

        # (batch_size, channels, frames)
        x = torch.mean(x, dim=2)

        # channel smoothing
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2

        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5, training=self.training)

        (clipwise_output, norm_att, segmentwise_output) = self.att_block(x)
        segmentwise_output = segmentwise_output.transpose(1, 2)

        framewise_output = interpolate(segmentwise_output,
                                       self.interpolate_ratio)
        framewise_output = pad_framewise_output(framewise_output, frames_num)

        outputs = {
            'framewise_output': framewise_output,
            'clipwise_output': clipwise_output
        }

        return outputs
