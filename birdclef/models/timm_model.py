import timm
import torch


class TimmModel(torch.nn.Module):

    def __init__(self, name, pretrained, **kwargs):
        super(TimmModel, self).__init__()
        self.model = timm.create_model(name, pretrained=pretrained, **kwargs)

    def forward(self, x):
        return self.model(x)


class TimmClassifier(TimmModel):

    def __init__(self, name, pretrained, features_only=True, **kwargs):
        super().__init__(name, pretrained, **kwargs)
        self.features_only = features_only
        if self.features_only:
            self.model.global_pool = torch.nn.Identity()
            self.model.classifier = torch.nn.Identity()
            self.model.head = torch.nn.Identity()

    def forward(self, x):
        if self.features_only:
            x = self.model.forward_features(x)
        else:
            x = self.model(x)
        return x

    @property
    def num_features(self):
        return self.model.num_features
