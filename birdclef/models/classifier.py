import torch


class Classifier(torch.nn.Module):

    def __init__(self, backbone, pool, head):
        super(Classifier, self).__init__()
        self.backbone = backbone
        self.pool = pool
        self.head = head

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x
