import torch
import torch.nn as nn


class SimpleHead(torch.nn.Module):

    def __init__(self,
                 num_classes,
                 input_features,
                 embedding_size=None,
                 dropout_rate=0.0):
        super(SimpleHead, self).__init__()
        self.drop = nn.Dropout(dropout_rate)
        self.embedding = nn.Linear(
            input_features,
            embedding_size) if embedding_size is not None else None
        self.fc = nn.Linear(
            embedding_size if embedding_size is not None else input_features,
            num_classes)

    def forward(self, x):
        x = self.drop(x)
        if self.embedding is not None:
            x = self.embedding(x)
        x = self.fc(x)
        return x
