import torch
import torch.nn as nn


"""
Class for torch models
"""

# initial exploration with simple regression model
class LogisticRegression(nn.Module):
    # initialize the class
    def __init__(self, **kwargs):
        super(LogisticRegression, self).__init__()
        self.c = kwargs['c']
        self.device = kwargs['device']
        self.w1 = nn.Linear(self.c['embedding_model_size'], 1)
        self.to(self.device)

    # forward pass
    def forward(self, x):
        return torch.sigmoid(self.w1(x))
    