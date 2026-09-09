import torch
import torch.nn as nn


class GaussianLocal(nn.Module):
    def __init__(self, center=0.0, width=0.05):
        super().__init__()
        self.center = center
        self.width = width

    def forward(self, x):
        return torch.exp(-((x - self.center) ** 2) / (2 * self.width**2))
