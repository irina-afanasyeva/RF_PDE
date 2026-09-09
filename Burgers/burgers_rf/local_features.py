import torch
import torch.nn as nn


class GaussianLocal(nn.Module):
    def __init__(self, center=0.0, width=0.05):
        super().__init__()
        self.center = center
        self.width = width

    def forward(self, x):
        return torch.exp(-((x - self.center) ** 2) / (2 * self.width**2))


class GaussianSumLocal(nn.Module):
    def __init__(self, center=0.0, widths=(0.05,)):
        super().__init__()
        self.center = center
        self.widths = list(widths)

    def forward(self, x):
        basis = [torch.exp(-((x - self.center) ** 2) / (2 * w**2)) for w in self.widths]
        return torch.cat(basis, dim=1)
