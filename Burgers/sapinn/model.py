"""Plain feedforward PINN network -- IDENTICAL for the conventional
baseline and the SA-PINN variant. The two modes differ ONLY in how the
loss weights the residuals (see attention.py / train.py), never in this
architecture, matching the task requirement that baseline and SA-PINN
share the same network."""
import torch
import torch.nn as nn

from . import config


class PINN(nn.Module):
    def __init__(self, layer_sizes=None):
        super().__init__()
        layer_sizes = layer_sizes or config.LAYER_SIZES
        linears = []
        for in_f, out_f in zip(layer_sizes[:-1], layer_sizes[1:]):
            linear = nn.Linear(in_f, out_f)
            nn.init.xavier_normal_(linear.weight)  # [CODE] kernel_initializer="glorot_normal"
            nn.init.zeros_(linear.bias)             # [CHOICE] Keras Dense default bias_initializer
            linears.append(linear)
        self.linears = nn.ModuleList(linears)

    def forward(self, x, t):
        h = torch.cat([x, t], dim=1)
        for linear in self.linears[:-1]:
            h = torch.tanh(linear(h))  # [CODE] tf.nn.tanh hidden activation
        return self.linears[-1](h)      # [CODE] linear output activation
