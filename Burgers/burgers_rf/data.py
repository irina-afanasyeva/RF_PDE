import numpy as np
import torch

from .config import HERMGAUSS_ORDER


def g(x):
    """
    Initial condition
    """
    return -torch.sin(torch.pi * x)


Gauss_pts, weights = np.polynomial.hermite.hermgauss(HERMGAUSS_ORDER)


def u_true(t, x, nv):
    temp = x - np.sqrt(4 * nv * t) * Gauss_pts
    val1 = weights * np.sin(np.pi * temp) * np.exp(-np.cos(np.pi * temp) / (2 * np.pi * nv))
    val2 = weights * np.exp(-np.cos(np.pi * temp) / (2 * np.pi * nv))
    return -np.sum(val1) / np.sum(val2)


def sample_training_points(m_train, device):
    x_train = 2 * torch.rand(m_train, 1, requires_grad=True) - 1
    x_train = x_train.to(device)
    t_train = torch.rand(m_train, 1, requires_grad=True)
    t_train = t_train.to(device)
    return x_train, t_train


def make_test_grid(m_test, n_test_times, device):
    x_test = torch.linspace(-1, 1, m_test)
    t_test = torch.linspace(0, 1, n_test_times)
    x_t_product = torch.cartesian_prod(t_test, x_test)
    x_test = x_t_product[:, [1]].to(device)
    t_test = x_t_product[:, [0]].to(device)
    return x_test, t_test


def make_reference_values(t_test, x_test, nv):
    y_true = [
        u_true(
            t_test.cpu().detach().numpy()[i],
            x_test.cpu().detach().numpy()[i],
            nv,
        )
        for i in range(x_test.shape[0])
    ]
    return np.array(y_true)
