"""PDE/IC/BC residuals via autograd, for the plain PINN model in this
package. Deliberately separate from burgers_rf/physics.py (that module
operates on BurgersRF's forward_components API and resamples IC/BC points
every epoch; this one operates on the plain PINN above with FIXED points,
see sampling.py). Same governing equation, same IC, same BC."""
import torch

from . import config


def pde_residual(model, x, t, nv=config.NV):
    """f = u_t + u*u_x - nv*u_xx. x,t must have requires_grad=True."""
    u = model(x, t)
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    return u_t + u * u_x - nv * u_xx


def initial_condition(x):
    """g(x) = -sin(pi*x), identical closed form to burgers_rf.data.g."""
    return -torch.sin(torch.pi * x)


def ic_prediction_error(model, x0, t0):
    u0_pred = model(x0, t0)
    u0_true = initial_condition(x0)
    return u0_pred - u0_true


def boundary_predictions(model, x_lb, t_lb, x_ub, t_ub):
    u_lb = model(x_lb, t_lb)
    u_ub = model(x_ub, t_ub)
    return u_lb, u_ub
