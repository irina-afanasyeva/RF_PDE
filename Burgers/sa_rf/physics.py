"""PDE/IC/BC residuals for SA-RF.

The PDE residual is NOT reimplemented here -- it is imported directly
from burgers_rf.physics.burgers_residual (unmodified), exactly as the
task requires ("preserve the original RF feature construction... do not
substitute a new RF basis"). IC/BC handling mirrors sapinn/physics.py's
convention (fixed points, split left/right boundary terms), since that is
the convention the self-adaptive per-point weighting scheme requires.
"""
import torch

from burgers_rf.data import g as initial_condition
from burgers_rf.physics import burgers_residual as pde_residual  # noqa: F401  (re-exported)

__all__ = ["pde_residual", "ic_prediction_error", "boundary_predictions"]


def ic_prediction_error(model, x0, t0):
    """u_RF(x0,0) - g(x0), g(x) = -sin(pi*x) (burgers_rf.data.g, unmodified)."""
    u0_pred = model(x0, t0)
    u0_true = initial_condition(x0)
    return u0_pred - u0_true.reshape(u0_pred.shape)


def boundary_predictions(model, x_lb, t_lb, x_ub, t_ub):
    """Homogeneous Dirichlet BC: target is 0 at both x=-1 and x=+1, so the
    residual is simply the prediction itself. Returned as two SEPARATE
    tensors (not concatenated) so the loss can form mean(u_lb**2) +
    mean(u_ub**2) as two additive terms, matching sapinn/physics.py's
    convention and the task's explicit "L_BC_left + L_BC_right" loss
    specification -- NOT burgers_rf.physics.boundary_residual's own
    convention of concatenating both sides before a single mean (which
    would give a different normalization when n_left == n_right)."""
    u_lb = model(x_lb, t_lb)
    u_ub = model(x_ub, t_ub)
    return u_lb, u_ub
