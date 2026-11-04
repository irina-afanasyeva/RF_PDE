"""Fixed, one-time training-point sampling for SA-RF.

Mirrors sapinn/sampling.py's central methodological point exactly: a
self-adaptive per-point weight only makes sense if the set of points is
FIXED for the entire training run (each weight is a trainable scalar tied
to one specific point index). This is a deliberate departure from
burgers_rf.physics's own convention of resampling IC/BC points fresh
every epoch -- that convention is correct for burgers_rf's unweighted
loss, but incompatible with per-point attention weights, so SA-RF must
NOT reuse it. PDE points are already fixed-for-the-run in burgers_rf
(unchanged here); IC/BC points are the ones that change convention.
"""
import numpy as np
import torch
from scipy.stats import qmc

from . import config


def sample_collocation_points(n_f, rng):
    """Latin Hypercube Sampling over [x_min,x_max] x [t_min,t_max],
    matching sapinn/sampling.py's convention (itself matching the SA-PINN
    reference's pyDOE.lhs(2, N_f) call)."""
    sampler = qmc.LatinHypercube(d=2, seed=rng)
    unit_sample = sampler.random(n_f)
    scaled = qmc.scale(unit_sample, [config.X_MIN, config.T_MIN], [config.X_MAX, config.T_MAX])
    x_f = scaled[:, 0:1]
    t_f = scaled[:, 1:2]
    return x_f, t_f


def sample_ic_points(n_0, rng):
    x0 = rng.uniform(config.X_MIN, config.X_MAX, size=(n_0, 1))
    t0 = np.zeros((n_0, 1))
    return x0, t0


def sample_bc_points(n_b_per_side, rng):
    t_lb = rng.uniform(config.T_MIN, config.T_MAX, size=(n_b_per_side, 1))
    t_ub = rng.uniform(config.T_MIN, config.T_MAX, size=(n_b_per_side, 1))
    x_lb = np.full((n_b_per_side, 1), config.X_MIN)
    x_ub = np.full((n_b_per_side, 1), config.X_MAX)
    return x_lb, t_lb, x_ub, t_ub


def build_training_tensors(seed, device, n_f=None, n_0=None, n_b_per_side=None):
    """Build ALL fixed training points ONCE, in a fixed order (collocation
    -> IC -> BC), so RNG consumption is deterministic regardless of
    caller. n_f/n_0/n_b_per_side default to config values but are
    overridable for fast unit tests."""
    n_f = config.N_F if n_f is None else n_f
    n_0 = config.N_0 if n_0 is None else n_0
    n_b_per_side = config.N_B_PER_SIDE if n_b_per_side is None else n_b_per_side

    rng_np = np.random.default_rng(seed)

    x_f, t_f = sample_collocation_points(n_f, rng_np)
    x0, t0 = sample_ic_points(n_0, rng_np)
    x_lb, t_lb, x_ub, t_ub = sample_bc_points(n_b_per_side, rng_np)

    def to_tensor(arr, requires_grad=False):
        return torch.tensor(arr, dtype=config.DTYPE, device=device, requires_grad=requires_grad)

    return {
        "x_f": to_tensor(x_f, requires_grad=True),
        "t_f": to_tensor(t_f, requires_grad=True),
        "x0": to_tensor(x0),
        "t0": to_tensor(t0),
        "x_lb": to_tensor(x_lb),
        "t_lb": to_tensor(t_lb),
        "x_ub": to_tensor(x_ub),
        "t_ub": to_tensor(t_ub),
    }
