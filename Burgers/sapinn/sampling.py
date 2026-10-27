"""Fixed, one-time training-point sampling for the SA-PINN benchmark.

CRITICAL METHODOLOGICAL POINT (verified from the reference code, not a
style choice): self-adaptive per-point weights only make sense if the set
of points is FIXED for the entire training run -- each weight is a
trainable scalar attached to one specific point index. The reference
code samples collocation/IC/BC points ONCE before training and reuses them
every epoch; it does NOT resample them per epoch (unlike burgers_rf's
physics.py, which redraws boundary/initial points fresh every epoch -- a
deliberately different convention for a different method). This module
replicates the reference's "fixed points for the whole run" behavior.

Point LOCATIONS differ from the reference in one disclosed way: the
official code selects IC/BC points from a precomputed 256x100 grid loaded
from burgers_shock.mat, which we do not have. We draw IC/BC x,t locations
via continuous uniform random sampling instead (documented in config.py),
then hold them fixed -- same mechanism (fixed throughout training),
different generation of the candidate locations.
"""
import numpy as np
import torch
from scipy.stats import qmc

from . import config


def sample_collocation_points(n_f, rng):
    """Latin Hypercube Sampling over [x_min,x_max] x [t_min,t_max],
    matching the reference's pyDOE.lhs(2, N_f) call. scipy.stats.qmc is
    used as a standard, well-tested LHS implementation (disclosed
    substitute for pyDOE, same sampling design)."""
    sampler = qmc.LatinHypercube(d=2, seed=rng)
    unit_sample = sampler.random(n_f)
    scaled = qmc.scale(
        unit_sample,
        [config.X_MIN, config.T_MIN],
        [config.X_MAX, config.T_MAX],
    )
    x_f = scaled[:, 0:1]
    t_f = scaled[:, 1:2]
    return x_f, t_f


def sample_ic_points(n_0, rng):
    """Continuous uniform x in [x_min,x_max] at t=0 (see module docstring
    for why this differs from the reference's discrete grid selection)."""
    x0 = rng.uniform(config.X_MIN, config.X_MAX, size=(n_0, 1))
    t0 = np.zeros((n_0, 1))
    return x0, t0


def sample_bc_points(n_b_per_side, rng):
    """Continuous uniform t in [t_min,t_max] for each boundary, n_b_per_side
    points per side (matching the reference code's N_b=25-per-side value;
    see config.py for the documented paper-vs-code N_b discrepancy)."""
    t_lb = rng.uniform(config.T_MIN, config.T_MAX, size=(n_b_per_side, 1))
    t_ub = rng.uniform(config.T_MIN, config.T_MAX, size=(n_b_per_side, 1))
    x_lb = np.full((n_b_per_side, 1), config.X_MIN)
    x_ub = np.full((n_b_per_side, 1), config.X_MAX)
    return x_lb, t_lb, x_ub, t_ub


def build_training_tensors(seed, device):
    """Build ALL fixed training points ONCE, for a given seed, in a fixed
    order (collocation -> IC -> BC) so that RNG consumption is identical
    and deterministic regardless of caller. Returns a dict of torch
    tensors on `device`, dtype=config.DTYPE, with requires_grad=True on
    the autograd-differentiated coordinates (x_f,t_f,x0,t0 not needed;
    x_lb/t_lb/x_ub/t_ub need grad only if boundary derivatives were
    required -- they are not, for Dirichlet BCs)."""
    rng_np = np.random.default_rng(seed)

    x_f, t_f = sample_collocation_points(config.N_F, rng_np)
    x0, t0 = sample_ic_points(config.N_0, rng_np)
    x_lb, t_lb, x_ub, t_ub = sample_bc_points(config.N_B_PER_SIDE, rng_np)

    def to_tensor(arr, requires_grad=False):
        return torch.tensor(arr, dtype=config.DTYPE, device=device, requires_grad=requires_grad)

    tensors = {
        "x_f": to_tensor(x_f, requires_grad=True),
        "t_f": to_tensor(t_f, requires_grad=True),
        "x0": to_tensor(x0),
        "t0": to_tensor(t0),
        "x_lb": to_tensor(x_lb),
        "t_lb": to_tensor(t_lb),
        "x_ub": to_tensor(x_ub),
        "t_ub": to_tensor(t_ub),
    }
    return tensors
