"""Self-adaptive (soft-attention) pointwise weights for SA-RF.

Directly mirrors sapinn/attention.py's construction and verified mask
arithmetic. The SA-RF loss (see train.py) computes

    mean((lambda_f * r)**2)             -- PDE term
    mean((lambda_0 * ic_residual)**2)   -- IC term

i.e. mean((lambda*residual)**2), algebraically mean(lambda**2 *
residual**2). The EFFECTIVE mask on the squared residual is therefore
QUADRATIC, m(lambda) = lambda**2, NOT the identity -- identical
arithmetic to SA-PINN, verified by construction (same formula, same
squaring order). There is NO self-adaptive weight for the BC term,
matching SA-PINN's Burgers example exactly.

Three modes, matching sapinn/attention.py's non-balanced modes one-to-one
(Phase 4 of the SA-RF implementation plan):
  - make_baseline_weights:  RF baseline. Fixed at 1.0, not trainable.
    Effective mask 1**2=1, i.e. exactly unweighted MSE -- the loss FORMULA
    is unchanged from the adaptive path, so baseline and SA-RF differ only
    in weighting, never in loss shape.
  - make_fixed_weights:     RF fixed-initial-SA. IDENTICAL initial values
    to make_adaptive_weights (same lambda_f=100 constant, same U(0,1)
    draw for lambda_0, given the same generator/seed), but
    requires_grad=False -- frozen for the entire run. Isolates whether any
    adaptive-vs-baseline difference comes from per-point ADAPTIVITY, or
    merely from the larger fixed effective loss scale the lambda_f=100
    initialization introduces on its own.
  - make_adaptive_weights:  SA-RF adaptive. Trainable, optimized by
    gradient ascent during the Adam phase (see train.py), frozen during
    L-BFGS.
  - make_frozen_constant_weights: balanced_10/balanced_100 controls,
    mirroring sapinn/attention.py's identical addition. BOTH lambda_f and
    lambda_0 set to the SAME constant, frozen for the whole run -- isolates
    whether any adaptive-vs-baseline difference comes from the per-point
    ADAPTIVITY of the weights, or merely from PDE/IC scale IMBALANCE.
"""
import torch

from . import config


def identity_mask(weights):
    """m_raw(lambda) = lambda -- the raw pre-multiplication transform, NOT
    the effective mask on the squared residual (which is lambda**2, see
    module docstring). Named for consistency with sapinn/attention.py."""
    return weights


def make_adaptive_weights(n_f, n_0, device, generator):
    lambda_f = torch.full((n_f, 1), config.LAMBDA_F_INIT, dtype=config.DTYPE, device=device)
    lambda_f.requires_grad_(True)

    lambda_0 = torch.rand((n_0, 1), generator=generator, dtype=config.DTYPE).to(device)
    lambda_0.requires_grad_(True)
    return lambda_f, lambda_0


def make_baseline_weights(n_f, n_0, device):
    lambda_f = torch.ones((n_f, 1), dtype=config.DTYPE, device=device)
    lambda_0 = torch.ones((n_0, 1), dtype=config.DTYPE, device=device)
    return lambda_f, lambda_0


def make_fixed_weights(n_f, n_0, device, generator):
    lambda_f = torch.full((n_f, 1), config.LAMBDA_F_INIT, dtype=config.DTYPE, device=device)
    lambda_0 = torch.rand((n_0, 1), generator=generator, dtype=config.DTYPE).to(device)
    # requires_grad left at its default False -- frozen throughout, by
    # construction (not merely "not optimized").
    return lambda_f, lambda_0


def make_frozen_constant_weights(n_f, n_0, device, value):
    """Balanced fixed-weight control (mirrors sapinn/attention.py's
    make_frozen_constant_weights exactly): BOTH lambda_f and lambda_0 set
    to the SAME constant `value`, frozen for the whole run. No RNG is
    consumed, so this introduces no RNG-footprint difference relative to
    the other modes."""
    lambda_f = torch.full((n_f, 1), float(value), dtype=config.DTYPE, device=device)
    lambda_0 = torch.full((n_0, 1), float(value), dtype=config.DTYPE, device=device)
    return lambda_f, lambda_0
