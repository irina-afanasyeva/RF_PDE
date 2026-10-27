"""Self-adaptive (soft-attention) pointwise weights.

Verified from the reference code (github.com/levimcclenny/SA-PINNs,
Burgers/burgers.py) -- exact quote:

    mse_f_u = tf.reduce_mean(tf.square(col_weights*f_u_pred))
    mse_0_u = tf.reduce_mean(tf.square(u_weights*(u0 - u0_pred)))

This computes mean((lambda * residual)**2), which is algebraically
equivalent to mean(lambda**2 * residual**2). The EFFECTIVE mask applied to
the squared residual is therefore QUADRATIC, m(lambda) = lambda**2, NOT
the identity m(lambda) = lambda -- even though no separate transform
(sigmoid/polynomial) is applied to the raw weight variable before it
enters the "square the weighted term" expression above. (Section 3 of the
paper, arXiv:2009.04544, writes the general formulation as
m(lambda)*residual**2 and lists polynomial masks m(lambda)=c*lambda**q as
one candidate family; the Burgers code's concrete expression is the
special case c=1, q=2, written via "square the weighted term" rather than
"square the residual, then multiply by lambda**2" -- the two are the same
number, just different ways of writing the same arithmetic.)

CONCRETE CONSEQUENCE: the initial PDE attention parameter lambda=100 (see
config.COL_WEIGHT_INIT) gives an EFFECTIVE initial PDE-loss multiplier of
lambda**2 = 10,000, not 100. This was verified against the official code
and is intentional/faithful -- see config.py for the corresponding note.

col_weights (one scalar per collocation point) and u_weights (one scalar
per IC point) are each multiplied in BEFORE squaring, exactly matching the
reference. There is NO self-adaptive weight for the boundary-condition
term; BC loss is always plain unweighted MSE in the reference Burgers
example.

Weight-construction modes provided:
  - make_adaptive_weights: SA-PINN, trainable, optimized by ascent.
  - make_baseline_weights: conventional PINN, fixed at 1.0 (effective
    mask 1**2=1, i.e. exactly unweighted MSE) -- the loss-FORMULA is
    unchanged from the adaptive path, guaranteeing baseline and SA-PINN
    differ ONLY in weighting, never in architecture or loss shape.
  - make_fixed_weights: loss-SCALE-matched control. Same initial values
    as make_adaptive_weights (lambda=100 for PDE, same U(0,1) draw for
    IC, given the same seed/generator), but NOT trainable -- isolates
    whether any adaptive-vs-baseline difference comes from the per-point
    ADAPTIVITY of the weights, or merely from the larger fixed effective
    loss scale (10,000x on the PDE term) that the lambda=100
    initialization already introduces before any ascent step happens.
  - make_frozen_constant_weights: BALANCED fixed-weight control (added to
    disentangle "frozen" from "imbalanced" as separate candidate causes of
    make_fixed_weights' failure -- see docs/RESEARCH_LOG.md). Unlike
    make_fixed_weights (col=100, u~U(0,1), i.e. PDE and IC terms given very
    different effective scales -- lambda**2=10,000 for PDE vs. ~0.25 mean
    for IC), this sets col_weights AND u_weights to the SAME constant
    value, both frozen. Two concrete instances are used:
    balanced_10 (value=10, effective mask 100 on both PDE and IC) and
    balanced_100 (value=100, effective mask 10,000 on both). If these
    configurations train successfully where make_fixed_weights failed, the
    failure was caused by the PDE/IC *imbalance*, not by the weights being
    frozen per se.
"""
import torch

from . import config


def identity_mask(weights):
    """m_raw(lambda) = lambda -- this is the transform applied to the raw
    weight variable BEFORE it is multiplied into the residual and the
    product is squared (see module docstring). It is NOT the effective
    mask on the squared residual: squaring (weight*residual) makes the
    effective mask on residual**2 equal to lambda**2, not lambda. The name
    "identity_mask" refers only to this raw pre-multiplication step, where
    no sigmoid/polynomial transform is applied -- it does not mean the
    overall scheme behaves as an identity (linear) weighting of the
    squared residual."""
    return weights


def make_adaptive_weights(n_f, n_0, device, generator):
    """[CODE] col_weights initialized to the constant 100.0 (effective
    PDE-loss multiplier lambda**2 = 10,000 at init, see module docstring);
    u_weights initialized Uniform(0,1). Both are leaf tensors with
    requires_grad=True so they can be optimized by their own (ascent)
    optimizers."""
    col_weights = torch.full(
        (n_f, 1), config.COL_WEIGHT_INIT, dtype=config.DTYPE, device=device
    )
    col_weights.requires_grad_(True)

    u_weights = torch.rand((n_0, 1), generator=generator, dtype=config.DTYPE).to(device)
    u_weights.requires_grad_(True)
    return col_weights, u_weights


def make_baseline_weights(n_f, n_0, device):
    """Conventional-PINN path: fixed weights of 1.0, not trainable. Using
    these in the SAME weighted-loss formula as the SA-PINN path reduces it
    exactly to unweighted MSE (effective mask 1**2=1)."""
    col_weights = torch.ones((n_f, 1), dtype=config.DTYPE, device=device)
    u_weights = torch.ones((n_0, 1), dtype=config.DTYPE, device=device)
    return col_weights, u_weights


def make_fixed_weights(n_f, n_0, device, generator):
    """Loss-scale-matched control: IDENTICAL initial values to
    make_adaptive_weights (same lambda=100 constant for col_weights, same
    U(0,1) draw for u_weights given the same generator/seed), but
    requires_grad=False -- frozen for the entire run, no ascent optimizer.
    Differs from make_adaptive_weights ONLY in trainability, enabling a
    clean isolation of "adaptivity" from "larger fixed loss scale"."""
    col_weights = torch.full(
        (n_f, 1), config.COL_WEIGHT_INIT, dtype=config.DTYPE, device=device
    )
    u_weights = torch.rand((n_0, 1), generator=generator, dtype=config.DTYPE).to(device)
    # requires_grad left at its default False -- frozen throughout, by
    # construction (not merely "not optimized").
    return col_weights, u_weights


def make_frozen_constant_weights(n_f, n_0, device, value):
    """Balanced fixed-weight control: BOTH col_weights and u_weights are
    set to the SAME constant `value` (not config.COL_WEIGHT_INIT, not a
    random draw), frozen for the whole run (requires_grad left at its
    default False). Uses the same squared-mask loss convention as every
    other mode -- effective mask on the squared residual is value**2 for
    both the PDE and IC terms, equally. No RNG is consumed (no generator
    needed), so this introduces no new RNG-footprint difference relative
    to the other modes."""
    col_weights = torch.full((n_f, 1), float(value), dtype=config.DTYPE, device=device)
    u_weights = torch.full((n_0, 1), float(value), dtype=config.DTYPE, device=device)
    return col_weights, u_weights
