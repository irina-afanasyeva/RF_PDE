"""Configuration for the SA-PINN Burgers benchmark.

Every numeric value below is tagged with its provenance:
  [CODE]   verified directly from the official repository
           (github.com/levimcclenny/SA-PINNs/Burgers/burgers.py,
           eager_lbfgs.py), fetched and quoted during Phase 1 inspection.
  [PAPER]  verified from the arXiv paper text (McClenny & Braga-Neto,
           arXiv:2009.04544, Section 3 / Section 4.1).
  [CHOICE] a deliberate, disclosed implementation choice made here because
           the reference did not specify a value, used an artifact we do
           not have (the raw burgers_shock.mat grid), or because a direct
           PyTorch equivalent was used instead of a TF-specific construct.

Do not silently change the [CODE]/[PAPER]-tagged values without updating
this docstring and docs/RESEARCH_LOG.md.
"""
import numpy as np
import torch

# --- PDE / domain (identical to burgers_rf.config; verified same equation) ---
NV = 0.01 / np.pi          # [CODE]/[PAPER] viscosity, matches our RF project exactly
X_MIN, X_MAX = -1.0, 1.0   # [CODE] lb=[-1.0], ub=[1.0]
T_MIN, T_MAX = 0.0, 1.0    # [CODE]/[PAPER] standard Burgers benchmark domain
HERMGAUSS_ORDER = 80       # [CHOICE] reused from burgers_rf for the analytic reference solution

# --- dtype ---
# [CODE] the reference casts all tensors to tf.float32. This is a
# deliberate, disclosed departure from burgers_rf's float64 convention,
# scoped ONLY to this isolated SA-PINN benchmark.
DTYPE = torch.float32

# --- Architecture ---
LAYER_SIZES = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]  # [CODE]/[PAPER] "8 hidden layers of 20 neurons"
ACTIVATION = "tanh"        # [CODE] tf.nn.tanh hidden activation, linear output
WEIGHT_INIT = "glorot_normal"  # [CODE] kernel_initializer="glorot_normal" -> torch.nn.init.xavier_normal_
BIAS_INIT = "zeros"        # [CHOICE] Keras Dense default bias_initializer, not overridden in the reference code

# --- Training point counts ---
N_F = 10000   # [CODE]/[PAPER] collocation (residual) points, Latin Hypercube Sampling
N_0 = 100     # [CODE]/[PAPER] initial-condition points
N_B_PER_SIDE = 25  # [CODE] boundary points per side (50 total).
# IMPORTANT VERIFIED DISCREPANCY: the paper's Section 4.1 text states
# N_b=200, but the actual committed code uses N_b=25 per side (50 total).
# We implement the CODE's value (runnable/verified) and record this
# discrepancy here rather than silently picking one. Do not claim this
# matches the paper's stated N_b.
N_B_TOTAL = 2 * N_B_PER_SIDE

# --- Self-adaptive weights (attention) ---
# [CODE] col_weights (one per collocation point) initialized to a constant
# 100.0; u_weights (one per IC point) initialized Uniform(0,1). There is
# NO self-adaptive weight for the boundary-condition term in the reference
# Burgers example -- BC loss is always plain, unweighted MSE.
#
# EFFECTIVE MASK IS QUADRATIC, NOT IDENTITY (verified against the official
# code -- see sapinn/attention.py's module docstring for the full
# derivation). The reference computes
#     mse_f_u = reduce_mean(square(col_weights * f_u_pred))
# i.e. mean((lambda*residual)**2) = mean(lambda**2 * residual**2). The
# EFFECTIVE mask applied to the squared residual is therefore
# m(lambda) = lambda**2, a special case (c=1, q=2) of the polynomial mask
# family the paper's Section 3 discusses in general -- NOT the identity
# m(lambda)=lambda. CONCRETE CONSEQUENCE: COL_WEIGHT_INIT=100 below gives
# an EFFECTIVE initial PDE-loss multiplier of 100**2 = 10,000, not 100.
# Our implementation (sapinn/train.py) reproduces this exact arithmetic
# (squares the weighted residual, matching the reference bit-for-bit);
# only this comment and attention.py's docstrings were previously
# mislabeled as "identity mask."
COL_WEIGHT_INIT = 100.0
U_WEIGHT_INIT_DIST = "uniform_0_1"

# [CHOICE] "Balanced" fixed-weight diagnostic modes, added to separate
# "frozen" from "imbalanced" as candidate explanations for the original
# fixed_weight mode's failure (col=100 i.e. effective 10,000x on PDE,
# u~U(0,1) i.e. effective ~0.25x mean on IC -- a 40,000x imbalance). Both
# PDE and IC masks are set to the SAME constant, frozen throughout. Mode
# names double as their own output-directory names (see scripts/
# train_sapinn.py), so these are also the authoritative set of valid
# "balanced_*" mode names.
BALANCED_MODE_VALUES = {
    "balanced_10": 10.0,
    "balanced_100": 100.0,
}

# --- Optimizers ---
ADAM_LR = 5e-3       # [CODE] lr=0.005 for all three Adam optimizers (network, col_weights, u_weights)
ADAM_BETAS = (0.90, 0.999)  # [CODE] beta_1=.90 explicit; beta_2 not overridden -> Keras/TF default 0.999
LBFGS_HISTORY_SIZE = 50     # [CODE] eager_lbfgs.py nCorrection=50
LBFGS_LR = 0.8              # [CODE] lbfgs(..., learningRate=0.8)
# [CHOICE] torch.optim.LBFGS(line_search_fn=None) is used instead of
# porting eager_lbfgs.py line-by-line. Same algorithm family (limited-
# memory two-loop-recursion L-BFGS, no real line search -- the reference
# itself only takes a fixed step after the first iteration), different
# concrete implementation (PyTorch native vs. a custom TF2-eager port).
# Attention weights (col_weights/u_weights) are NOT updated during the
# L-BFGS phase, matching the reference exactly (lbfgs() is only ever
# called on get_weights(u_model), the network weights).

# --- Training budgets ---
# [CODE] the repository's actual committed call is fit(..., tf_iter=100,
# newton_iter=100) -- "repo_default" below. [PAPER] states "10k Adam + 10k
# L-BFGS" -- "paper" below. "smoke" is our own addition for a fast CPU
# sanity check, not present in the reference.
BUDGETS = {
    "smoke": {"adam_iters": 20, "lbfgs_iters": 5},
    "repo_default": {"adam_iters": 100, "lbfgs_iters": 100},
    "paper": {"adam_iters": 10000, "lbfgs_iters": 10000},
}

# --- Reproducibility ---
# [CHOICE] the reference sets NO random seed anywhere (verified absent
# from burgers.py). We set explicit seeds for reproducibility; this means
# our runs are internally reproducible but cannot be claimed to reproduce
# the paper's exact published 4.80e-4 +/- 1.01e-4 (itself an average over
# 10 UNSEEDED random restarts, per the paper text).
DEFAULT_SEED = 0

# --- Evaluation (reused from burgers_rf, not reimplemented) ---
M_TEST = 100
N_TEST_TIMES = 10
LOG_EVERY = 10          # [CODE] reference logs every 10 Adam epochs -- cheap, loss-value-only logging
EVAL_EVERY_ADAM = 250   # [CHOICE] full relative-L2 evaluation cadence -- decoupled from LOG_EVERY
                        # because evaluate_relative_l2() does two extra forward passes plus a
                        # device sync (.cpu().numpy()); at paper scale (10,000 Adam epochs) evaluating
                        # every LOG_EVERY=10 would mean 1,000 forced syncs. Loss values needed for
                        # LOG_EVERY-cadence logging are already computed as part of the training step
                        # itself, so that logging remains cheap at its original cadence.

# --- Checkpointing / resume ---
CHECKPOINT_EVERY_ADAM = 500   # [CHOICE] Adam-phase checkpoint cadence, in epochs
LBFGS_CHUNK_SIZE = 500        # [CHOICE] max_iter per torch.optim.LBFGS.step() call; chunking is
                              # used ONLY to enable checkpointing/progress visibility during a long
                              # L-BFGS phase -- empirically verified bit-identical to one
                              # uninterrupted call for this problem, see
                              # scripts/test_sapinn_lbfgs_chunking.py and docs/RESEARCH_LOG.md.


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_all_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
