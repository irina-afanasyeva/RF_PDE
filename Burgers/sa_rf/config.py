"""Configuration for Self-Adaptive Random Features (SA-RF).

Every numeric choice is tagged with its provenance, following the
convention established in sapinn/config.py:
  [RF]      inherited unchanged from the existing pure-RF solver
            (burgers_rf/config.py) -- domain, viscosity, RF basis
            construction, evaluation grid.
  [SAPINN]  inherited unchanged from the existing SA-PINN implementation
            (sapinn/config.py) -- attention initialization, effective
            mask, Adam hyperparameters for the ascent optimizers.
  [CHOICE]  a deliberate, disclosed choice made specifically for SA-RF
            (e.g. which optimizer learning rate to use for the RF
            coefficients themselves, since the two source projects use
            different conventions here -- see ADAM_LR_COEF below).

Do not silently change [RF]/[SAPINN]-tagged values without updating this
docstring and docs/RESEARCH_LOG.md.
"""
import numpy as np
import torch

# --- PDE / domain (identical to both burgers_rf.config and sapinn.config) ---
NV = 0.01 / np.pi          # [RF]/[SAPINN] viscosity, same equation throughout this project
X_MIN, X_MAX = -1.0, 1.0   # [RF]/[SAPINN]
T_MIN, T_MAX = 0.0, 1.0    # [RF]/[SAPINN]
HERMGAUSS_ORDER = 80       # [RF] analytic reference solution (Cole-Hopf + Gauss-Hermite quadrature)

DTYPE = torch.float64      # [RF] burgers_rf.config.DTYPE -- NOT sapinn's float32; we reuse the
                           # unmodified BurgersRF model, so we keep ITS dtype convention, not
                           # SA-PINN's. Must be set via torch.set_default_dtype(DTYPE) before
                           # constructing BurgersRF (see train.py).

# --- RF basis (reused unmodified from burgers_rf.model.BurgersRF) ---
# [CHOICE] Default feature grid: the ORIGINAL canonical RF configuration
# (Nx=500, Nt=10, 5,000 coefficients) is NOT used here by default -- see
# the parameter-matching prototype below. SIGMA_X/SIGMA_T are reused
# unchanged from burgers_rf.config (same inverse-frequency-scale
# convention, w = z/sigma).
SIGMA_X = 1 / 25  # [RF] burgers_rf.config.SIGMA_X
SIGMA_T = 1 / 3   # [RF] burgers_rf.config.SIGMA_T

# --- Parameter-matching prototype (Phase 3, PROPOSED, NOT CONFIRMED) ---
# SA-PINN has 3,021 network parameters (verified: 9 Linear layers from
# LAYER_SIZES=[2,20,...,20,1], see docs/RESEARCH_LOG.md Section 41).
# Nx=159, Nt=19 gives Nx*Nt=3,021 trainable RF coefficients -- an exact
# match, under the working (unconfirmed) interpretation that "match
# trainable parameters" means match the function-approximator parameters
# only, not the attention parameters. See docs/meetings/2026-10-28.md.
NX_MATCHED = 159
NT_MATCHED = 19
assert NX_MATCHED * NT_MATCHED == 3021, "parameter-matching prototype must yield exactly 3,021 coefficients"

# --- Collocation / attention-point counts ---
# [CHOICE] Default to SA-PINN's OWN current counts (N_F=10,000 PDE,
# N_0=100 IC, 25-per-side/50-total BC) so that the attention-parameter
# budget (10,100) is identical between SA-PINN and SA-RF "for free",
# without requiring a separate matching decision. This mirrors SA-PINN's
# CURRENT (not yet paper-revised) N_b; if/when SA-PINN moves to N_b=200
# (docs/meetings/2026-10-28.md item 4), SA-RF's N_B_PER_SIDE should be
# revisited for consistency, but that is explicitly out of scope here.
N_F = 10000          # [SAPINN] sapinn.config.N_F
N_0 = 100            # [SAPINN] sapinn.config.N_0
N_B_PER_SIDE = 25    # [SAPINN] sapinn.config.N_B_PER_SIDE (current code value, not the paper's 200)

# --- Self-adaptive attention (identical convention to sapinn.config) ---
# Effective mask on the squared residual is m(lambda) = lambda**2 (see
# sa_rf/attention.py docstring) -- same quadratic-mask arithmetic as
# SA-PINN, verified against sapinn/attention.py.
LAMBDA_F_INIT = 100.0          # [SAPINN] sapinn.config.COL_WEIGHT_INIT
LAMBDA_0_INIT_DIST = "uniform_0_1"  # [SAPINN] sapinn.config.U_WEIGHT_INIT_DIST

# [SAPINN] Balanced fixed-weight controls, mirroring
# sapinn.config.BALANCED_MODE_VALUES exactly -- both lambda_f and
# lambda_0 frozen at the SAME constant (10 or 100).
BALANCED_MODE_VALUES = {
    "balanced_10": 10.0,
    "balanced_100": 100.0,
}

# --- Optimizers ---
# [CHOICE] SA-RF uses SA-PINN's Adam hyperparameters for ALL THREE
# optimizers (RF coefficients, lambda_f ascent, lambda_0 ascent), not
# burgers_rf's own canonical (LR=5e-4, weight_decay=1) recipe. Rationale:
# the explicit goal is to transfer the SA-PINN weighting MECHANISM
# faithfully; using SA-PINN's own optimizer convention for the whole
# training loop (including the coefficient optimizer) keeps the Adam
# dynamics comparable to SA-PINN's, rather than mixing two different
# optimizer recipes with no clear justification for either choice over
# the other. This is a disclosed, reconsiderable choice, not an
# authoritative transfer -- flag for review if results are sensitive to it.
ADAM_LR = 5e-3                 # [SAPINN] sapinn.config.ADAM_LR
ADAM_BETAS = (0.90, 0.999)     # [SAPINN] sapinn.config.ADAM_BETAS
# [CHOICE] No weight_decay is applied to the RF coefficients here (SA-PINN
# has no analogous regularization on its network weights either); this
# deliberately does NOT carry over burgers_rf's own WEIGHT_DECAY=1, which
# was tuned for a different optimizer/LR regime.

LBFGS_HISTORY_SIZE = 50        # [SAPINN] sapinn.config.LBFGS_HISTORY_SIZE
LBFGS_LR = 0.8                 # [SAPINN] sapinn.config.LBFGS_LR
# [SAPINN] torch.optim.LBFGS(line_search_fn=None), matching sapinn/train.py.
# Attention weights (lambda_f/lambda_0) are NEVER passed to the L-BFGS
# optimizer -- only the RF model's trainable linear-layer coefficients are
# (see train.py), matching SA-PINN's "attention frozen during L-BFGS"
# convention exactly.

# --- Training budgets ---
# [CHOICE] Deliberately conservative -- this module has not yet been used
# for a full paper-scale run. "smoke" is for the Phase 5 unit tests and
# the fastest sanity check; "short" is the Phase 6 smoke-test budget
# (cheap, but long enough to see real loss movement); "paper" mirrors
# SA-PINN's own full budget and MUST NOT be launched without explicit
# approval (see docs/meetings/2026-10-28.md and the Stage-1-style
# "do not launch full experiments yet" convention).
BUDGETS = {
    "smoke": {"adam_iters": 20, "lbfgs_iters": 5},
    "short": {"adam_iters": 300, "lbfgs_iters": 50},
    "controlled": {"adam_iters": 1000, "lbfgs_iters": 100},  # [CHOICE] one-seed comparison budget,
                                                              # estimated ~4.8 min/run from the "short"
                                                              # sweep's measured per-iteration cost --
                                                              # still far short of "paper".
    "long": {"adam_iters": 5000, "lbfgs_iters": 1000},       # [CHOICE] longer-training server budget,
                                                              # estimated ~26 min/run (CPU) from the same
                                                              # measured per-iteration cost as "controlled"
                                                              # -- intended for the compute server, not
                                                              # local/interactive use.
    "paper": {"adam_iters": 10000, "lbfgs_iters": 10000},
}

# --- Evaluation (reused unmodified from burgers_rf) ---
M_TEST = 100
N_TEST_TIMES = 10
LOG_EVERY = 10
EVAL_EVERY_ADAM = 50   # [CHOICE] more frequent than SA-PINN's 250 -- SA-RF's evaluation forward
                       # pass is far cheaper (a single linear layer) than a PINN forward pass,
                       # and short-budget smoke tests benefit from finer-grained logging.

DEFAULT_SEED = 0


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_all_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
