import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(PROJECT_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from burgers_rf.config import (  # noqa: E402
    DTYPE,
    IC_BC_WEIGHT,
    LR,
    M_BD,
    M_INT,
    M_TEST,
    N_TEST_TIMES,
    NV,
    WEIGHT_DECAY,
    set_seed,
)
from burgers_rf.data import g, make_reference_values, make_test_grid, sample_training_points  # noqa: E402
from burgers_rf.model import BurgersRF  # noqa: E402
from burgers_rf.local_features import GaussianLocal  # noqa: E402
from burgers_rf.physics import boundary_residual, burgers_residual, initial_residual  # noqa: E402
from burgers_rf.evaluation import evaluate_error_metrics  # noqa: E402
from sweep_local_width import build_shock_grid  # noqa: E402
import test_pde_trained_gaussian_correction as ptgc  # noqa: E402 -- reused unchanged:
# load_pde_trained_model, verify_baseline, build_search_baseline
import test_fixed_gaussian_correction as tfgc  # noqa: E402 -- reused unchanged: gaussian, C_VALUES, SIGMA_VALUES
import test_direct_spatial_fit as dsf  # noqa: E402 -- reused unchanged: lstsq_solve, DTYPE

# TWO-STAGE, TWO-GAUSSIAN TRAINING experiment. Stage 1 (RF-only PDE
# training) is ALREADY COMPLETE -- loaded from the verified
# checkpoint_complete.pt and frozen exactly. Stage 2 trains ONLY two scalar
# amplitudes d1,d2 for a FIXED pair of Gaussians (center/width taken
# unchanged from the strongest jointly-searched post-hoc K=2 pair-search
# geometry), initialized at exactly 0.
#
# WHY NOT model.py's built-in local-feature machinery: `local_type=
# "gaussian_sum"` forces a SINGLE shared center across widths (see
# local_features.GaussianSumLocal), and model.py explicitly raises
# NotImplementedError for any local_time_mode != "standard" when
# local_type="gaussian_sum". Neither restriction is compatible with this
# design (two INDEPENDENTLY centered Gaussians, t=0-vanishing temporal
# form). Rather than modify model.py, this script reuses the exact
# duck-typing pattern already established in
# diagnose_fixed_rf_linear_d_sweep.py: physics.burgers_residual/
# boundary_residual/initial_residual only ever call `model(x,t)`, so a
# plain closure
#
#     u_fn(x,t) = u_RF(x,t) + t * (d1*G1(x) + d2*G2(x))
#
# is a drop-in replacement -- d1,d2 are free-standing nn.Parameter scalars
# (NOT attributes of `model`), and G1,G2 are standalone, parameterless
# burgers_rf.local_features.GaussianLocal instances. model.py is not
# modified at all.
#
# Post-hoc oracle amplitudes are NOT hardcoded: the fixed (c1,sigma1,c2,
# sigma2) geometry is reused as exact existing C_VALUES/SIGMA_VALUES
# library entries, and d1_oracle,d2_oracle are refit fresh via closed-form
# least squares (dsf.lstsq_solve) on the same t=1 canonical shock grid used
# by test_pde_trained_two_gaussian_correction.py/
# test_pde_trained_k_gaussian_correction.py -- reference information only,
# never used for initialization or training.
#
# device="cpu" throughout is a CORRECTNESS requirement (see model.py's
# known nn.Parameter(...).to(device) bug): the strict checkpoint load only
# succeeds via the harmless .to("cpu") no-op path.

CHECKPOINT_PATH = PROJECT_DIR / "outputs" / "pde_trained_rf_residual" / "checkpoint_complete.pt"
CANONICAL_GLOBAL = 0.27149430095802507
CANONICAL_SMOOTH = 0.2443827084159696
CANONICAL_SHOCK = 0.8390487998889169
VERIFY_RTOL = 1e-4

EPOCHS = 1000
IC_GRID_N_X = 1000

# Fixed two-Gaussian geometry, from the strongest jointly-searched K=2
# pair-search result (test_pde_trained_two_gaussian_correction.py). These
# are NOMINAL/approximate values; the actual trainable geometry used below
# is the nearest EXACT C_VALUES/SIGMA_VALUES library entry (verified to
# match within GEOMETRY_MATCH_RTOL, never silently substituted).
C1_NOMINAL = -0.010000
SIGMA1_NOMINAL = 1.067384e-02
C2_NOMINAL = 0.010000
SIGMA2_NOMINAL = 1.490050e-02
GEOMETRY_MATCH_RTOL = 1e-4

OUTPUT_DIR = PROJECT_DIR / "outputs" / "two_stage_two_gaussian_local_feature"


def relative_l2(y_true, y_pred):
    return float(np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true))


def sign_str(x):
    if x > 0:
        return "positive"
    if x < 0:
        return "negative"
    return "zero"


def find_exact_library_value(target, library_values, rtol, label):
    """Locate the nearest entry in an EXISTING library array and verify it
    matches the target within rtol -- never silently substitutes an
    off-grid value."""
    arr = np.asarray(library_values)
    idx = int(np.argmin(np.abs(arr - target)))
    val = float(arr[idx])
    rel_diff = abs(val - target) / abs(target) if target != 0 else abs(val)
    if rel_diff > rtol:
        raise RuntimeError(
            f"{label}: nearest library value {val} does not match target {target} "
            f"within rtol={rtol} (actual rel diff {rel_diff:.3e}). Refusing to proceed "
            "with an unverified geometry reference."
        )
    return val


def load_frozen_rf_model(device="cpu"):
    """Load the complete, verified Stage-1 RF-only checkpoint with a strict
    state_dict match, use_local=False (so `model` has NO local-feature
    attributes at all), then freeze model.0.weight (the only parameter that
    defaults to requires_grad=True)."""
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model = BurgersRF(
        checkpoint["NX"], checkpoint["NT"], checkpoint["SIGMA_X"], checkpoint["SIGMA_T"],
        use_local=False, device=device,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.model[0].weight.requires_grad_(False)
    return model, checkpoint


def u_fn_factory(model, G1, G2, d1, d2):
    """u_fn(x,t) = u_RF(x,t) + t*(d1*G1(x) + d2*G2(x)). Drop-in replacement
    for `model` wherever physics.py's residual helpers expect one -- they
    only ever call model(x,t)."""
    def u_fn(x, t):
        return model(x, t) + t * (d1 * G1(x) + d2 * G2(x))
    return u_fn


def evaluate_fixed_two_gaussian(baseline, c1, sigma1, d1, c2, sigma2, d2):
    """Evaluate a FIXED (c1,sigma1,d1,c2,sigma2,d2) correction against the
    t=1 fine-grid baseline, using evaluate_error_metrics unmodified."""
    x_full, y_full, u_rf_full = baseline["x_eval_full"], baseline["y_eval_full"], baseline["u_RF_full"]
    x_shock, y_shock, u_rf_shock = baseline["x_eval_shock"], baseline["y_eval_shock"], baseline["u_RF_shock"]
    correction_full = d1 * tfgc.gaussian(x_full, c1, sigma1) + d2 * tfgc.gaussian(x_full, c2, sigma2)
    correction_shock = d1 * tfgc.gaussian(x_shock, c1, sigma1) + d2 * tfgc.gaussian(x_shock, c2, sigma2)
    u_pred_full = u_rf_full + correction_full
    u_pred_shock = u_rf_shock + correction_shock
    metrics = evaluate_error_metrics(x_full, y_full, u_pred_full, y_shock, u_pred_shock)
    return metrics, correction_full, correction_shock


def compute_oracle_amplitudes(model, device, c1, sigma1, c2, sigma2):
    """Recompute the post-hoc shock-optimal (d1,d2) fresh via closed-form
    least squares on the canonical 200-point t=1 shock grid -- NOT
    hardcoded, NOT used for initialization/training, reference only."""
    baseline = ptgc.build_search_baseline(model, device)
    e_shock = baseline["y_eval_shock"] - baseline["u_RF_shock"]
    q1_shock = tfgc.gaussian(baseline["x_eval_shock"], c1, sigma1)
    q2_shock = tfgc.gaussian(baseline["x_eval_shock"], c2, sigma2)
    Q_shock = np.column_stack([q1_shock, q2_shock])
    d_oracle, rank, s = dsf.lstsq_solve(Q_shock, e_shock)
    cond = float(s[0] / s[-1]) if s[-1] > 0 else float("inf")
    d1_oracle, d2_oracle = float(d_oracle[0]), float(d_oracle[1])

    metrics, correction_full, correction_shock = evaluate_fixed_two_gaussian(
        baseline, c1, sigma1, d1_oracle, c2, sigma2, d2_oracle
    )

    return {
        "d1_oracle": d1_oracle, "d2_oracle": d2_oracle,
        "rank": int(rank), "cond": cond,
        "metrics": metrics, "baseline": baseline,
        "correction_full": correction_full, "correction_shock": correction_shock,
    }


def verify_pretraining_state(model, checkpoint, device, u_fn, d1, d2):
    """Before Stage 2 training begins: verify `model` has ZERO trainable
    parameters, verify d1,d2 both require grad and start at exactly 0, and
    verify the d1=d2=0 forward pass (via u_fn) reproduces the canonical
    RF-only metrics. Raises and stops before training on any failure."""
    print("model.named_parameters() (expected: ALL requires_grad=False):")
    model_trainable = []
    for name, p in model.named_parameters():
        print(f"  {name}: shape={tuple(p.shape)}, requires_grad={p.requires_grad}")
        if p.requires_grad:
            model_trainable.append(name)
    if model_trainable:
        raise RuntimeError(
            f"Expected model to have ZERO trainable parameters, got {model_trainable}. "
            "Refusing to proceed -- Stage 2 would not be a clean frozen-RF experiment."
        )
    print("Confirmed: model has zero trainable parameters.")

    if not (d1.requires_grad and d2.requires_grad):
        raise RuntimeError(f"Expected d1.requires_grad and d2.requires_grad both True, got {d1.requires_grad}, {d2.requires_grad}.")
    print(f"Confirmed: d1.requires_grad={d1.requires_grad}, d2.requires_grad={d2.requires_grad}")

    d1_init, d2_init = d1.item(), d2.item()
    if d1_init != 0.0 or d2_init != 0.0:
        raise RuntimeError(f"Expected d1==d2==0.0 at initialization, got d1={d1_init}, d2={d2_init}.")
    print(f"Confirmed: d1=={d1_init}, d2=={d2_init} at initialization.")

    nv = checkpoint.get("NV", NV)
    x_test, t_test = make_test_grid(M_TEST, N_TEST_TIMES, device)
    y_true = make_reference_values(t_test, x_test, nv)
    x_shock, t_shock = build_shock_grid(t_test, device)
    y_true_shock = make_reference_values(t_shock, x_shock, nv)

    with torch.no_grad():
        y_pred = u_fn(x_test, t_test).cpu().numpy().reshape(-1)
        y_pred_shock = u_fn(x_shock, t_shock).cpu().numpy().reshape(-1)
    metrics = evaluate_error_metrics(x_test, y_true, y_pred, y_true_shock, y_pred_shock)

    def rel_diff(a, b):
        return abs(a - b) / abs(b)

    checks = {
        "global": rel_diff(metrics["global_relative_l2_error"], CANONICAL_GLOBAL),
        "smooth": rel_diff(metrics["smooth_relative_l2_error"], CANONICAL_SMOOTH),
        "shock": rel_diff(metrics["shock_relative_l2_error"], CANONICAL_SHOCK),
    }
    print("d1=d2=0 pre-training verification vs. canonical metrics:")
    for k, v in checks.items():
        print(f"  {k}: relative difference = {v:.3e} (tolerance {VERIFY_RTOL:.1e})")
    failed = {k: v for k, v in checks.items() if v > VERIFY_RTOL}
    if failed:
        raise RuntimeError(f"Pre-training verification FAILED: {failed}. Refusing to start Stage 2 training.")
    print("Pre-training verification PASSED.")
    return metrics


def reconstruct_interior_points(checkpoint, device="cpu"):
    """Reproduce the EXACT seed-42 interior collocation points, same
    approach as reconstruct_pde_trained_rf_checkpoint.py /
    diagnose_fixed_rf_linear_d_sweep.py / test_two_stage_linear_local_feature.py."""
    set_seed(checkpoint["seed"])
    x_train, t_train = sample_training_points(checkpoint["M_TRAIN"], device)
    return x_train, t_train


def snapshot_frozen_tensors(model):
    return {
        "Wx": model.Wx.clone(),
        "Wt": model.Wt.clone(),
        "bx": model.bx.clone(),
        "bt": model.bt.clone(),
        "model.0.weight": model.model[0].weight.clone(),
    }


def check_frozen_invariance(model, before):
    diffs = {}
    after = {
        "Wx": model.Wx,
        "Wt": model.Wt,
        "bx": model.bx,
        "bt": model.bt,
        "model.0.weight": model.model[0].weight,
    }
    for key in before:
        diffs[key] = torch.max(torch.abs(after[key] - before[key])).item()
    return diffs


def train_stage2(u_fn, d1, d2, x_train, t_train, device):
    """Stage-2 training loop. Mathematically identical to physics.train's
    loop body -- reimplemented locally (not a modification to physics.py)
    solely to expose per-epoch d1,d2/loss components and the pre-first-step
    gradient of each."""
    optimizer = optim.Adam([d1, d2], lr=LR, weight_decay=WEIGHT_DECAY)

    history = []
    grad_d1_initial = None
    grad_d2_initial = None
    d1_after_first_step = None
    d2_after_first_step = None

    for epoch in range(EPOCHS):
        optimizer.zero_grad()

        residual_pde = burgers_residual(u_fn, x_train, t_train, NV)
        l_pde = torch.mean(residual_pde**2)
        residual_bd = boundary_residual(u_fn, M_BD, device)
        l_bc = torch.mean(residual_bd**2)
        residual_int = initial_residual(u_fn, M_INT, g, device)
        l_ic = torch.mean(residual_int**2)
        loss = l_pde + IC_BC_WEIGHT * (l_ic + l_bc)

        # retain_graph=True matches physics.train's own loss.backward() call
        # exactly -- required because x_train/t_train are reused leaf
        # tensors across epochs.
        loss.backward(retain_graph=True)

        if epoch == 0:
            grad_d1_initial = d1.grad.item()
            grad_d2_initial = d2.grad.item()

        optimizer.step()

        d1_now, d2_now = d1.item(), d2.item()
        if epoch == 0:
            d1_after_first_step, d2_after_first_step = d1_now, d2_now

        history.append({
            "epoch": epoch, "d1": d1_now, "d2": d2_now,
            "pde_loss": l_pde.item(), "ic_loss": l_ic.item(), "bc_loss": l_bc.item(),
            "total_loss": loss.item(),
        })

        if epoch % 100 == 0:
            print(f"epoch {epoch}: d1={d1_now:.10f}, d2={d2_now:.10f}, pde_loss={l_pde.item():.6e}, total_loss={loss.item():.6e}")

    return history, grad_d1_initial, grad_d2_initial, d1_after_first_step, d2_after_first_step


def write_history_csv(history):
    header = ["epoch", "d1", "d2", "pde_loss", "ic_loss", "bc_loss", "total_loss"]
    path = OUTPUT_DIR / "two_stage_two_gaussian_training_history.csv"
    with path.open("w") as f:
        f.write(",".join(header) + "\n")
        for row in history:
            f.write(",".join(str(row[k]) for k in header) + "\n")


def plot_d_vs_epoch(history, d1_oracle, d2_oracle):
    epochs = [r["epoch"] for r in history]
    d1_values = [r["d1"] for r in history]
    d2_values = [r["d2"] for r in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, d1_values, "-", color="tab:blue", label="d1 (learned)")
    ax.plot(epochs, d2_values, "-", color="tab:orange", label="d2 (learned)")
    ax.axhline(d1_oracle, color="tab:blue", linestyle="--", linewidth=1, label=f"d1_oracle={d1_oracle:.4g}")
    ax.axhline(d2_oracle, color="tab:orange", linestyle="--", linewidth=1, label=f"d2_oracle={d2_oracle:.4g}")
    ax.axhline(0.0, color="k", linewidth=0.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("amplitude")
    ax.set_title("Stage-2 learned d1,d2 vs epoch (frozen RF, fixed G1,G2 geometry)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "d1_d2_vs_epoch.png")
    plt.close(fig)


def plot_t1_shock_solution(baseline, oracle_info, correction_pde_shock):
    x_shock = baseline["x_eval_shock"]
    y_shock = baseline["y_eval_shock"]
    u_rf_shock = baseline["u_RF_shock"]
    u_oracle = u_rf_shock + oracle_info["correction_shock"]
    u_pde = u_rf_shock + correction_pde_shock

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(x_shock, y_shock, "k-", linewidth=2, label="u_true")
    ax.plot(x_shock, u_rf_shock, "--", label="u_RF-only")
    ax.plot(x_shock, u_oracle, ":", label="u_RF + oracle K=2")
    ax.plot(x_shock, u_pde, "-", label="u_RF + PDE-trained K=2")
    ax.set_title("t=1 shock solution: true vs RF-only vs oracle vs PDE-trained")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "t1_shock_solution_comparison.png")
    plt.close(fig)


def plot_t1_shock_residual(baseline, oracle_info, correction_pde_shock):
    x_shock = baseline["x_eval_shock"]
    e_rf = baseline["y_eval_shock"] - baseline["u_RF_shock"]
    e_oracle = e_rf - oracle_info["correction_shock"]
    e_pde = e_rf - correction_pde_shock

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(x_shock, e_rf, "k-", linewidth=2, label="e_RF = u_true - u_RF")
    ax.plot(x_shock, e_oracle, ":", label="e_RF - oracle correction")
    ax.plot(x_shock, e_pde, "-", label="e_RF - PDE-trained correction")
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.set_title("t=1 shock residual: RF-only vs oracle-corrected vs PDE-trained-corrected")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "t1_shock_residual_comparison.png")
    plt.close(fig)


def plot_local_contribution(baseline, oracle_info, correction_pde_shock):
    x_shock = baseline["x_eval_shock"]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(x_shock, oracle_info["correction_shock"], "--", label="oracle local contribution (t=1)")
    ax.plot(x_shock, correction_pde_shock, "-", label="PDE-trained local contribution (t=1)")
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.set_title("Learned vs oracle local contribution at t=1")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "local_contribution_comparison.png")
    plt.close(fig)


def write_summary(
    c1, sigma1, c2, sigma2,
    oracle_info, grad_d1_initial, grad_d2_initial,
    d1_after_first_step, d2_after_first_step,
    d1_final, d2_final,
    final_metrics_canonical, metrics_pde_t1,
    ic_relative_l2, ratio_full, ratio_shock,
    frozen_diffs, history,
):
    d1_oracle, d2_oracle = oracle_info["d1_oracle"], oracle_info["d2_oracle"]
    update_dir_d1 = -grad_d1_initial
    update_dir_d2 = -grad_d2_initial

    lines = []
    lines.append("=== Fixed two-Gaussian geometry (exact library entries) ===")
    lines.append(f"c1 = {c1}, sigma1 = {sigma1}")
    lines.append(f"c2 = {c2}, sigma2 = {sigma2}")
    lines.append("")

    lines.append("=== Post-hoc oracle amplitudes (reference only, recomputed fresh, NOT used for init/training) ===")
    lines.append(f"d1_oracle = {d1_oracle}")
    lines.append(f"d2_oracle = {d2_oracle}")
    lines.append(f"rank = {oracle_info['rank']}, condition number = {oracle_info['cond']}")
    lines.append(f"oracle t=1 metrics: {oracle_info['metrics']}")
    lines.append("")

    lines.append("=== 1. Initial-gradient sign interpretation (at d1=d2=0) ===")
    lines.append(f"grad_d1 = dL/dd1 = {grad_d1_initial}  (sign: {sign_str(grad_d1_initial)})")
    lines.append(f"grad_d2 = dL/dd2 = {grad_d2_initial}  (sign: {sign_str(grad_d2_initial)})")
    lines.append("gradient-descent update direction = -grad:")
    lines.append(f"  -grad_d1 = {update_dir_d1}  (sign: {sign_str(update_dir_d1)})")
    lines.append(f"  -grad_d2 = {update_dir_d2}  (sign: {sign_str(update_dir_d2)})")
    lines.append(
        f"d1_oracle sign: {sign_str(d1_oracle)}  -- agrees with update direction: "
        f"{sign_str(update_dir_d1) == sign_str(d1_oracle)}"
    )
    lines.append(
        f"d2_oracle sign: {sign_str(d2_oracle)}  -- agrees with update direction: "
        f"{sign_str(update_dir_d2) == sign_str(d2_oracle)}"
    )
    lines.append("")

    lines.append("=== 2. First optimizer step ===")
    lines.append(f"d1 after epoch 0 = {d1_after_first_step}  (sign: {sign_str(d1_after_first_step)})")
    lines.append(f"d2 after epoch 0 = {d2_after_first_step}  (sign: {sign_str(d2_after_first_step)})")
    moved_d1 = (
        "toward oracle sign" if sign_str(d1_after_first_step) == sign_str(d1_oracle)
        else ("away from oracle sign" if d1_after_first_step != 0 else "no movement (zero)")
    )
    moved_d2 = (
        "toward oracle sign" if sign_str(d2_after_first_step) == sign_str(d2_oracle)
        else ("away from oracle sign" if d2_after_first_step != 0 else "no movement (zero)")
    )
    lines.append(f"first step for d1 moved: {moved_d1}")
    lines.append(f"first step for d2 moved: {moved_d2}")
    lines.append("")

    lines.append("=== 3. Final amplitude comparison ===")
    lines.append(f"final d1 (epoch {len(history) - 1}) = {d1_final}")
    lines.append(f"final d2 (epoch {len(history) - 1}) = {d2_final}")
    if d1_oracle != 0:
        lines.append(f"d1_PDE / d1_oracle = {d1_final / d1_oracle}")
        lines.append(f"|d1_PDE| / |d1_oracle| = {abs(d1_final) / abs(d1_oracle)}")
    else:
        lines.append("d1_oracle == 0: ratio undefined.")
    if d2_oracle != 0:
        lines.append(f"d2_PDE / d2_oracle = {d2_final / d2_oracle}")
        lines.append(f"|d2_PDE| / |d2_oracle| = {abs(d2_final) / abs(d2_oracle)}")
    else:
        lines.append("d2_oracle == 0: ratio undefined.")
    lines.append("")

    lines.append("=== Final metrics ===")
    lines.append("canonical multi-time metrics:")
    for k, v in final_metrics_canonical.items():
        lines.append(f"  {k} = {v}")
    lines.append("t=1-only metrics (directly comparable to the oracle above, NOT the canonical multi-time metric):")
    for k, v in metrics_pde_t1.items():
        lines.append(f"  {k} = {v}")
    lines.append(f"ic_relative_l2_error = {ic_relative_l2}")
    lines.append(f"ratio_full (canonical multi-time grid) = {ratio_full}")
    lines.append(f"ratio_shock (canonical multi-time grid) = {ratio_shock}")
    lines.append("")

    lines.append("=== Frozen-RF invariance check (max |after-before|, expected exactly 0.0) ===")
    for k, v in frozen_diffs.items():
        lines.append(f"  {k}: {v}")
    lines.append("")

    lines.append("=== Comparison ===")
    lines.append("A. RF-only baseline (Stage 1): d1=d2=0")
    lines.append(
        f"B. post-hoc/oracle K=2 (t=1): d1={d1_oracle}, d2={d2_oracle}, "
        f"shock_error={oracle_info['metrics']['shock_relative_l2_error']}"
    )
    lines.append(
        f"C. PDE-trained frozen-RF K=2 (t=1): d1={d1_final}, d2={d2_final}, "
        f"shock_error={metrics_pde_t1['shock_relative_l2_error']}"
    )
    lines.append(
        "The post-hoc amplitudes (B) minimize t=1 solution error on the shock "
        "region; PDE training (C) minimizes a multi-time PDE/IC/BC objective. "
        "These are NOT expected to be mathematically identical. Their signs "
        "and orders of magnitude remain scientifically important -- see the "
        "three separate questions below."
    )
    lines.append("")

    lines.append("=== 4. Three separate questions (do NOT infer C from A or B) ===")
    lines.append(
        "A. REPRESENTATION: can this fixed two-Gaussian basis represent the "
        "shock residual? Already established by the post-hoc oracle fit "
        f"(oracle shock error = {oracle_info['metrics']['shock_relative_l2_error']:.6f} at t=1) -- "
        "see RESEARCH_LOG.md Sections 17-20 and the K=2 pair-search / greedy "
        "K-Gaussian diagnostics for the full representation-capacity evidence. "
        "NOT re-derived or re-tested in this script."
    )
    b_yes_d1 = sign_str(update_dir_d1) == sign_str(d1_oracle)
    b_yes_d2 = sign_str(update_dir_d2) == sign_str(d2_oracle)
    lines.append(
        "B. DIRECTION: does the PDE objective initially push d1,d2 toward the "
        f"useful (oracle) signs? d1: {'YES' if b_yes_d1 else 'NO'}. d2: {'YES' if b_yes_d2 else 'NO'}."
    )
    ratio_d1_str = f"{(d1_final / d1_oracle):.6f}" if d1_oracle != 0 else "undefined (oracle is zero)"
    ratio_d2_str = f"{(d2_final / d2_oracle):.6f}" if d2_oracle != 0 else "undefined (oracle is zero)"
    lines.append(
        "C. AMPLITUDE: does PDE training learn amplitudes comparable to the "
        f"useful post-hoc amplitudes? d1_PDE/d1_oracle = {ratio_d1_str}, "
        f"d2_PDE/d2_oracle = {ratio_d2_str} -- see Section 3 above for the full "
        "ratio comparison. C is answered independently of A and B: a correct "
        "direction (B) does not imply a useful amplitude (C), and "
        "representational sufficiency (A) does not imply either B or C."
    )
    lines.append("")
    lines.append(
        f"NOTE: WEIGHT_DECAY={WEIGHT_DECAY} is large and (for vanilla Adam, not "
        "decoupled AdamW) adds weight_decay*d directly to the gradient for EACH "
        "of d1,d2 independently, in proportion to its own current magnitude -- "
        "preserved unchanged per the established convention, not treated as a "
        "confound to remove, but relevant when interpreting how far d1,d2 were "
        "able to move from zero."
    )
    lines.append(
        "NOTE: ic_relative_l2_error is, by construction, IDENTICAL for any "
        "(d1,d2) here, since u_local(x,0) = 0*(...) = 0 exactly -- a structural "
        "guarantee, not a trained outcome (same as Section 23's IC-independence "
        "finding for the single-Gaussian case)."
    )

    (OUTPUT_DIR / "two_stage_two_gaussian_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    torch.set_default_dtype(dsf.DTYPE)
    device = "cpu"

    model, checkpoint = load_frozen_rf_model(device)
    ptgc.verify_baseline(model, checkpoint, device)

    c1 = find_exact_library_value(C1_NOMINAL, tfgc.C_VALUES, GEOMETRY_MATCH_RTOL, "c1")
    sigma1 = find_exact_library_value(SIGMA1_NOMINAL, tfgc.SIGMA_VALUES, GEOMETRY_MATCH_RTOL, "sigma1")
    c2 = find_exact_library_value(C2_NOMINAL, tfgc.C_VALUES, GEOMETRY_MATCH_RTOL, "c2")
    sigma2 = find_exact_library_value(SIGMA2_NOMINAL, tfgc.SIGMA_VALUES, GEOMETRY_MATCH_RTOL, "sigma2")
    print(f"Fixed geometry: c1={c1}, sigma1={sigma1}, c2={c2}, sigma2={sigma2}")

    print("Computing post-hoc oracle amplitudes (reference only, NOT used for init/training)...")
    oracle_info = compute_oracle_amplitudes(model, device, c1, sigma1, c2, sigma2)
    print(f"d1_oracle={oracle_info['d1_oracle']}, d2_oracle={oracle_info['d2_oracle']}, "
          f"rank={oracle_info['rank']}, cond={oracle_info['cond']}")
    print(f"oracle t=1 metrics: {oracle_info['metrics']}")

    G1 = GaussianLocal(center=c1, width=sigma1)
    G2 = GaussianLocal(center=c2, width=sigma2)
    d1 = torch.nn.Parameter(torch.zeros(1))
    d2 = torch.nn.Parameter(torch.zeros(1))
    u_fn = u_fn_factory(model, G1, G2, d1, d2)

    verify_pretraining_state(model, checkpoint, device, u_fn, d1, d2)

    x_train, t_train = reconstruct_interior_points(checkpoint, device)
    frozen_before = snapshot_frozen_tensors(model)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    history, grad_d1_initial, grad_d2_initial, d1_after_first_step, d2_after_first_step = train_stage2(
        u_fn, d1, d2, x_train, t_train, device
    )

    frozen_diffs = check_frozen_invariance(model, frozen_before)

    write_history_csv(history)
    plot_d_vs_epoch(history, oracle_info["d1_oracle"], oracle_info["d2_oracle"])

    d1_final, d2_final = d1.item(), d2.item()

    nv = checkpoint.get("NV", NV)
    x_test, t_test = make_test_grid(M_TEST, N_TEST_TIMES, device)
    y_true = make_reference_values(t_test, x_test, nv)
    x_shock, t_shock = build_shock_grid(t_test, device)
    y_true_shock = make_reference_values(t_shock, x_shock, nv)

    model.eval()
    with torch.no_grad():
        y_pred = u_fn(x_test, t_test).cpu().numpy().reshape(-1)
        y_pred_shock = u_fn(x_shock, t_shock).cpu().numpy().reshape(-1)
    final_metrics_canonical = evaluate_error_metrics(x_test, y_true, y_pred, y_true_shock, y_pred_shock)

    x_ic = torch.linspace(-1, 1, IC_GRID_N_X).reshape(-1, 1).to(device)
    t_ic = torch.zeros_like(x_ic).to(device)
    y_ic_true = g(x_ic).detach().cpu().numpy().reshape(-1)
    with torch.no_grad():
        y_ic_pred = u_fn(x_ic, t_ic).cpu().numpy().reshape(-1)
    ic_relative_l2 = relative_l2(y_ic_true, y_ic_pred)

    with torch.no_grad():
        u_rf_test = model(x_test, t_test)
        u_rf_shock_canon = model(x_shock, t_shock)
        local_test = t_test * (d1_final * G1(x_test) + d2_final * G2(x_test))
        local_shock_canon = t_shock * (d1_final * G1(x_shock) + d2_final * G2(x_shock))
    ratio_full = float(torch.linalg.norm(local_test) / torch.linalg.norm(u_rf_test))
    ratio_shock = float(torch.linalg.norm(local_shock_canon) / torch.linalg.norm(u_rf_shock_canon))

    baseline = oracle_info["baseline"]
    metrics_pde_t1, correction_pde_full, correction_pde_shock = evaluate_fixed_two_gaussian(
        baseline, c1, sigma1, d1_final, c2, sigma2, d2_final
    )

    plot_t1_shock_solution(baseline, oracle_info, correction_pde_shock)
    plot_t1_shock_residual(baseline, oracle_info, correction_pde_shock)
    plot_local_contribution(baseline, oracle_info, correction_pde_shock)

    write_summary(
        c1, sigma1, c2, sigma2,
        oracle_info, grad_d1_initial, grad_d2_initial,
        d1_after_first_step, d2_after_first_step,
        d1_final, d2_final,
        final_metrics_canonical, metrics_pde_t1,
        ic_relative_l2, ratio_full, ratio_shock,
        frozen_diffs, history,
    )

    print("[test_two_stage_two_gaussian_local_feature] done.")


if __name__ == "__main__":
    main()
