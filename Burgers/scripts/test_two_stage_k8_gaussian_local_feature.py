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
    set_seed,
)
from burgers_rf.data import g, make_reference_values, make_test_grid  # noqa: E402
from burgers_rf.local_features import GaussianLocal  # noqa: E402
from burgers_rf.physics import boundary_residual, burgers_residual, initial_residual  # noqa: E402
from burgers_rf.evaluation import evaluate_error_metrics  # noqa: E402
from sweep_local_width import build_shock_grid  # noqa: E402
import test_pde_trained_gaussian_correction as ptgc  # noqa: E402 -- reused unchanged: build_search_baseline
import test_fixed_gaussian_correction as tfgc  # noqa: E402 -- reused unchanged: gaussian, C_VALUES, SIGMA_VALUES, solution_ylim
import test_direct_spatial_fit as dsf  # noqa: E402 -- reused unchanged: lstsq_solve, DTYPE
from test_two_stage_two_gaussian_local_feature import (  # noqa: E402 -- reused UNCHANGED (generic, not K-specific):
    sign_str,
    find_exact_library_value,
    load_frozen_rf_model,
    reconstruct_interior_points,
    snapshot_frozen_tensors,
    check_frozen_invariance,
)

# TWO-STAGE, K=8-GAUSSIAN TRAINING experiment. Structural generalization of
# test_two_stage_two_gaussian_local_feature.py (K=2) to the exact K=8
# spatial basis recovered from the greedy post-hoc capacity study
# (test_pde_trained_k_gaussian_correction.py). model.py/physics.py/
# config.py/the K=2 script itself are NOT modified; generic (non-K-specific)
# helper functions are reused directly via import from the K=2 script
# (sign_str, find_exact_library_value, load_frozen_rf_model,
# reconstruct_interior_points, snapshot_frozen_tensors,
# check_frozen_invariance) -- only the K-specific functions (u_fn
# construction, oracle computation, pretraining verification, the training
# loop, and reporting) are reimplemented here, generalized from 2 to 8
# Gaussians.
#
# MODEL: u(x,t) = u_RF(x,t) + t * sum_{k=1}^8 d_k * G_k(x), with ONLY
# d_1..d_8 trainable (free-standing nn.Parameter scalars, NOT model
# attributes) and the entire canonical Nx=500 RF frozen. Same duck-typing
# mechanism as K=2: physics.burgers_residual/boundary_residual/
# initial_residual only ever call `model(x,t)`, so u_fn is a drop-in
# replacement.
#
# CONTROLLED CHANGE vs. K=2: WEIGHT_DECAY is set to an EXPERIMENT-LOCAL 0.0
# constant, NOT imported from config.py. The completed K=2 weight-decay
# sweep (RESEARCH_LOG.md Section 28) showed WEIGHT_DECAY=1 materially
# suppresses the learned local amplitude; this experiment asks whether a
# richer K=8 basis can learn a more complete correction once that known
# suppression is removed. All other training settings (LR, EPOCHS,
# IC_BC_WEIGHT, M_TRAIN, M_BD, M_INT, seed/collocation reconstruction,
# optimizer type) are identical to the K=2 experiment.
#
# ORACLE: the recovered server K=8 amplitudes below are REFERENCE values
# only (for a sanity gate), never hardcoded as the oracle and never used to
# initialize/train. The oracle is recomputed fresh via closed-form joint
# least squares (dsf.lstsq_solve) on the SAME frozen checkpoint and t=1
# shock grid used throughout this study.

CHECKPOINT_PATH = PROJECT_DIR / "outputs" / "pde_trained_rf_residual" / "checkpoint_complete.pt"
CANONICAL_GLOBAL = 0.27149430095802507
CANONICAL_SMOOTH = 0.2443827084159696
CANONICAL_SHOCK = 0.8390487998889169
VERIFY_RTOL = 1e-4

EPOCHS = 1000
# CONTROLLED CHANGE: experiment-local, NOT imported from config.py.
WEIGHT_DECAY = 0.0

# Exact recovered greedy K=8 geometry (c, sigma), in order k=1..8. NOMINAL
# values as recovered from the server; the actual trainable geometry used
# below is the nearest EXACT C_VALUES/SIGMA_VALUES library entry (verified
# to match within GEOMETRY_MATCH_RTOL, never silently substituted).
# Repeated centers (k=1/k=3, k=2/k=4) are NOT deduplicated -- each
# (center,sigma) pair is a distinct basis function.
K8_GEOMETRY_NOMINAL = [
    (0.014000, 7.646112e-03),
    (-0.014000, 7.646112e-03),
    (0.014000, 5.477226e-03),
    (-0.014000, 3.923563e-03),
    (0.002000, 2.810610e-03),
    (-0.008000, 2.013356e-03),
    (0.018000, 1.442250e-03),
    (-0.020000, 5.301518e-04),
]
GEOMETRY_MATCH_RTOL = 1e-4

# Recovered server K=8 joint-oracle amplitudes, in the SAME k=1..8 order as
# K8_GEOMETRY_NOMINAL. REFERENCE/VALIDATION ONLY -- never hardcoded as the
# oracle, never used for initialization or training. compute_oracle_
# amplitudes_k8() below reconstructs the IDENTICAL deterministic
# computation the greedy capacity script performed at its own K=8 step
# (same checkpoint via load_frozen_rf_model, same t=1 grid via
# ptgc.build_search_baseline, same e_shock = y_eval_shock - u_RF_shock,
# same 8 geometries, same dsf.lstsq_solve machinery) -- there is no
# training/Adam/RNG anywhere in this path, so the two computations should
# agree to near machine precision. The values below are ROUNDED printed
# values from the server CSV (~7 significant figures), so ORACLE_MATCH_RTOL
# only needs to accommodate that rounding (~1e-6-1e-7 expected), not an
# independent/approximate re-derivation -- hence using the project's
# standard VERIFY_RTOL=1e-4 convention (same tolerance used everywhere else
# in this study for reproducing an identical deterministic computation),
# not a loose tolerance.
K8_ORACLE_REFERENCE_APPROX = [
    -1.184397,
    0.5111319,
    0.5883732,
    -0.1436740,
    -0.02556836,
    -0.01290866,
    0.009140194,
    0.01219014,
]
ORACLE_MATCH_RTOL = 1e-4

IC_GRID_N_X = 1000

OUTPUT_DIR = PROJECT_DIR / "outputs" / "two_stage_k8_gaussian_local_feature"


def relative_l2(y_true, y_pred):
    return float(np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true))


def verify_k8_geometry():
    """Verify every (center,sigma) pair against the EXISTING candidate
    library (tfgc.C_VALUES x tfgc.SIGMA_VALUES), same mechanism as the K=2
    script. Never silently substitutes an off-grid value."""
    geometry = []
    for k, (c_nominal, sigma_nominal) in enumerate(K8_GEOMETRY_NOMINAL, start=1):
        c = find_exact_library_value(c_nominal, tfgc.C_VALUES, GEOMETRY_MATCH_RTOL, f"c_{k}")
        sigma = find_exact_library_value(sigma_nominal, tfgc.SIGMA_VALUES, GEOMETRY_MATCH_RTOL, f"sigma_{k}")
        geometry.append((c, sigma))
    return geometry


def u_fn_factory_k8(model, Gs, ds):
    """u_fn(x,t) = u_RF(x,t) + t*sum_k(d_k*G_k(x)). Drop-in replacement for
    `model` wherever physics.py's residual helpers expect one."""
    def u_fn(x, t):
        correction = ds[0] * Gs[0](x)
        for d, G in zip(ds[1:], Gs[1:]):
            correction = correction + d * G(x)
        return model(x, t) + t * correction
    return u_fn


def evaluate_fixed_k8_gaussian(baseline, geometry, ds):
    """Evaluate a FIXED K=8 correction against the t=1 fine-grid baseline,
    using evaluate_error_metrics unmodified."""
    x_full, y_full, u_rf_full = baseline["x_eval_full"], baseline["y_eval_full"], baseline["u_RF_full"]
    x_shock, y_shock, u_rf_shock = baseline["x_eval_shock"], baseline["y_eval_shock"], baseline["u_RF_shock"]

    correction_full = np.zeros_like(u_rf_full)
    correction_shock = np.zeros_like(u_rf_shock)
    for (c, sigma), d in zip(geometry, ds):
        correction_full = correction_full + d * tfgc.gaussian(x_full, c, sigma)
        correction_shock = correction_shock + d * tfgc.gaussian(x_shock, c, sigma)

    u_pred_full = u_rf_full + correction_full
    u_pred_shock = u_rf_shock + correction_shock
    metrics = evaluate_error_metrics(x_full, y_full, u_pred_full, y_shock, u_pred_shock)
    return metrics, correction_full, correction_shock


def compute_oracle_amplitudes_k8(model, device, geometry):
    """Recompute the post-hoc shock-optimal d_1..d_8 fresh via closed-form
    JOINT least squares on the canonical 200-point t=1 shock grid -- NOT
    hardcoded, NOT used for initialization/training, reference only."""
    baseline = ptgc.build_search_baseline(model, device)
    e_shock = baseline["y_eval_shock"] - baseline["u_RF_shock"]

    Q_shock = np.column_stack([tfgc.gaussian(baseline["x_eval_shock"], c, sigma) for c, sigma in geometry])
    d_oracle, rank, s = dsf.lstsq_solve(Q_shock, e_shock)
    cond = float(s[0] / s[-1]) if s[-1] > 0 else float("inf")
    d_oracle_list = [float(v) for v in d_oracle]

    metrics, correction_full, correction_shock = evaluate_fixed_k8_gaussian(baseline, geometry, d_oracle_list)

    return {
        "d_oracle": d_oracle_list,
        "rank": int(rank), "cond": cond,
        "metrics": metrics, "baseline": baseline,
        "correction_full": correction_full, "correction_shock": correction_shock,
    }


def verify_oracle_gate(d_oracle):
    """Reproduction gate against K8_ORACLE_REFERENCE_APPROX. This recomputes
    the IDENTICAL deterministic computation (same checkpoint, same t=1
    grid, same residual, same geometries, same lstsq machinery) the greedy
    capacity script performed at its own K=8 step -- see the comment above
    K8_ORACLE_REFERENCE_APPROX. ORACLE_MATCH_RTOL=1e-4 matches the
    project's standard VERIFY_RTOL convention for exactly this situation;
    it is not a loose sanity check."""
    print("Fresh oracle vs. recovered server K=8 reference (reproduction gate):")
    failed = {}
    for k, (fresh, ref) in enumerate(zip(d_oracle, K8_ORACLE_REFERENCE_APPROX), start=1):
        rel_diff = abs(fresh - ref) / abs(ref) if ref != 0 else abs(fresh)
        status = "OK" if rel_diff <= ORACLE_MATCH_RTOL else "FAIL"
        print(f"  d_{k}: fresh={fresh:.6f}, reference~={ref:.6f}, rel_diff={rel_diff:.3e} (tol={ORACLE_MATCH_RTOL:.1e}) [{status}]")
        if rel_diff > ORACLE_MATCH_RTOL:
            failed[f"d_{k}"] = rel_diff
    if failed:
        raise RuntimeError(
            f"Oracle sanity gate FAILED: {failed}. The freshly recomputed oracle differs "
            "materially from the recovered server K=8 reference. Refusing to proceed to "
            "training -- check geometry order/checkpoint/implementation before retrying."
        )
    print("Oracle sanity gate PASSED.")


def verify_pretraining_state_k8(model, checkpoint, device, u_fn, ds):
    """Before Stage 2 training begins: verify `model` has ZERO trainable
    parameters, verify all 8 d_k require grad and start at exactly 0,
    verify u_local(x,0)=0 numerically, and verify the all-zero forward pass
    (via u_fn) reproduces the canonical RF-only metrics. Raises and stops
    before training on any failure."""
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
    print(f"Confirmed: model has zero trainable parameters (count=0).")

    if not all(d.requires_grad for d in ds):
        raise RuntimeError(f"Expected all d_k.requires_grad==True, got {[d.requires_grad for d in ds]}.")
    print(f"Confirmed: all {len(ds)} local scalars have requires_grad=True (trainable local scalars = {len(ds)}).")

    d_init = [d.item() for d in ds]
    if any(v != 0.0 for v in d_init):
        raise RuntimeError(f"Expected all d_k==0.0 at initialization, got {d_init}.")
    print(f"Confirmed: all d_k == 0.0 at initialization: {d_init}")

    # Explicit numerical check that u_local(x,0)=0 identically (not just
    # asserted by construction) -- probe a handful of x values at t=0 with
    # NONZERO ds to make the check meaningful (if ds were all zero the
    # check would trivially pass regardless of the t-factor).
    probe_x = torch.linspace(-1, 1, 11).reshape(-1, 1)
    probe_t0 = torch.zeros_like(probe_x)
    with torch.no_grad():
        saved = [d.clone() for d in ds]
        for d in ds:
            d.fill_(1.0)
        u_total_t0, u_rf_t0, _ = model.forward_components(probe_x, probe_t0)
        u_fn_t0 = u_fn(probe_x, probe_t0)
        u_local_t0 = u_fn_t0 - u_rf_t0
        max_abs_u_local_t0 = torch.max(torch.abs(u_local_t0)).item()
        for d, s in zip(ds, saved):
            d.copy_(s)
    print(f"Numerical check with all d_k=1.0 (then restored to 0): max|u_local(x,0)| = {max_abs_u_local_t0:.3e} (expected 0.0)")
    if max_abs_u_local_t0 > 1e-12:
        raise RuntimeError(
            f"u_local(x,0) is not numerically zero (max abs = {max_abs_u_local_t0:.3e}). "
            "Refusing to proceed -- the IC-compatible temporal structure is broken."
        )
    print("Confirmed: u_local(x,0) = 0 numerically.")

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
    print("all-d_k=0 pre-training verification vs. canonical metrics:")
    for k, v in checks.items():
        print(f"  {k}: relative difference = {v:.3e} (tolerance {VERIFY_RTOL:.1e})")
    failed = {k: v for k, v in checks.items() if v > VERIFY_RTOL}
    if failed:
        raise RuntimeError(f"Pre-training verification FAILED: {failed}. Refusing to start Stage 2 training.")
    print("Pre-training verification PASSED.")
    return metrics


def train_stage2_k8(u_fn, ds, x_train, t_train, device):
    """Stage-2 training loop. Mathematically identical to physics.train's
    loop body -- reimplemented locally (not a modification to physics.py)
    solely to expose per-epoch d_k/loss components and the pre-first-step
    gradient of each. WEIGHT_DECAY here is the experiment-local 0.0
    constant defined above, NOT config.WEIGHT_DECAY."""
    print(f"WEIGHT_DECAY (experiment-local, NOT from config.py) = {WEIGHT_DECAY}")
    optimizer = optim.Adam(ds, lr=LR, weight_decay=WEIGHT_DECAY)

    history = []
    grad_initial = None
    d_after_first_step = None

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
            grad_initial = [d.grad.item() for d in ds]

        optimizer.step()

        d_now = [d.item() for d in ds]
        if epoch == 0:
            d_after_first_step = d_now

        history.append({
            "epoch": epoch, "d": d_now,
            "pde_loss": l_pde.item(), "ic_loss": l_ic.item(), "bc_loss": l_bc.item(),
            "total_loss": loss.item(),
        })

        if epoch % 100 == 0:
            print(f"epoch {epoch}: d={['%.8f' % v for v in d_now]}, pde_loss={l_pde.item():.6e}, total_loss={loss.item():.6e}")

    return history, grad_initial, d_after_first_step


def write_history_csv(history):
    header = ["epoch"] + [f"d_{k}" for k in range(1, 9)] + ["pde_loss", "ic_loss", "bc_loss", "total_loss"]
    path = OUTPUT_DIR / "two_stage_k8_training_history.csv"
    with path.open("w") as f:
        f.write(",".join(header) + "\n")
        for row in history:
            values = [row["epoch"]] + row["d"] + [row["pde_loss"], row["ic_loss"], row["bc_loss"], row["total_loss"]]
            f.write(",".join(str(v) for v in values) + "\n")


def plot_d_vs_epoch_k8(history, d_oracle):
    epochs = [r["epoch"] for r in history]
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, 8))
    for k in range(8):
        d_values = [r["d"][k] for r in history]
        ax.plot(epochs, d_values, "-", color=colors[k], label=f"d_{k+1} (learned)")
        ax.axhline(d_oracle[k], color=colors[k], linestyle="--", linewidth=1)
    ax.axhline(0.0, color="k", linewidth=0.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("amplitude")
    ax.set_title("Stage-2 learned d_1..d_8 vs epoch (frozen RF, fixed K=8 geometry)\ndashed lines = fresh oracle values")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "d_vs_epoch.png")
    plt.close(fig)


def plot_t1_solution(suffix, x, y_true, u_rf, u_oracle, u_pde, title_prefix):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(x, y_true, "k-", linewidth=2, label="u_true")
    ax.plot(x, u_rf, "--", label="frozen u_RF")
    ax.plot(x, u_pde, "-", label="frozen u_RF + PDE-trained K=8")
    ax.plot(x, u_oracle, ":", label="frozen u_RF + oracle K=8")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t=1)")
    ax.set_ylim(tfgc.solution_ylim(y_true, u_rf, u_pde, u_oracle))
    ax.set_title(f"{title_prefix}, t=1")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"solution_{suffix}.png")
    plt.close(fig)


def write_summary(
    geometry, oracle_info, grad_initial, d_after_first_step, d_final,
    final_metrics_canonical, metrics_pde_t1, metrics_oracle_t1, metrics_rf_only_t1,
    frozen_diffs, history,
):
    d_oracle = oracle_info["d_oracle"]
    lines = []
    lines.append("=== Fixed K=8 geometry (exact library entries, recovered greedy capacity-study basis) ===")
    for k, (c, sigma) in enumerate(geometry, start=1):
        lines.append(f"  k={k}: c={c}, sigma={sigma}")
    lines.append("")

    lines.append("=== Fresh post-hoc oracle amplitudes (reference only, NOT used for init/training) ===")
    for k, d in enumerate(d_oracle, start=1):
        lines.append(f"  d_{k}_oracle = {d}")
    lines.append(f"rank = {oracle_info['rank']}, condition number = {oracle_info['cond']}")
    lines.append(f"oracle t=1 metrics: {oracle_info['metrics']}")
    lines.append("")

    lines.append("=== Initial-gradient sign interpretation (at all d_k=0) ===")
    for k in range(8):
        update_dir = -grad_initial[k]
        agrees = sign_str(update_dir) == sign_str(d_oracle[k])
        lines.append(
            f"  d_{k+1}: grad={grad_initial[k]:.6e} (sign {sign_str(grad_initial[k])}), "
            f"update_dir sign={sign_str(update_dir)}, oracle sign={sign_str(d_oracle[k])}, agrees={agrees}"
        )
    lines.append("")

    lines.append("=== First optimizer step ===")
    for k in range(8):
        lines.append(f"  d_{k+1} after epoch 0 = {d_after_first_step[k]} (sign: {sign_str(d_after_first_step[k])})")
    lines.append("")

    lines.append("=== Final amplitude comparison ===")
    for k in range(8):
        learned, oracle = d_final[k], d_oracle[k]
        ratio = learned / oracle if oracle != 0 else float("nan")
        abs_ratio = abs(learned) / abs(oracle) if oracle != 0 else float("nan")
        agrees = sign_str(learned) == sign_str(oracle)
        lines.append(
            f"  d_{k+1}: learned={learned}, oracle={oracle}, ratio={ratio:.6f}, "
            f"abs_ratio={abs_ratio:.6f}, sign_agrees={agrees}"
        )
    lines.append("")

    lines.append("=== Final metrics ===")
    lines.append(f"RF-only canonical (multi-time): global={CANONICAL_GLOBAL}, smooth={CANONICAL_SMOOTH}, shock={CANONICAL_SHOCK}")
    lines.append(f"corrected canonical (multi-time): {final_metrics_canonical}")
    lines.append(f"RF-only t=1 (fine grid): {metrics_rf_only_t1}")
    lines.append(f"corrected t=1 (PDE-trained K=8): {metrics_pde_t1}")
    lines.append(f"corrected t=1 (oracle K=8): {metrics_oracle_t1}")
    h_last = history[-1]
    lines.append(f"final_pde_loss = {h_last['pde_loss']}, final_ic_loss = {h_last['ic_loss']}, final_bc_loss = {h_last['bc_loss']}")
    lines.append("")

    lines.append("=== Frozen-RF invariance check (max |after-before|, expected exactly 0.0) ===")
    for k, v in frozen_diffs.items():
        lines.append(f"  {k}: {v}")
    lines.append("")

    lines.append(
        f"NOTE: WEIGHT_DECAY={WEIGHT_DECAY} for this experiment (experiment-local, NOT "
        "config.py's canonical value) -- controlled change testing whether removing the "
        "known amplitude-suppressing regularization (RESEARCH_LOG.md Section 28) lets a "
        "richer K=8 basis learn a more complete correction."
    )

    (OUTPUT_DIR / "two_stage_k8_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    torch.set_default_dtype(dsf.DTYPE)
    device = "cpu"

    geometry = verify_k8_geometry()
    print("Verified K=8 geometry (exact library entries):")
    for k, (c, sigma) in enumerate(geometry, start=1):
        print(f"  k={k}: c={c}, sigma={sigma}")

    model, checkpoint = load_frozen_rf_model(device)
    ptgc.verify_baseline(model, checkpoint, device)

    oracle_info = compute_oracle_amplitudes_k8(model, device, geometry)
    print(f"Fresh oracle d_1..d_8 = {oracle_info['d_oracle']}")
    print(f"rank={oracle_info['rank']}, cond={oracle_info['cond']}")
    verify_oracle_gate(oracle_info["d_oracle"])

    Gs = [GaussianLocal(center=c, width=sigma) for c, sigma in geometry]
    ds = [torch.nn.Parameter(torch.zeros(1)) for _ in range(8)]
    u_fn = u_fn_factory_k8(model, Gs, ds)

    verify_pretraining_state_k8(model, checkpoint, device, u_fn, ds)

    x_train, t_train = reconstruct_interior_points(checkpoint, device)
    frozen_before = snapshot_frozen_tensors(model)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    history, grad_initial, d_after_first_step = train_stage2_k8(u_fn, ds, x_train, t_train, device)

    frozen_diffs = check_frozen_invariance(model, frozen_before)

    write_history_csv(history)
    plot_d_vs_epoch_k8(history, oracle_info["d_oracle"])

    d_final = [d.item() for d in ds]

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

    baseline = oracle_info["baseline"]
    metrics_pde_t1, correction_pde_full, correction_pde_shock = evaluate_fixed_k8_gaussian(baseline, geometry, d_final)
    metrics_rf_only_t1 = evaluate_error_metrics(
        baseline["x_eval_full"], baseline["y_eval_full"], baseline["u_RF_full"],
        baseline["y_eval_shock"], baseline["u_RF_shock"],
    )

    u_oracle_full = baseline["u_RF_full"] + oracle_info["correction_full"]
    u_oracle_shock = baseline["u_RF_shock"] + oracle_info["correction_shock"]
    u_pde_full = baseline["u_RF_full"] + correction_pde_full
    u_pde_shock = baseline["u_RF_shock"] + correction_pde_shock

    plot_t1_solution(
        "full", baseline["x_eval_full"], baseline["y_eval_full"], baseline["u_RF_full"],
        u_oracle_full, u_pde_full, "Frozen-RF K=8 full-domain solution",
    )
    plot_t1_solution(
        "shock", baseline["x_eval_shock"], baseline["y_eval_shock"], baseline["u_RF_shock"],
        u_oracle_shock, u_pde_shock, "Frozen-RF K=8 shock-region solution",
    )

    write_summary(
        geometry, oracle_info, grad_initial, d_after_first_step, d_final,
        final_metrics_canonical, metrics_pde_t1, oracle_info["metrics"], metrics_rf_only_t1,
        frozen_diffs, history,
    )

    print("[test_two_stage_k8_gaussian_local_feature] done.")


if __name__ == "__main__":
    main()
