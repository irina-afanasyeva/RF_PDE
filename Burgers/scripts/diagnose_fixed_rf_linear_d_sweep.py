import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(PROJECT_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from burgers_rf.config import (  # noqa: E402
    DTYPE,
    IC_BC_WEIGHT,
    M_BD,
    M_INT,
    M_TEST,
    N_TEST_TIMES,
    NV,
    set_seed,
)
from burgers_rf.data import g, make_reference_values, make_test_grid, sample_training_points  # noqa: E402
from burgers_rf.model import BurgersRF  # noqa: E402
from burgers_rf.local_features import GaussianLocal  # noqa: E402
from burgers_rf.physics import boundary_residual, burgers_residual, initial_residual  # noqa: E402
from burgers_rf.evaluation import evaluate_error_metrics  # noqa: E402
from sweep_local_width import build_shock_grid  # noqa: E402
import test_direct_spatial_fit as dsf  # noqa: E402 -- reused: EVAL_FULL_N, T_SNAPSHOT, reference_values

# Diagnostic ONLY: no PDE training, no optimizer, no modification of the
# fixed PDE-trained RF checkpoint. Determines whether the POSITIVE sign of
# the linear_t local coefficient learned by joint PDE training
# (d_joint=+0.02335906311371956) is already preferred by the fixed-RF
# PDE/IC/BC objective itself (CASE A), opposed by it (CASE B), or weakly
# identified (CASE C) -- by directly sweeping d with the RF held fixed at
# its checkpoint_complete.pt values.
#
# physics.burgers_residual / boundary_residual / initial_residual are reused
# COMPLETELY UNMODIFIED: they only ever call `model(x, t)`, so a plain
# closure u_fn(x,t) = u_RF(x,t) + d*t*G(x) is a drop-in replacement for the
# "model" argument -- no loss/residual math is duplicated.
#
# d_posthoc_t1_shock=-0.6578595 below is quoted ONLY as reference
# information (from the t=1 post-hoc solution-error fit); it is never used
# as an initialization or training target anywhere in this script.

CHECKPOINT_PATH = PROJECT_DIR / "outputs" / "pde_trained_rf_residual" / "checkpoint_complete.pt"
CANONICAL_GLOBAL = 0.27149430095802507
CANONICAL_SMOOTH = 0.2443827084159696
CANONICAL_SHOCK = 0.8390487998889169
VERIFY_RTOL = 1e-4

CENTER = 0.014
WIDTH = 0.007646112

# Range chosen to comfortably contain both reference d values with margin:
# G(x) peaks at 1 and q=t*G(x) has |q|<=1, d_joint~0.023 is tiny, d_posthoc
# ~-0.658 is order-1 -- [-1.5, 1.5] covers both with room to spare. If the
# sweep's minimizing d lands at either boundary, main() prints a warning
# rather than silently trusting this range.
D_MIN = -1.5
D_MAX = 1.5
D_N_POINTS = 601

D_REFERENCE_JOINT = 0.02335906311371956
D_REFERENCE_POSTHOC = -0.6578595

FD_H = 1e-3  # finite-difference step for the r0/r1/r2 decomposition check
             # and the numerical dL_PDE/dd|0 cross-check

# Fixed-per-evaluation seeds for BD/IC points: boundary_residual/
# initial_residual redraw their points via torch.rand on every call, which
# would inject resampling noise unrelated to d into the sweep. Resetting to
# a fixed seed before each call (every d) makes the BD batch and the IC
# batch identical across the whole sweep, isolating the effect of d alone.
# Two distinct seeds are used because both draw a same-shape (500,1) tensor
# -- reusing one seed for both would make the two batches numerically
# identical to each other, which is unnecessary and avoidable.
BD_SEED = 42
IC_SEED = 43

OUTPUT_DIR = PROJECT_DIR / "outputs" / "fixed_rf_linear_d_sweep"


def load_fixed_rf_model(device="cpu"):
    """Load the complete, verified PDE-trained RF checkpoint. No training,
    no optimizer, no parameter modification anywhere in this script."""
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model = BurgersRF(
        checkpoint["NX"], checkpoint["NT"], checkpoint["SIGMA_X"], checkpoint["SIGMA_T"],
        use_local=checkpoint["use_local"], device=device,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def verify_baseline(model, checkpoint, device):
    """Recompute canonical global/smooth/shock metrics at d=0 and compare
    against the hardcoded canonical reference. Raises and stops BEFORE the
    sweep if verification fails."""
    nv = checkpoint.get("NV", NV)
    x_test, t_test = make_test_grid(M_TEST, N_TEST_TIMES, device)
    y_true = make_reference_values(t_test, x_test, nv)
    x_shock, t_shock = build_shock_grid(t_test, device)
    y_true_shock = make_reference_values(t_shock, x_shock, nv)

    with torch.no_grad():
        y_pred = model(x_test, t_test).cpu().numpy().reshape(-1)
        y_pred_shock = model(x_shock, t_shock).cpu().numpy().reshape(-1)
    metrics = evaluate_error_metrics(x_test, y_true, y_pred, y_true_shock, y_pred_shock)

    def rel_diff(a, b):
        return abs(a - b) / abs(b)

    checks = {
        "global": rel_diff(metrics["global_relative_l2_error"], CANONICAL_GLOBAL),
        "smooth": rel_diff(metrics["smooth_relative_l2_error"], CANONICAL_SMOOTH),
        "shock": rel_diff(metrics["shock_relative_l2_error"], CANONICAL_SHOCK),
    }
    print("d=0 baseline verification vs. canonical metrics:")
    for k, v in checks.items():
        print(f"  {k}: relative difference = {v:.3e} (tolerance {VERIFY_RTOL:.1e})")
    failed = {k: v for k, v in checks.items() if v > VERIFY_RTOL}
    if failed:
        raise RuntimeError(
            f"d=0 baseline verification FAILED: {failed}. Refusing to proceed with "
            "the d sweep until the checkpoint reproduces canonical metrics."
        )
    print("Baseline verification PASSED.")


def reconstruct_interior_points(checkpoint, device="cpu"):
    """Reproduce the EXACT seed-42 interior collocation points (x_train,
    t_train) used by the original canonical PDE training run, using the
    checkpoint's own recorded seed/M_TRAIN -- same reconstruction approach
    already verified in reconstruct_pde_trained_rf_checkpoint.py. Model
    construction is not needed here (checkpoint_complete.pt already
    supplies a complete state_dict), so only set_seed + sample_training_
    points are replayed, in that order."""
    set_seed(checkpoint["seed"])
    x_train, t_train = sample_training_points(checkpoint["M_TRAIN"], device)
    return x_train, t_train


def make_u_fn(model, G, d):
    """Drop-in replacement for a BurgersRF model: u_fn(x, t) = u_RF(x, t) +
    d * t * G(x). physics.py's residual helpers only ever call
    `model(x, t)`, so this callable can be passed to them unmodified."""
    def u_fn(x, t):
        return model(x, t) + d * t * G(x)
    return u_fn


def compute_losses(model, G, d, x_train, t_train, device):
    """PDE / IC / BC / weighted-total losses at a fixed d, reusing
    physics.py's residual functions UNMODIFIED via the make_u_fn adapter.
    IC_BC_WEIGHT combination mirrors physics.loss_fn's own formula exactly."""
    u_fn = make_u_fn(model, G, d)

    residual_pde = burgers_residual(u_fn, x_train, t_train, NV)
    l_pde = torch.mean(residual_pde**2).item()

    set_seed(BD_SEED)
    residual_bd = boundary_residual(u_fn, M_BD, device)
    l_bc = torch.mean(residual_bd**2).item()

    set_seed(IC_SEED)
    residual_int = initial_residual(u_fn, M_INT, g, device)
    l_ic = torch.mean(residual_int**2).item()

    l_total = l_pde + IC_BC_WEIGHT * (l_ic + l_bc)
    return l_pde, l_ic, l_bc, l_total


def compute_solution_errors(u_fn, x_global, t_global, y_true_global, x_shock, t_shock, y_true_shock):
    with torch.no_grad():
        y_pred = u_fn(x_global, t_global).cpu().numpy().reshape(-1)
        y_pred_shock = u_fn(x_shock, t_shock).cpu().numpy().reshape(-1)
    return evaluate_error_metrics(x_global, y_true_global, y_pred, y_true_shock, y_pred_shock)


def residual_decomposition(model, G, x, t, nv):
    """Analytic r0 + d*r1 + d^2*r2 decomposition of the Burgers residual for
    u = u_RF + d*t*G(x), derived from physics.burgers_residual's own
    convention r = u_t + u*u_x - nv*u_xx:

        u_t  = u_RF_t + d*G(x)
        u_x  = u_RF_x + d*t*G'(x)
        u_xx = u_RF_xx + d*t*G''(x)
        u*u_x = u_RF*u_RF_x
                + d*(u_RF*t*G'(x) + t*G(x)*u_RF_x)
                + d^2*(t*G(x) * t*G'(x))

    collecting powers of d:
        r0 = u_RF_t + u_RF*u_RF_x - nv*u_RF_xx
        r1 = G(x) + u_RF*t*G'(x) + t*G(x)*u_RF_x - nv*t*G''(x)
           = q_t + u_RF*q_x + q*u_RF_x - nv*q_xx      (q = t*G(x))
        r2 = t^2 * G(x) * G'(x) = q*q_x

    G'(x)/G''(x) are obtained via autograd on q directly (not hand-coded),
    so this is verified against the actual computational graph.
    """
    u_rf = model(x, t)
    u_rf_t = torch.autograd.grad(u_rf, t, grad_outputs=torch.ones_like(u_rf), create_graph=True)[0]
    u_rf_x = torch.autograd.grad(u_rf, x, grad_outputs=torch.ones_like(u_rf), create_graph=True)[0]
    u_rf_xx = torch.autograd.grad(u_rf_x, x, grad_outputs=torch.ones_like(u_rf_x), create_graph=True)[0]
    r0 = u_rf_t + u_rf * u_rf_x - nv * u_rf_xx

    q = t * G(x)
    q_t = torch.autograd.grad(q, t, grad_outputs=torch.ones_like(q), create_graph=True)[0]
    q_x = torch.autograd.grad(q, x, grad_outputs=torch.ones_like(q), create_graph=True)[0]
    q_xx = torch.autograd.grad(q_x, x, grad_outputs=torch.ones_like(q_x), create_graph=True)[0]
    r1 = q_t + u_rf * q_x + q * u_rf_x - nv * q_xx
    r2 = q * q_x

    return r0.detach(), r1.detach(), r2.detach()


def verify_decomposition_numerically(model, G, x_train, t_train, r0, r1, r2, d_check_values):
    """Verify r(d) == r0 + d*r1 + d^2*r2 by comparing the quadratic
    reconstruction against the full nonlinear residual computed directly
    from the UNMODIFIED burgers_residual, at several d values."""
    results = {}
    for d in d_check_values:
        u_fn = make_u_fn(model, G, d)
        residual_full = burgers_residual(u_fn, x_train, t_train, NV).detach()
        residual_quad = r0 + d * r1 + d**2 * r2
        max_abs_diff = torch.max(torch.abs(residual_full - residual_quad)).item()
        results[d] = max_abs_diff
    return results


def verify_ic_bc_exact_d_independence(model, G, device):
    """Confirm (not just assume) that L_IC is exactly d-independent (since
    q(x,0)=0 for all x identically) and check how strongly L_BC depends on
    d (G(+-1) is expected to be astronomically small / exactly underflowed
    to 0.0 in float64, given center=0.014, width=0.007646112)."""
    g_minus1 = float(G(torch.tensor([[-1.0]], dtype=DTYPE)).item())
    g_plus1 = float(G(torch.tensor([[1.0]], dtype=DTYPE)).item())

    probe_d_values = [0.0, D_REFERENCE_JOINT, D_REFERENCE_POSTHOC, 1.0, -1.0]
    ic_losses = []
    bc_losses = []
    for d in probe_d_values:
        u_fn = make_u_fn(model, G, d)
        set_seed(IC_SEED)
        residual_int = initial_residual(u_fn, M_INT, g, device)
        ic_losses.append(torch.mean(residual_int**2).item())
        set_seed(BD_SEED)
        residual_bd = boundary_residual(u_fn, M_BD, device)
        bc_losses.append(torch.mean(residual_bd**2).item())

    return {
        "G(-1)": g_minus1,
        "G(+1)": g_plus1,
        "probe_d_values": probe_d_values,
        "ic_losses": ic_losses,
        "bc_losses": bc_losses,
        "ic_losses_identical": len(set(ic_losses)) == 1,
        "bc_losses_identical": len(set(bc_losses)) == 1,
    }


def write_csv(rows):
    header = [
        "d",
        "pde_loss",
        "ic_loss_unweighted",
        "bc_loss_unweighted",
        "total_loss_weighted",
        "global_relative_l2_error",
        "smooth_relative_l2_error",
        "shock_relative_l2_error",
        "t1_global_relative_l2_error",
        "t1_smooth_relative_l2_error",
        "t1_shock_relative_l2_error",
    ]
    path = OUTPUT_DIR / "fixed_rf_linear_d_sweep.csv"
    with path.open("w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(row[k]) for k in header) + "\n")


def best_d(rows, key):
    r = min(rows, key=lambda row: row[key])
    return r["d"], r[key]


def write_summary(rows, decomposition_stats, ic_bc_check):
    d_values = [r["d"] for r in rows]
    boundary_warning = []
    for label, key in [
        ("pde_loss", "pde_loss"),
        ("total_loss_weighted", "total_loss_weighted"),
        ("global_relative_l2_error", "global_relative_l2_error"),
        ("shock_relative_l2_error", "shock_relative_l2_error"),
        ("t1_global_relative_l2_error", "t1_global_relative_l2_error"),
        ("t1_shock_relative_l2_error", "t1_shock_relative_l2_error"),
    ]:
        d_star, val = best_d(rows, key)
        if d_star in (d_values[0], d_values[-1]):
            boundary_warning.append(
                f"  WARNING: minimizing d for {label} is at the sweep boundary "
                f"(d={d_star}) -- range [{D_MIN}, {D_MAX}] may be too narrow."
            )

    lines = []
    lines.append(f"Sweep range: d in [{D_MIN}, {D_MAX}], {D_N_POINTS} points")
    lines.append("")
    lines.append("Minimizing d for each objective:")
    for label, key in [
        ("PDE loss", "pde_loss"),
        ("weighted total training loss", "total_loss_weighted"),
        ("canonical global solution error", "global_relative_l2_error"),
        ("canonical shock solution error", "shock_relative_l2_error"),
        ("t=1 global solution error", "t1_global_relative_l2_error"),
        ("t=1 shock solution error", "t1_shock_relative_l2_error"),
    ]:
        d_star, val = best_d(rows, key)
        lines.append(f"  {label}: d* = {d_star:.6f}  (value = {val:.6e})")
    if boundary_warning:
        lines.append("")
        lines.extend(boundary_warning)
    lines.append("")

    def row_at(d_target):
        return min(rows, key=lambda row: abs(row["d"] - d_target))

    lines.append("Values at reference d:")
    for label, d_ref in [
        ("d = 0", 0.0),
        ("d_joint = +0.02335906311371956", D_REFERENCE_JOINT),
        ("d_posthoc_t1_shock = -0.6578595", D_REFERENCE_POSTHOC),
    ]:
        r = row_at(d_ref)
        lines.append(f"  {label} (nearest sampled d={r['d']:.6f}):")
        for key in [
            "pde_loss", "ic_loss_unweighted", "bc_loss_unweighted", "total_loss_weighted",
            "global_relative_l2_error", "smooth_relative_l2_error", "shock_relative_l2_error",
            "t1_global_relative_l2_error", "t1_smooth_relative_l2_error", "t1_shock_relative_l2_error",
        ]:
            lines.append(f"    {key} = {r[key]:.6e}")
    lines.append("")

    lines.append("Residual decomposition r(d) = r0 + d*r1 + d^2*r2 (interior collocation points):")
    for k, v in decomposition_stats.items():
        lines.append(f"  {k} = {v}")
    lines.append("")

    lines.append("IC/BC exact-d-independence check:")
    lines.append(f"  G(-1) = {ic_bc_check['G(-1)']:.6e}")
    lines.append(f"  G(+1) = {ic_bc_check['G(+1)']:.6e}")
    lines.append(f"  probe d values = {ic_bc_check['probe_d_values']}")
    lines.append(f"  ic_losses = {ic_bc_check['ic_losses']}")
    lines.append(f"  ic_losses identical across all probe d: {ic_bc_check['ic_losses_identical']}")
    lines.append(f"  bc_losses = {ic_bc_check['bc_losses']}")
    lines.append(f"  bc_losses identical across all probe d: {ic_bc_check['bc_losses_identical']}")

    (OUTPUT_DIR / "fixed_rf_linear_d_sweep_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def plot_metric_vs_d(d_values, y_values, ylabel, title, filename):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(d_values, y_values, "-")
    ax.axvline(0.0, color="k", linestyle="--", linewidth=1, label="d=0")
    ax.axvline(D_REFERENCE_JOINT, color="tab:blue", linestyle=":", linewidth=1.5, label="d_joint")
    ax.axvline(D_REFERENCE_POSTHOC, color="tab:red", linestyle=":", linewidth=1.5, label="d_posthoc")
    ax.set_xlabel("d")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename)
    plt.close(fig)


def main():
    torch.set_default_dtype(DTYPE)
    device = "cpu"  # diagnostic only: checkpoint loaded with map_location="cpu" for portability

    model, checkpoint = load_fixed_rf_model(device)
    verify_baseline(model, checkpoint, device)

    x_train, t_train = reconstruct_interior_points(checkpoint, device)
    G = GaussianLocal(center=CENTER, width=WIDTH)

    # canonical multi-time grids
    x_test, t_test = make_test_grid(M_TEST, N_TEST_TIMES, device)
    y_true = make_reference_values(t_test, x_test, NV)
    x_shock, t_shock = build_shock_grid(t_test, device)
    y_true_shock = make_reference_values(t_shock, x_shock, NV)

    # t=1-only grids (same convention as test_pde_trained_gaussian_correction.py)
    x_eval_full_np = np.linspace(-1, 1, dsf.EVAL_FULL_N)
    y_eval_full_t1 = dsf.reference_values(x_eval_full_np, dsf.T_SNAPSHOT)
    x_eval_full_t1 = torch.tensor(x_eval_full_np.reshape(-1, 1), dtype=DTYPE)
    t_eval_full_t1 = torch.full_like(x_eval_full_t1, dsf.T_SNAPSHOT)

    t_probe = torch.tensor([[dsf.T_SNAPSHOT]], dtype=DTYPE)
    x_eval_shock_t1, _ = build_shock_grid(t_probe, device)
    y_eval_shock_t1 = dsf.reference_values(x_eval_shock_t1.detach().cpu().numpy().reshape(-1), dsf.T_SNAPSHOT)
    t_eval_shock_t1 = torch.full_like(x_eval_shock_t1, dsf.T_SNAPSHOT)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- residual decomposition (r0, r1, r2 are d-independent by construction) ---
    r0, r1, r2 = residual_decomposition(model, G, x_train, t_train, NV)
    inner_r0_r1 = torch.mean(r0 * r1).item()
    norm_r1_sq = torch.mean(r1 * r1).item()
    inner_r0_r2 = torch.mean(r0 * r2).item()
    dL_pde_dd_at_0_analytic = 2.0 * inner_r0_r1

    l_pde_plus, _, _, _ = compute_losses(model, G, FD_H, x_train, t_train, device)
    l_pde_minus, _, _, _ = compute_losses(model, G, -FD_H, x_train, t_train, device)
    dL_pde_dd_at_0_numeric = (l_pde_plus - l_pde_minus) / (2 * FD_H)

    decomposition_verification = verify_decomposition_numerically(
        model, G, x_train, t_train, r0, r1, r2,
        d_check_values=[FD_H, -FD_H, D_REFERENCE_JOINT, D_REFERENCE_POSTHOC],
    )

    decomposition_stats = {
        "<r0,r1> (mean over interior points)": inner_r0_r1,
        "||r1||^2 (mean over interior points)": norm_r1_sq,
        "<r0,r2> (mean over interior points)": inner_r0_r2,
        "dL_PDE/dd|0 analytic (2*<r0,r1>)": dL_pde_dd_at_0_analytic,
        "dL_PDE/dd|0 numeric (finite difference)": dL_pde_dd_at_0_numeric,
        "dL_PDE/dd|0 sign": "negative (d increases)" if dL_pde_dd_at_0_analytic < 0 else "positive (d decreases)",
        "max|residual_full - (r0+d*r1+d^2*r2)| at probed d": decomposition_verification,
    }

    ic_bc_check = verify_ic_bc_exact_d_independence(model, G, device)

    # --- full sweep ---
    d_values = np.linspace(D_MIN, D_MAX, D_N_POINTS)
    rows = []
    for d in d_values:
        d = float(d)
        l_pde, l_ic, l_bc, l_total = compute_losses(model, G, d, x_train, t_train, device)
        u_fn = make_u_fn(model, G, d)
        metrics = compute_solution_errors(u_fn, x_test, t_test, y_true, x_shock, t_shock, y_true_shock)
        metrics_t1 = compute_solution_errors(
            u_fn, x_eval_full_t1, t_eval_full_t1, y_eval_full_t1, x_eval_shock_t1, t_eval_shock_t1, y_eval_shock_t1
        )
        rows.append({
            "d": d,
            "pde_loss": l_pde,
            "ic_loss_unweighted": l_ic,
            "bc_loss_unweighted": l_bc,
            "total_loss_weighted": l_total,
            "global_relative_l2_error": metrics["global_relative_l2_error"],
            "smooth_relative_l2_error": metrics["smooth_relative_l2_error"],
            "shock_relative_l2_error": metrics["shock_relative_l2_error"],
            "t1_global_relative_l2_error": metrics_t1["global_relative_l2_error"],
            "t1_smooth_relative_l2_error": metrics_t1["smooth_relative_l2_error"],
            "t1_shock_relative_l2_error": metrics_t1["shock_relative_l2_error"],
        })

    write_csv(rows)
    write_summary(rows, decomposition_stats, ic_bc_check)

    plot_metric_vs_d([r["d"] for r in rows], [r["pde_loss"] for r in rows], "PDE loss", "PDE loss vs d (fixed RF)", "pde_loss_vs_d.png")
    plot_metric_vs_d([r["d"] for r in rows], [r["total_loss_weighted"] for r in rows], "weighted total training loss", "Weighted total loss vs d (fixed RF)", "total_loss_vs_d.png")
    plot_metric_vs_d([r["d"] for r in rows], [r["global_relative_l2_error"] for r in rows], "global relative L2 error", "Canonical global error vs d (fixed RF)", "global_error_vs_d.png")
    plot_metric_vs_d([r["d"] for r in rows], [r["shock_relative_l2_error"] for r in rows], "shock relative L2 error", "Canonical shock error vs d (fixed RF)", "shock_error_vs_d.png")
    plot_metric_vs_d([r["d"] for r in rows], [r["t1_global_relative_l2_error"] for r in rows], "t=1 global relative L2 error", "t=1 global error vs d (fixed RF)", "t1_global_error_vs_d.png")
    plot_metric_vs_d([r["d"] for r in rows], [r["t1_shock_relative_l2_error"] for r in rows], "t=1 shock relative L2 error", "t=1 shock error vs d (fixed RF)", "t1_shock_error_vs_d.png")

    print("[diagnose_fixed_rf_linear_d_sweep] done.")


if __name__ == "__main__":
    main()
