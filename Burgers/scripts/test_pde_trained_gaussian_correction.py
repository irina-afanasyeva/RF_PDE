import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(PROJECT_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from burgers_rf.config import DTYPE, M_TEST, N_TEST_TIMES, NV  # noqa: E402
from burgers_rf.model import BurgersRF  # noqa: E402
from burgers_rf.data import make_reference_values, make_test_grid  # noqa: E402
from burgers_rf.evaluation import evaluate_error_metrics  # noqa: E402
from sweep_local_width import build_shock_grid  # noqa: E402
import test_direct_spatial_fit as dsf  # noqa: E402 -- reused: EVAL_FULL_N, T_SNAPSHOT, reference_values
import test_fixed_gaussian_correction as tfgc  # noqa: E402 -- reused unchanged: gaussian, solution_ylim,
# evaluate_grid, GRID_HEADER, best_row, pct_improvement, C_VALUES, SIGMA_VALUES

# Diagnostic of the FIXED, already PDE-trained RF baseline (checkpoint loaded
# from disk). NO PDE training, NO optimizer, NO modification of the
# checkpoint/model parameters anywhere in this script. Repeats the Sept-30
# fixed/shifted Gaussian correction search (test_fixed_gaussian_correction.py)
# but with u_RF taken from PDE training instead of the direct least-squares
# fit, per the advisor's instruction "take u_RF from training, not direct
# fit." This first controlled comparison is performed ONLY at t=1, changing
# exactly one variable (the source of u_RF) relative to the Sept-30 result.

CHECKPOINT_PATH = PROJECT_DIR / "outputs" / "pde_trained_rf_residual" / "checkpoint_complete.pt"
CANONICAL_GLOBAL = 0.27149430095802507
CANONICAL_SHOCK = 0.8390487998889169
VERIFY_RTOL = 1e-4

OUTPUT_DIR = PROJECT_DIR / "outputs" / "pde_trained_gaussian_correction"


def load_pde_trained_model(device="cpu"):
    """Reconstruct BurgersRF entirely from checkpoint metadata and load the
    saved state_dict. No training, no optimizer, no parameter changes."""
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model = BurgersRF(
        checkpoint["NX"], checkpoint["NT"], checkpoint["SIGMA_X"], checkpoint["SIGMA_T"],
        use_local=checkpoint["use_local"], device=device,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def verify_baseline(model, checkpoint, device):
    """Recompute canonical global/shock metrics and compare against both the
    hardcoded canonical reference and the checkpoint's own saved metrics.
    Raises and stops BEFORE the Gaussian search if verification fails."""
    nv = checkpoint.get("NV", NV)

    x_test, t_test = make_test_grid(M_TEST, N_TEST_TIMES, device)
    y_true = make_reference_values(t_test, x_test, nv)
    x_shock, t_shock = build_shock_grid(t_test, device)
    y_true_shock = make_reference_values(t_shock, x_shock, nv)

    y_pred = model(x_test, t_test).cpu().detach().numpy().reshape(-1)
    y_pred_shock = model(x_shock, t_shock).cpu().detach().numpy().reshape(-1)
    metrics = evaluate_error_metrics(x_test, y_true, y_pred, y_true_shock, y_pred_shock)

    def rel_diff(a, b):
        return abs(a - b) / abs(b)

    checks = {
        "global_vs_canonical": rel_diff(metrics["global_relative_l2_error"], CANONICAL_GLOBAL),
        "shock_vs_canonical": rel_diff(metrics["shock_relative_l2_error"], CANONICAL_SHOCK),
    }
    if "global_relative_l2_error" in checkpoint:
        checks["global_vs_checkpoint"] = rel_diff(
            metrics["global_relative_l2_error"], checkpoint["global_relative_l2_error"]
        )
    if "shock_relative_l2_error" in checkpoint:
        checks["shock_vs_checkpoint"] = rel_diff(
            metrics["shock_relative_l2_error"], checkpoint["shock_relative_l2_error"]
        )

    print("Baseline verification:")
    for k, v in checks.items():
        print(f"  {k}: relative difference = {v:.3e} (tolerance {VERIFY_RTOL:.1e})")

    failed = {k: v for k, v in checks.items() if v > VERIFY_RTOL}
    if failed:
        raise RuntimeError(
            f"Baseline verification FAILED: {failed}. Refusing to proceed with the "
            "Gaussian correction search until the checkpoint reproduces the canonical "
            "PDE-trained baseline within tolerance."
        )
    print("Baseline verification PASSED.")
    return metrics


def build_search_baseline(model, device):
    """Same evaluation grids/convention as test_fixed_gaussian_correction.py
    (EVAL_FULL_N, canonical shock grid, t=T_SNAPSHOT=1 only), but u_RF comes
    from the loaded PDE-trained model instead of a least-squares fit."""
    x_eval_full = np.linspace(-1, 1, dsf.EVAL_FULL_N)
    t_probe = torch.tensor([[dsf.T_SNAPSHOT]], dtype=DTYPE)
    x_shock_t, _ = build_shock_grid(t_probe, device)
    x_eval_shock = x_shock_t.detach().cpu().numpy().reshape(-1)

    y_eval_full = dsf.reference_values(x_eval_full, dsf.T_SNAPSHOT)
    y_eval_shock = dsf.reference_values(x_eval_shock, dsf.T_SNAPSHOT)

    x_eval_full_t = torch.tensor(x_eval_full.reshape(-1, 1), dtype=DTYPE)
    t_eval_full_t = torch.full_like(x_eval_full_t, dsf.T_SNAPSHOT)
    x_eval_shock_t = torch.tensor(x_eval_shock.reshape(-1, 1), dtype=DTYPE)
    t_eval_shock_t = torch.full_like(x_eval_shock_t, dsf.T_SNAPSHOT)

    # Pure forward evaluation of the ALREADY-trained, fixed model. No
    # training, no optimizer, no gradient step, no parameter modification.
    with torch.no_grad():
        _, u_rf_full_t, _ = model.forward_components(x_eval_full_t, t_eval_full_t)
        _, u_rf_shock_t, _ = model.forward_components(x_eval_shock_t, t_eval_shock_t)
    u_RF_full = u_rf_full_t.detach().cpu().numpy().reshape(-1)
    u_RF_shock = u_rf_shock_t.detach().cpu().numpy().reshape(-1)

    return {
        "x_eval_full": x_eval_full,
        "x_eval_shock": x_eval_shock,
        "y_eval_full": y_eval_full,
        "y_eval_shock": y_eval_shock,
        "u_RF_full": u_RF_full,
        "u_RF_shock": u_RF_shock,
    }


def write_grid_csv(rows):
    """Locally scoped (writes ONLY to this script's OUTPUT_DIR) -- not
    imported from test_fixed_gaussian_correction.py, which closes over its
    own OUTPUT_DIR and would overwrite the Sept-30 results."""
    path = OUTPUT_DIR / "pde_trained_gaussian_correction_grid.csv"
    with path.open("w") as f:
        f.write(",".join(tfgc.GRID_HEADER) + "\n")
        for row in rows:
            f.write(",".join(str(row[k]) for k in tfgc.GRID_HEADER) + "\n")


def write_summary(baseline_verification_metrics, baseline_metrics, rows):
    lines = []
    lines.append("Baseline verification (PDE-trained checkpoint vs. canonical reference):")
    lines.append(f"  canonical global = {CANONICAL_GLOBAL}")
    lines.append(f"  canonical shock  = {CANONICAL_SHOCK}")
    lines.append(f"  recomputed global = {baseline_verification_metrics['global_relative_l2_error']}")
    lines.append(f"  recomputed shock  = {baseline_verification_metrics['shock_relative_l2_error']}")
    lines.append("")
    lines.append("BASELINE used for the Gaussian search (PDE-trained u_RF, t=1, same grids as Sept-30 diagnostic):")
    for k, v in baseline_metrics.items():
        lines.append(f"  {k} = {v}")
    lines.append("")

    base_global = baseline_metrics["global_relative_l2_error"]
    base_shock = baseline_metrics["shock_relative_l2_error"]

    objectives = [
        ("minimum global error, fixed +q", "plus_global", "plus", None),
        ("minimum shock error, fixed +q", "plus_shock", "plus", None),
        ("minimum global error, fixed -q", "minus_global", "minus", None),
        ("minimum shock error, fixed -q", "minus_shock", "minus", None),
        ("minimum global error, d_star_global", "oracle_global_global", "oracle_global", "d_star_global"),
        ("minimum shock error, d_star_global", "oracle_global_shock", "oracle_global", "d_star_global"),
        ("minimum global error, d_star_shock", "oracle_shock_global", "oracle_shock", "d_star_shock"),
        ("minimum shock error, d_star_shock", "oracle_shock_shock", "oracle_shock", "d_star_shock"),
    ]
    for label, sort_key, prefix, d_key in objectives:
        r = tfgc.best_row(rows, sort_key)
        lines.append(f"{label}:")
        lines.append(f"  c = {r['c']:.6f}, sigma = {r['sigma']:.6e}")
        if d_key is not None:
            lines.append(f"  d_star = {r[d_key]:.6e}")
        g, s, sh = r[f"{prefix}_global"], r[f"{prefix}_smooth"], r[f"{prefix}_shock"]
        lines.append(f"  global_error = {g:.6f}  (baseline {base_global:.6f}, {tfgc.pct_improvement(base_global, g):+.3f}% )")
        lines.append(f"  smooth_error = {s:.6f}")
        lines.append(f"  shock_error  = {sh:.6f}  (baseline {base_shock:.6f}, {tfgc.pct_improvement(base_shock, sh):+.3f}% )")
        lines.append("")

    max_corr_global = max(rows, key=lambda r: abs(r["corr_global"]))
    max_corr_shock = max(rows, key=lambda r: abs(r["corr_shock"]))
    lines.append(f"max |corr_global| = {max_corr_global['corr_global']:.6f} at c={max_corr_global['c']:.6f}, sigma={max_corr_global['sigma']:.6e}")
    lines.append(f"max |corr_shock|  = {max_corr_shock['corr_shock']:.6f} at c={max_corr_shock['c']:.6f}, sigma={max_corr_shock['sigma']:.6e}")

    (OUTPUT_DIR / "pde_trained_gaussian_correction_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def plot_case(name, c, sigma, d_amp, baseline):
    """Locally scoped (writes ONLY to this script's OUTPUT_DIR)."""
    x_full, x_shock = baseline["x_eval_full"], baseline["x_eval_shock"]
    y_full, y_shock = baseline["y_eval_full"], baseline["y_eval_shock"]
    u_rf_full, u_rf_shock = baseline["u_RF_full"], baseline["u_RF_shock"]

    q_full = tfgc.gaussian(x_full, c, sigma)
    q_shock = tfgc.gaussian(x_shock, c, sigma)
    corr_full = d_amp * q_full
    corr_shock = d_amp * q_shock
    u_corr_full = u_rf_full + corr_full
    u_corr_shock = u_rf_shock + corr_shock

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(x_full, y_full, "k-", linewidth=2, label="u_true")
    ax.plot(x_full, u_rf_full, "--", label="u_RF (PDE-trained baseline)")
    ax.plot(x_full, corr_full, ":", label=f"correction (d={d_amp:.4g})")
    ax.plot(x_full, u_corr_full, "-", label="u_RF + correction")
    ax.set_ylim(tfgc.solution_ylim(y_full, u_rf_full, u_corr_full))
    ax.set_title(f"{name}: c={c:.4g}, sigma={sigma:.4g}, full domain")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{name}_full.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(x_shock, y_shock, "k-", linewidth=2, label="u_true")
    ax.plot(x_shock, u_rf_shock, "--", label="u_RF (PDE-trained baseline)")
    ax.plot(x_shock, corr_shock, ":", label=f"correction (d={d_amp:.4g})")
    ax.plot(x_shock, u_corr_shock, "-", label="u_RF + correction")
    ax.set_ylim(tfgc.solution_ylim(y_shock, u_rf_shock, u_corr_shock))
    ax.set_title(f"{name}: c={c:.4g}, sigma={sigma:.4g}, shock zoom")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{name}_shock.png")
    plt.close(fig)


def plot_heatmap(rows, value_key, title, filename):
    """Locally scoped (writes ONLY to this script's OUTPUT_DIR)."""
    c_vals = sorted(set(r["c"] for r in rows))
    s_vals = sorted(set(r["sigma"] for r in rows))
    idx_c = {c: i for i, c in enumerate(c_vals)}
    idx_s = {s: i for i, s in enumerate(s_vals)}
    grid = np.zeros((len(s_vals), len(c_vals)))
    for r in rows:
        grid[idx_s[r["sigma"]], idx_c[r["c"]]] = r[value_key]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(c_vals, s_vals, grid, shading="auto")
    ax.set_yscale("log")
    ax.set_xlabel("c")
    ax.set_ylabel("sigma (log)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename)
    plt.close(fig)


def main():
    torch.set_default_dtype(DTYPE)
    device = "cpu"  # diagnostic only: checkpoint loaded with map_location="cpu" for portability

    model, checkpoint = load_pde_trained_model(device)
    baseline_verification_metrics = verify_baseline(model, checkpoint, device)

    baseline = build_search_baseline(model, device)
    baseline_metrics = evaluate_error_metrics(
        baseline["x_eval_full"], baseline["y_eval_full"], baseline["u_RF_full"],
        baseline["y_eval_shock"], baseline["u_RF_shock"],
    )
    print(f"Search baseline metrics (t=1, PDE-trained u_RF): {baseline_metrics}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Reused UNCHANGED from test_fixed_gaussian_correction.py: same C_VALUES/
    # SIGMA_VALUES grids, same fixed +-G / oracle-amplitude / correlation logic.
    rows = tfgc.evaluate_grid(baseline)

    write_grid_csv(rows)
    write_summary(baseline_verification_metrics, baseline_metrics, rows)

    plot_heatmap(rows, "corr_shock", "|corr_shock| shape-alignment diagnostic (PDE-trained baseline)", "heatmap_corr_shock.png")
    plot_heatmap(rows, "d_star_shock", "d_star_shock (PDE-trained baseline)", "heatmap_d_star_shock.png")

    best_corr = max(rows, key=lambda r: abs(r["corr_shock"]))
    best_oracle_shock = min(rows, key=lambda r: r["oracle_shock_shock"])
    plot_case("best_corr_shock", best_corr["c"], best_corr["sigma"], best_corr["d_star_shock"], baseline)
    plot_case("best_oracle_shock", best_oracle_shock["c"], best_oracle_shock["sigma"], best_oracle_shock["d_star_shock"], baseline)

    print("[test_pde_trained_gaussian_correction] done.")


if __name__ == "__main__":
    main()
