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

from burgers_rf.evaluation import evaluate_error_metrics  # noqa: E402
import test_direct_spatial_fit as dsf  # noqa: E402 -- reused unmodified: RF realization, reference solution, grids, lstsq

# Pure diagnostic: NO Adam, NO PDE residual, NO training, NO changes to any
# trained model. Tests whether a FIXED (untrained) Gaussian correction
# q(x,t;c,sigma) = t * G(x;c,sigma) has the right spatial shape to reduce
# the discrepancy between an already-fitted RF-only approximation and the
# true Burgers snapshot at t=1.
#
# BASELINE: the dense overdetermined least-squares RF-only fit from
# test_direct_spatial_fit.py's Experiment B (seed-42 Wx/bx, M_fit=4001,
# t=1) -- a meaningful, already-fitted, non-trivial RF baseline. No PDE-
# trained model is available locally (no checkpoint exists, and training
# is explicitly out of scope for this diagnostic), so this is the
# unambiguous choice given the task's own constraints.
#
# SCOPING NOTE: everything here is evaluated at the single fixed snapshot
# t=1 (to reuse the baseline above), so q(x,1;c,sigma) = G(x;c,sigma)
# exactly -- the "t*" factor is a constant multiplier of 1 at this
# snapshot and has no effect on any number reported here.

C_VALUES = np.linspace(-0.02, 0.02, 21)
SIGMA_VALUES = np.logspace(np.log10(1e-4), np.log10(3e-1), 25)

OUTPUT_DIR = PROJECT_DIR / "outputs" / "fixed_gaussian_correction"


def gaussian(x, c, sigma):
    return np.exp(-((x - c) ** 2) / (2 * sigma**2))


def solution_ylim(*arrays, margin=0.05):
    """Axis range determined ONLY by the physically relevant curves passed
    in (never by a possibly-pathological correction term on its own) --
    the corrected-plotting lesson from test_direct_spatial_fit.py."""
    vals = np.concatenate(arrays)
    lo, hi = float(vals.min()), float(vals.max())
    pad = margin * (hi - lo) if hi > lo else 1.0
    return lo - pad, hi + pad


def build_baseline(device):
    Wx_t, bx_t = dsf.get_rf_params(device)

    x_fit_B = np.linspace(-1, 1, dsf.M_FIT_B)
    A_fit_RF = dsf.phi_x_matrix(x_fit_B, Wx_t, bx_t)
    y_fit = dsf.reference_values(x_fit_B, dsf.T_SNAPSHOT)
    c_RF, rank_rf, s_rf = dsf.lstsq_solve(A_fit_RF, y_fit)

    x_eval_full = np.linspace(-1, 1, dsf.EVAL_FULL_N)
    t_probe = torch.tensor([[dsf.T_SNAPSHOT]], dtype=dsf.DTYPE)
    x_shock_t, _ = dsf.build_shock_grid(t_probe, device)
    x_eval_shock = x_shock_t.detach().cpu().numpy().reshape(-1)

    y_eval_full = dsf.reference_values(x_eval_full, dsf.T_SNAPSHOT)
    y_eval_shock = dsf.reference_values(x_eval_shock, dsf.T_SNAPSHOT)

    A_eval_full_RF = dsf.phi_x_matrix(x_eval_full, Wx_t, bx_t)
    A_eval_shock_RF = dsf.phi_x_matrix(x_eval_shock, Wx_t, bx_t)
    u_RF_full = A_eval_full_RF @ c_RF
    u_RF_shock = A_eval_shock_RF @ c_RF

    return {
        "c_RF": c_RF,
        "rank_rf": rank_rf,
        "x_eval_full": x_eval_full,
        "x_eval_shock": x_eval_shock,
        "y_eval_full": y_eval_full,
        "y_eval_shock": y_eval_shock,
        "u_RF_full": u_RF_full,
        "u_RF_shock": u_RF_shock,
    }


def evaluate_grid(baseline):
    x_eval_full = baseline["x_eval_full"]
    x_eval_shock = baseline["x_eval_shock"]
    y_eval_full = baseline["y_eval_full"]
    y_eval_shock = baseline["y_eval_shock"]
    u_RF_full = baseline["u_RF_full"]
    u_RF_shock = baseline["u_RF_shock"]

    e_full = y_eval_full - u_RF_full
    e_shock = y_eval_shock - u_RF_shock

    rows = []
    for c in C_VALUES:
        for sigma in SIGMA_VALUES:
            q_full = gaussian(x_eval_full, c, sigma)
            q_shock = gaussian(x_eval_shock, c, sigma)

            # A. fixed +q, B. fixed -q -- no fitting, no optimizer.
            m_plus = evaluate_error_metrics(
                x_eval_full, y_eval_full, u_RF_full + q_full, y_eval_shock, u_RF_shock + q_shock
            )
            m_minus = evaluate_error_metrics(
                x_eval_full, y_eval_full, u_RF_full - q_full, y_eval_shock, u_RF_shock - q_shock
            )

            # C. oracle scalar amplitude: closed-form LS solution, NOT an optimizer.
            d_star_global = float(np.dot(q_full, e_full) / np.dot(q_full, q_full))
            d_star_shock = float(np.dot(q_shock, e_shock) / np.dot(q_shock, q_shock))

            m_oracle_global = evaluate_error_metrics(
                x_eval_full, y_eval_full, u_RF_full + d_star_global * q_full,
                y_eval_shock, u_RF_shock + d_star_global * q_shock,
            )
            m_oracle_shock = evaluate_error_metrics(
                x_eval_full, y_eval_full, u_RF_full + d_star_shock * q_full,
                y_eval_shock, u_RF_shock + d_star_shock * q_shock,
            )

            corr_global = float(np.dot(q_full, e_full) / (np.linalg.norm(q_full) * np.linalg.norm(e_full)))
            corr_shock = float(np.dot(q_shock, e_shock) / (np.linalg.norm(q_shock) * np.linalg.norm(e_shock)))

            rows.append({
                "c": c, "sigma": sigma,
                "plus_global": m_plus["global_relative_l2_error"],
                "plus_smooth": m_plus["smooth_relative_l2_error"],
                "plus_shock": m_plus["shock_relative_l2_error"],
                "minus_global": m_minus["global_relative_l2_error"],
                "minus_smooth": m_minus["smooth_relative_l2_error"],
                "minus_shock": m_minus["shock_relative_l2_error"],
                "d_star_global": d_star_global,
                "oracle_global_global": m_oracle_global["global_relative_l2_error"],
                "oracle_global_smooth": m_oracle_global["smooth_relative_l2_error"],
                "oracle_global_shock": m_oracle_global["shock_relative_l2_error"],
                "d_star_shock": d_star_shock,
                "oracle_shock_global": m_oracle_shock["global_relative_l2_error"],
                "oracle_shock_smooth": m_oracle_shock["smooth_relative_l2_error"],
                "oracle_shock_shock": m_oracle_shock["shock_relative_l2_error"],
                "corr_global": corr_global,
                "corr_shock": corr_shock,
            })
    return rows


GRID_HEADER = [
    "c", "sigma",
    "plus_global", "plus_smooth", "plus_shock",
    "minus_global", "minus_smooth", "minus_shock",
    "d_star_global", "oracle_global_global", "oracle_global_smooth", "oracle_global_shock",
    "d_star_shock", "oracle_shock_global", "oracle_shock_smooth", "oracle_shock_shock",
    "corr_global", "corr_shock",
]


def write_grid_csv(rows):
    path = OUTPUT_DIR / "fixed_gaussian_correction_grid.csv"
    with path.open("w") as f:
        f.write(",".join(GRID_HEADER) + "\n")
        for row in rows:
            f.write(",".join(str(row[k]) for k in GRID_HEADER) + "\n")


def best_row(rows, key, minimize=True):
    return min(rows, key=lambda r: r[key]) if minimize else max(rows, key=lambda r: abs(r[key]))


def pct_improvement(baseline_val, new_val):
    return 100 * (baseline_val - new_val) / baseline_val


def write_summary(baseline_metrics, rows, baseline):
    lines = []
    lines.append("BASELINE (uncorrected u_RF, dense LS fit, Experiment B convention, t=1):")
    for k, v in baseline_metrics.items():
        lines.append(f"  {k} = {v}")
    lines.append("")

    base_global = baseline_metrics["global_relative_l2_error"]
    base_shock = baseline_metrics["shock_relative_l2_error"]

    # (label, sort_key to minimize, result-triplet prefix, d_star field or None)
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
        r = best_row(rows, sort_key)
        lines.append(f"{label}:")
        lines.append(f"  c = {r['c']:.6f}, sigma = {r['sigma']:.6e}")
        if d_key is not None:
            lines.append(f"  d_star = {r[d_key]:.6e}")
        g, s, sh = r[f"{prefix}_global"], r[f"{prefix}_smooth"], r[f"{prefix}_shock"]
        lines.append(f"  global_error = {g:.6f}  (baseline {base_global:.6f}, {pct_improvement(base_global, g):+.3f}% )")
        lines.append(f"  smooth_error = {s:.6f}")
        lines.append(f"  shock_error  = {sh:.6f}  (baseline {base_shock:.6f}, {pct_improvement(base_shock, sh):+.3f}% )")
        lines.append("")

    max_corr_global = max(rows, key=lambda r: abs(r["corr_global"]))
    max_corr_shock = max(rows, key=lambda r: abs(r["corr_shock"]))
    lines.append(f"max |corr_global| = {max_corr_global['corr_global']:.6f} at c={max_corr_global['c']:.6f}, sigma={max_corr_global['sigma']:.6e}")
    lines.append(f"max |corr_shock|  = {max_corr_shock['corr_shock']:.6f} at c={max_corr_shock['c']:.6f}, sigma={max_corr_shock['sigma']:.6e}")

    (OUTPUT_DIR / "fixed_gaussian_correction_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def plot_case(name, c, sigma, d_amp, baseline):
    x_full, x_shock = baseline["x_eval_full"], baseline["x_eval_shock"]
    y_full, y_shock = baseline["y_eval_full"], baseline["y_eval_shock"]
    u_rf_full, u_rf_shock = baseline["u_RF_full"], baseline["u_RF_shock"]

    q_full = gaussian(x_full, c, sigma)
    q_shock = gaussian(x_shock, c, sigma)
    corr_full = d_amp * q_full
    corr_shock = d_amp * q_shock
    u_corr_full = u_rf_full + corr_full
    u_corr_shock = u_rf_shock + corr_shock

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(x_full, y_full, "k-", linewidth=2, label="u_true")
    ax.plot(x_full, u_rf_full, "--", label="u_RF (baseline)")
    ax.plot(x_full, corr_full, ":", label=f"correction (d={d_amp:.4g})")
    ax.plot(x_full, u_corr_full, "-", label="u_RF + correction")
    ax.set_ylim(solution_ylim(y_full, u_rf_full, u_corr_full))
    ax.set_title(f"{name}: c={c:.4g}, sigma={sigma:.4g}, full domain")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{name}_full.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(x_shock, y_shock, "k-", linewidth=2, label="u_true")
    ax.plot(x_shock, u_rf_shock, "--", label="u_RF (baseline)")
    ax.plot(x_shock, corr_shock, ":", label=f"correction (d={d_amp:.4g})")
    ax.plot(x_shock, u_corr_shock, "-", label="u_RF + correction")
    ax.set_ylim(solution_ylim(y_shock, u_rf_shock, u_corr_shock))
    ax.set_title(f"{name}: c={c:.4g}, sigma={sigma:.4g}, shock zoom")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{name}_shock.png")
    plt.close(fig)


def plot_heatmap(rows, value_key, title, filename):
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
    torch.set_default_dtype(dsf.DTYPE)
    device = "cpu"

    baseline = build_baseline(device)
    baseline_metrics = evaluate_error_metrics(
        baseline["x_eval_full"], baseline["y_eval_full"], baseline["u_RF_full"],
        baseline["y_eval_shock"], baseline["u_RF_shock"],
    )
    print(f"Baseline RF: rank={baseline['rank_rf']}, ||c_RF||={np.linalg.norm(baseline['c_RF']):.4e}")
    print(f"Baseline metrics: {baseline_metrics}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = evaluate_grid(baseline)
    write_grid_csv(rows)
    write_summary(baseline_metrics, rows, baseline)

    plot_heatmap(
        rows,
        "corr_shock",
        "|corr_shock| shape-alignment diagnostic",
        "heatmap_corr_shock.png",
    )
    plot_heatmap(rows, "d_star_shock", "d_star_shock", "heatmap_d_star_shock.png")

    best_corr = max(rows, key=lambda r: abs(r["corr_shock"]))
    best_oracle_shock = min(rows, key=lambda r: r["oracle_shock_shock"])
    plot_case("best_corr_shock", best_corr["c"], best_corr["sigma"], best_corr["d_star_shock"], baseline)
    plot_case("best_oracle_shock", best_oracle_shock["c"], best_oracle_shock["sigma"], best_oracle_shock["d_star_shock"], baseline)

    print("[fixed_gaussian_correction] done.")


if __name__ == "__main__":
    main()
