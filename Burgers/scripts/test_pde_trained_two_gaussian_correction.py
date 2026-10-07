import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(PROJECT_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from burgers_rf.evaluation import evaluate_error_metrics  # noqa: E402
import test_pde_trained_gaussian_correction as ptgc  # noqa: E402 -- reused unchanged:
# load_pde_trained_model, verify_baseline, build_search_baseline
import test_fixed_gaussian_correction as tfgc  # noqa: E402 -- reused unchanged:
# gaussian, solution_ylim, C_VALUES, SIGMA_VALUES, evaluate_grid, best_row, pct_improvement
import test_direct_spatial_fit as dsf  # noqa: E402 -- reused unchanged: lstsq_solve, DTYPE

# Post-hoc TWO-GAUSSIAN fixed-RF diagnostic. NOT PDE training: the canonical
# PDE-trained RF checkpoint is loaded and kept completely fixed throughout
# (reusing ptgc.load_pde_trained_model/verify_baseline/build_search_baseline
# unmodified); d1, d2 are solved by closed-form least squares/pseudoinverse
# (dsf.lstsq_solve) at every candidate (c1,sigma1,c2,sigma2), never by Adam
# or any optimizer. Evaluated at t=1 only, same grids/metric definitions as
# test_pde_trained_gaussian_correction.py, so results are directly
# comparable to the existing single-Gaussian diagnostic.
#
# SEARCH DESIGN -- this is a RESIDUAL-INFORMED LEFT/RIGHT TWO-GAUSSIAN
# SEARCH, NOT an unconstrained global search over all possible Gaussian
# pairs: G1's center is restricted to the non-positive half of the existing
# C_VALUES grid (c1<=0), G2's center to the non-negative half (c2>=0), each
# combined independently with the FULL existing SIGMA_VALUES grid. This
# directly encodes the known two-lobe residual structure (positive left /
# negative right, RESEARCH_LOG.md Section 15) and avoids both swapped-pair
# duplication (G1/G2 have fixed, distinct roles, not drawn from one shared
# symmetric pool) and the much larger, less-motivated full symmetric
# Cartesian square over all 525 single-Gaussian candidates.
#
# d1/d2 signs are NEVER constrained -- least squares determines them freely.

COND_REPORT_THRESHOLD = 1e8  # REPORTING/FLAGGING only -- NOT an exclusion
                              # threshold. High-condition-number winners
                              # remain eligible and are prominently flagged
                              # in the summary rather than silently
                              # discarded, per explicit instruction.

OUTPUT_DIR = PROJECT_DIR / "outputs" / "pde_trained_two_gaussian_correction"


def split_centers():
    """G1 <= 0 (left lobe), G2 >= 0 (right lobe), from the EXISTING
    C_VALUES grid -- no new center grid is introduced."""
    c1_values = [c for c in tfgc.C_VALUES if c <= 0]
    c2_values = [c for c in tfgc.C_VALUES if c >= 0]
    return c1_values, c2_values


def precompute_gaussians(c_values, sigma_values, x_eval_full, x_eval_shock):
    entries = []
    for c in c_values:
        for sigma in sigma_values:
            q_full = tfgc.gaussian(x_eval_full, c, sigma)
            q_shock = tfgc.gaussian(x_eval_shock, c, sigma)
            entries.append((float(c), float(sigma), q_full, q_shock))
    return entries


def fit_pair(Q, e):
    """Closed-form least-squares solve for (d1,d2) via dsf.lstsq_solve --
    NOT an optimizer. Returns (d, rank, condition_number, r_squared)."""
    theta, rank, s = dsf.lstsq_solve(Q, e)
    if len(s) == 2 and s[-1] > 0:
        cond = float(s[0] / s[-1])
    elif len(s) == 2 and s[-1] == 0:
        cond = float("inf")
    else:
        cond = float("inf")
    residual = e - Q @ theta
    ss_res = float(np.dot(residual, residual))
    ss_tot = float(np.dot(e, e))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return theta, int(rank), cond, r_squared


def evaluate_two_gaussian_grid(baseline):
    x_eval_full = baseline["x_eval_full"]
    x_eval_shock = baseline["x_eval_shock"]
    y_eval_full = baseline["y_eval_full"]
    y_eval_shock = baseline["y_eval_shock"]
    u_RF_full = baseline["u_RF_full"]
    u_RF_shock = baseline["u_RF_shock"]

    e_full = y_eval_full - u_RF_full
    e_shock = y_eval_shock - u_RF_shock

    c1_values, c2_values = split_centers()
    g1_entries = precompute_gaussians(c1_values, tfgc.SIGMA_VALUES, x_eval_full, x_eval_shock)
    g2_entries = precompute_gaussians(c2_values, tfgc.SIGMA_VALUES, x_eval_full, x_eval_shock)

    rows = []
    for c1, s1, q1_full, q1_shock in g1_entries:
        for c2, s2, q2_full, q2_shock in g2_entries:
            Q_full = np.column_stack([q1_full, q2_full])
            Q_shock = np.column_stack([q1_shock, q2_shock])

            d_global, rank_global, cond_global, r2_global = fit_pair(Q_full, e_full)
            d_shock, rank_shock, cond_shock, r2_shock = fit_pair(Q_shock, e_shock)

            u_oracle_global_full = u_RF_full + Q_full @ d_global
            u_oracle_global_shock = u_RF_shock + Q_shock @ d_global
            m_oracle_global = evaluate_error_metrics(
                x_eval_full, y_eval_full, u_oracle_global_full,
                y_eval_shock, u_oracle_global_shock,
            )

            u_oracle_shock_full = u_RF_full + Q_full @ d_shock
            u_oracle_shock_shock = u_RF_shock + Q_shock @ d_shock
            m_oracle_shock = evaluate_error_metrics(
                x_eval_full, y_eval_full, u_oracle_shock_full,
                y_eval_shock, u_oracle_shock_shock,
            )

            rows.append({
                "c1": c1, "sigma1": s1, "c2": c2, "sigma2": s2,
                "d1_global": float(d_global[0]), "d2_global": float(d_global[1]),
                "rank_global": rank_global, "cond_global": cond_global, "r2_global_fit": r2_global,
                "oracle_global_global": m_oracle_global["global_relative_l2_error"],
                "oracle_global_smooth": m_oracle_global["smooth_relative_l2_error"],
                "oracle_global_shock": m_oracle_global["shock_relative_l2_error"],
                "d1_shock": float(d_shock[0]), "d2_shock": float(d_shock[1]),
                "rank_shock": rank_shock, "cond_shock": cond_shock, "r2_shock_fit": r2_shock,
                "oracle_shock_global": m_oracle_shock["global_relative_l2_error"],
                "oracle_shock_smooth": m_oracle_shock["smooth_relative_l2_error"],
                "oracle_shock_shock": m_oracle_shock["shock_relative_l2_error"],
            })
    return rows


GRID_HEADER_2G = [
    "c1", "sigma1", "c2", "sigma2",
    "d1_global", "d2_global", "rank_global", "cond_global", "r2_global_fit",
    "oracle_global_global", "oracle_global_smooth", "oracle_global_shock",
    "d1_shock", "d2_shock", "rank_shock", "cond_shock", "r2_shock_fit",
    "oracle_shock_global", "oracle_shock_smooth", "oracle_shock_shock",
]


def write_grid_csv(rows):
    """Locally scoped (writes ONLY to this script's OUTPUT_DIR)."""
    path = OUTPUT_DIR / "pde_trained_two_gaussian_correction_grid.csv"
    with path.open("w") as f:
        f.write(",".join(GRID_HEADER_2G) + "\n")
        for row in rows:
            f.write(",".join(str(row[k]) for k in GRID_HEADER_2G) + "\n")


def best_eligible(rows, rank_key, sort_key):
    """Exclude rank-deficient candidates (rank < 2) from eligibility. Does
    NOT apply any condition-number exclusion -- high-condition eligible
    candidates may still win and are flagged (not discarded) by the caller."""
    eligible = [r for r in rows if r[rank_key] == 2]
    pool = eligible if eligible else rows
    return min(pool, key=lambda r: r[sort_key]), (len(eligible) > 0)


def write_summary(baseline_metrics, best_1g_shock, best_1g_global, best_2g_shock, best_2g_global,
                   had_eligible_shock, had_eligible_global):
    lines = []
    lines.append(
        "SEARCH DESIGN: residual-informed LEFT/RIGHT two-Gaussian search "
        "(G1 center <= 0, G2 center >= 0, using the existing C_VALUES/"
        "SIGMA_VALUES grids) -- NOT an unconstrained global search over all "
        "possible Gaussian pairs."
    )
    lines.append("")

    rf_shock = baseline_metrics["shock_relative_l2_error"]
    g1_shock = best_1g_shock["oracle_shock_shock"]
    g2_shock = best_2g_shock["oracle_shock_shock"]

    abs_rf_to_1g = rf_shock - g1_shock
    pct_rf_to_1g = tfgc.pct_improvement(rf_shock, g1_shock)
    abs_1g_to_2g = g1_shock - g2_shock
    pct_1g_to_2g = tfgc.pct_improvement(g1_shock, g2_shock)
    pct_rf_to_2g = tfgc.pct_improvement(rf_shock, g2_shock)

    lines.append("=== Primary comparison: shock error ===")
    lines.append(f"RF-only shock error            = {rf_shock:.6f}")
    lines.append(f"best single-Gaussian shock error = {g1_shock:.6f}")
    lines.append(f"best two-Gaussian shock error     = {g2_shock:.6f}")
    lines.append("")
    lines.append(f"absolute reduction RF-only -> 1G   = {abs_rf_to_1g:.6f}")
    lines.append(f"percentage reduction RF-only -> 1G = {pct_rf_to_1g:+.3f}%")
    lines.append(f"absolute incremental reduction 1G -> 2G   = {abs_1g_to_2g:.6f}")
    lines.append(f"percentage incremental reduction 1G -> 2G = {pct_1g_to_2g:+.3f}%")
    lines.append(f"total percentage reduction RF-only -> 2G  = {pct_rf_to_2g:+.3f}%")
    lines.append("")

    lines.append("=== Winning shock-optimal two-Gaussian case ===")
    if not had_eligible_shock:
        lines.append(
            "WARNING: no rank>=2 (well-conditioned) candidate was found for "
            "the shock-oracle fit; falling back to the full candidate pool "
            "including rank-deficient pairs. Treat this result with caution."
        )
    r = best_2g_shock
    d1, d2 = r["d1_shock"], r["d2_shock"]
    lines.append(f"c1 = {r['c1']:.6f}, sigma1 = {r['sigma1']:.6e}, d1 = {d1:.6e}")
    lines.append(f"c2 = {r['c2']:.6f}, sigma2 = {r['sigma2']:.6e}, d2 = {d2:.6e}")
    lines.append(f"sign(d1) = {'negative' if d1 < 0 else 'positive' if d1 > 0 else 'zero'}")
    lines.append(f"sign(d2) = {'negative' if d2 < 0 else 'positive' if d2 > 0 else 'zero'}")
    lines.append(f"rank = {r['rank_shock']}")
    lines.append(f"condition number = {r['cond_shock']:.6e}")
    lines.append(f"global error = {r['oracle_shock_global']:.6f}")
    lines.append(f"smooth error = {r['oracle_shock_smooth']:.6f}")
    lines.append(f"shock error  = {r['oracle_shock_shock']:.6f}")
    lines.append(f"R^2 on shock residual = {r['r2_shock_fit']:.6f}")
    if r["cond_shock"] > COND_REPORT_THRESHOLD:
        lines.append("")
        lines.append(
            f"WARNING: the winning shock-optimal candidate has a very high "
            f"condition number ({r['cond_shock']:.3e} > {COND_REPORT_THRESHOLD:.1e}). "
            f"d1={d1:.6e} and d2={d2:.6e} may be large, (near-)canceling "
            "coefficients reflecting near-collinear G1/G2 rather than a "
            "genuinely well-separated two-lobe fit. This is reported "
            "explicitly, not silently discarded -- interpret the individual "
            "d1/d2 values with caution even though the resulting corrected "
            "solution/error values above remain numerically valid."
        )
    lines.append("")

    lines.append("=== Winning global-optimal two-Gaussian case ===")
    if not had_eligible_global:
        lines.append(
            "WARNING: no rank>=2 (well-conditioned) candidate was found for "
            "the global-oracle fit; falling back to the full candidate pool "
            "including rank-deficient pairs. Treat this result with caution."
        )
    r = best_2g_global
    d1, d2 = r["d1_global"], r["d2_global"]
    lines.append(f"c1 = {r['c1']:.6f}, sigma1 = {r['sigma1']:.6e}, d1 = {d1:.6e}")
    lines.append(f"c2 = {r['c2']:.6f}, sigma2 = {r['sigma2']:.6e}, d2 = {d2:.6e}")
    lines.append(f"sign(d1) = {'negative' if d1 < 0 else 'positive' if d1 > 0 else 'zero'}")
    lines.append(f"sign(d2) = {'negative' if d2 < 0 else 'positive' if d2 > 0 else 'zero'}")
    lines.append(f"rank = {r['rank_global']}")
    lines.append(f"condition number = {r['cond_global']:.6e}")
    lines.append(f"global error = {r['oracle_global_global']:.6f}")
    lines.append(f"smooth error = {r['oracle_global_smooth']:.6f}")
    lines.append(f"shock error  = {r['oracle_global_shock']:.6f}")
    lines.append(f"R^2 on global residual = {r['r2_global_fit']:.6f}")
    if r["cond_global"] > COND_REPORT_THRESHOLD:
        lines.append("")
        lines.append(
            f"WARNING: the winning global-optimal candidate has a very high "
            f"condition number ({r['cond_global']:.3e} > {COND_REPORT_THRESHOLD:.1e}). "
            f"d1={d1:.6e} and d2={d2:.6e} may be large, (near-)canceling "
            "coefficients reflecting near-collinear G1/G2. Reported "
            "explicitly, not silently discarded."
        )
    lines.append("")

    lines.append("=== Reference: best single-Gaussian results (recomputed fresh) ===")
    lines.append(
        f"shock-optimal: c={best_1g_shock['c']:.6f}, sigma={best_1g_shock['sigma']:.6e}, "
        f"d_star={best_1g_shock['d_star_shock']:.6e}, shock_error={best_1g_shock['oracle_shock_shock']:.6f}"
    )
    lines.append(
        f"global-optimal: c={best_1g_global['c']:.6f}, sigma={best_1g_global['sigma']:.6e}, "
        f"d_star={best_1g_global['d_star_global']:.6e}, global_error={best_1g_global['oracle_global_global']:.6f}"
    )

    (OUTPUT_DIR / "pde_trained_two_gaussian_correction_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def plot_components(baseline, best_2g_shock):
    """d1*G1, d2*G2, their sum, overlaid on e(x); full domain and shock zoom."""
    x_full, x_shock = baseline["x_eval_full"], baseline["x_eval_shock"]
    y_full, y_shock = baseline["y_eval_full"], baseline["y_eval_shock"]
    u_rf_full, u_rf_shock = baseline["u_RF_full"], baseline["u_RF_shock"]
    e_full = y_full - u_rf_full
    e_shock = y_shock - u_rf_shock

    r = best_2g_shock
    c1, s1, d1 = r["c1"], r["sigma1"], r["d1_shock"]
    c2, s2, d2 = r["c2"], r["sigma2"], r["d2_shock"]

    for suffix, x, e in [("full", x_full, e_full), ("shock", x_shock, e_shock)]:
        g1 = d1 * tfgc.gaussian(x, c1, s1)
        g2 = d2 * tfgc.gaussian(x, c2, s2)
        total = g1 + g2
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(x, e, "k-", linewidth=2, label="e(x) = u_true - u_RF")
        ax.plot(x, g1, "--", label=f"d1*G1 (c1={c1:.4g}, sigma1={s1:.4g})")
        ax.plot(x, g2, "--", label=f"d2*G2 (c2={c2:.4g}, sigma2={s2:.4g})")
        ax.plot(x, total, "-", label="d1*G1 + d2*G2")
        ax.set_title(f"Two-Gaussian components vs residual, {suffix} domain")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"components_{suffix}.png")
        plt.close(fig)


def plot_residual(baseline, best_2g_shock):
    """Residual e(x) alone, and corrected residual e(x) - (d1*G1+d2*G2)."""
    x_full, x_shock = baseline["x_eval_full"], baseline["x_eval_shock"]
    y_full, y_shock = baseline["y_eval_full"], baseline["y_eval_shock"]
    u_rf_full, u_rf_shock = baseline["u_RF_full"], baseline["u_RF_shock"]
    e_full = y_full - u_rf_full
    e_shock = y_shock - u_rf_shock

    r = best_2g_shock
    c1, s1, d1 = r["c1"], r["sigma1"], r["d1_shock"]
    c2, s2, d2 = r["c2"], r["sigma2"], r["d2_shock"]

    for suffix, x, e in [("full", x_full, e_full), ("shock", x_shock, e_shock)]:
        correction = d1 * tfgc.gaussian(x, c1, s1) + d2 * tfgc.gaussian(x, c2, s2)
        corrected_residual = e - correction

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(x, e, "k-", linewidth=2, label="e(x) (uncorrected)")
        ax.axhline(0.0, color="gray", linewidth=0.8)
        ax.set_title(f"Residual e(x), {suffix} domain")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"residual_{suffix}.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(x, corrected_residual, "r-", linewidth=2, label="e(x) - (d1*G1 + d2*G2)")
        ax.axhline(0.0, color="gray", linewidth=0.8)
        ax.set_title(f"Corrected residual after two-Gaussian fit, {suffix} domain")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"corrected_residual_{suffix}.png")
        plt.close(fig)


def plot_solution(baseline, best_2g_shock):
    """u_true vs u_RF vs best two-Gaussian corrected; full domain and shock zoom."""
    x_full, x_shock = baseline["x_eval_full"], baseline["x_eval_shock"]
    y_full, y_shock = baseline["y_eval_full"], baseline["y_eval_shock"]
    u_rf_full, u_rf_shock = baseline["u_RF_full"], baseline["u_RF_shock"]

    r = best_2g_shock
    c1, s1, d1 = r["c1"], r["sigma1"], r["d1_shock"]
    c2, s2, d2 = r["c2"], r["sigma2"], r["d2_shock"]

    for suffix, x, y, u_rf in [("full", x_full, y_full, u_rf_full), ("shock", x_shock, y_shock, u_rf_shock)]:
        correction = d1 * tfgc.gaussian(x, c1, s1) + d2 * tfgc.gaussian(x, c2, s2)
        u_corrected = u_rf + correction

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(x, y, "k-", linewidth=2, label="u_true")
        ax.plot(x, u_rf, "--", label="u_RF (baseline)")
        ax.plot(x, u_corrected, "-", label="u_RF + (d1*G1 + d2*G2)")
        ax.set_ylim(tfgc.solution_ylim(y, u_rf, u_corrected))
        ax.set_title(f"Best two-Gaussian corrected solution, {suffix} domain")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"solution_{suffix}.png")
        plt.close(fig)


def plot_three_way_shock_comparison(baseline, best_1g_shock, best_2g_shock):
    """RF-only vs best single-Gaussian corrected vs best two-Gaussian
    corrected, shock-region zoom."""
    x_shock = baseline["x_eval_shock"]
    y_shock = baseline["y_eval_shock"]
    u_rf_shock = baseline["u_RF_shock"]

    r1 = best_1g_shock
    u_1g = u_rf_shock + r1["d_star_shock"] * tfgc.gaussian(x_shock, r1["c"], r1["sigma"])

    r2 = best_2g_shock
    correction_2g = (
        r2["d1_shock"] * tfgc.gaussian(x_shock, r2["c1"], r2["sigma1"])
        + r2["d2_shock"] * tfgc.gaussian(x_shock, r2["c2"], r2["sigma2"])
    )
    u_2g = u_rf_shock + correction_2g

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(x_shock, y_shock, "k-", linewidth=2, label="u_true")
    ax.plot(x_shock, u_rf_shock, "--", label="u_RF (baseline)")
    ax.plot(x_shock, u_1g, ":", label="u_RF + best single Gaussian")
    ax.plot(x_shock, u_2g, "-", label="u_RF + best two Gaussians")
    ax.set_ylim(tfgc.solution_ylim(y_shock, u_rf_shock, u_1g, u_2g))
    ax.set_title("Shock-region comparison: RF-only vs best 1G vs best 2G")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "three_way_shock_comparison.png")
    plt.close(fig)


def main():
    torch.set_default_dtype(dsf.DTYPE)
    device = "cpu"

    model, checkpoint = ptgc.load_pde_trained_model(device)
    ptgc.verify_baseline(model, checkpoint, device)

    baseline = ptgc.build_search_baseline(model, device)
    baseline_metrics = evaluate_error_metrics(
        baseline["x_eval_full"], baseline["y_eval_full"], baseline["u_RF_full"],
        baseline["y_eval_shock"], baseline["u_RF_shock"],
    )
    print(f"RF-only baseline metrics (t=1): {baseline_metrics}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Computing single-Gaussian reference grid (reused unchanged)...")
    rows_1g = tfgc.evaluate_grid(baseline)
    best_1g_shock = tfgc.best_row(rows_1g, "oracle_shock_shock")
    best_1g_global = tfgc.best_row(rows_1g, "oracle_global_global")

    print("Computing residual-informed left/right two-Gaussian grid...")
    rows_2g = evaluate_two_gaussian_grid(baseline)
    write_grid_csv(rows_2g)

    best_2g_shock, had_eligible_shock = best_eligible(rows_2g, "rank_shock", "oracle_shock_shock")
    best_2g_global, had_eligible_global = best_eligible(rows_2g, "rank_global", "oracle_global_global")

    write_summary(
        baseline_metrics, best_1g_shock, best_1g_global, best_2g_shock, best_2g_global,
        had_eligible_shock, had_eligible_global,
    )

    plot_solution(baseline, best_2g_shock)
    plot_residual(baseline, best_2g_shock)
    plot_components(baseline, best_2g_shock)
    plot_three_way_shock_comparison(baseline, best_1g_shock, best_2g_shock)

    print("[test_pde_trained_two_gaussian_correction] done.")


if __name__ == "__main__":
    main()
