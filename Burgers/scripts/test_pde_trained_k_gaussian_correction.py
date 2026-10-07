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
# gaussian, C_VALUES, SIGMA_VALUES
import test_direct_spatial_fit as dsf  # noqa: E402 -- reused unchanged: lstsq_solve, DTYPE, T_SNAPSHOT, reference_values
from sweep_local_width import SHOCK_X_HALF_WIDTH, SHOCK_N_X  # noqa: E402

# Post-hoc K-GAUSSIAN REPRESENTATION-CAPACITY study. NOT PDE training: the
# canonical PDE-trained RF checkpoint is loaded and kept completely fixed
# (reusing ptgc.load_pde_trained_model/verify_baseline/build_search_baseline
# unmodified); amplitudes are solved by closed-form least squares
# (dsf.lstsq_solve) at every step, never by Adam or any optimizer.
# Evaluated at t=1 only.
#
# ALGORITHM: greedy forward basis selection with joint coefficient
# refitting. At each step K, every unused (center,sigma) candidate from the
# existing 525-point library (tfgc.C_VALUES x tfgc.SIGMA_VALUES, no
# left/right split -- greedy is free to choose from the whole library) is
# tried as an addition to the currently selected K-1 Gaussians; ALL K
# amplitudes are refit jointly (never just the newest one) via SVD-based
# least squares against the canonical 200-point shock residual; the
# candidate minimizing the resulting shock-residual norm is kept. This is
# NOT claimed to be the globally optimal K-Gaussian nonlinear solution --
# it is a greedy, structurally duplicate-free (selection is over indices
# not yet chosen) approximation to it.
#
# DENSE-GRID VALIDATION: an independent 2001-point shock grid (same
# |x|<=0.02 region, t=1) is used ONLY to evaluate each K's already-fixed
# correction -- it NEVER participates in selection or coefficient fitting.
# Its purpose is to distinguish genuine representational improvement from
# discrete 200-point fitting-grid overfitting (the fitting grid spacing is
# comparable to, or coarser than, the narrowest library sigma values).
#
# GREEDY K=2 vs PAIR-SEARCH K=2: these are DIFFERENT searches and are
# reported SEPARATELY, never conflated. Greedy K=2 is constrained to retain
# the greedy K=1 winner; the pair-search K=2 (test_pde_trained_two_gaussian_
# correction.py) jointly searched both Gaussian geometries without that
# constraint and remains the stronger controlled K=2 reference. Its
# confirmed geometry (c1,sigma1,c2,sigma2) is reused here as exact matching
# library entries; amplitudes and result metrics are NOT hardcoded -- only
# the geometry is reused, and d1,d2/rank/cond/metrics are refit fresh on
# the same grids used throughout this script.

K_MAX = 8
DENSE_SHOCK_N_X = 2001

# Pair-search reference geometry (confirmed from the server
# test_pde_trained_two_gaussian_correction.py run). NOT the amplitudes or
# result metrics -- those are refit/recomputed fresh below.
PAIR_SEARCH_C1 = -0.010000
PAIR_SEARCH_SIGMA1 = 1.067384e-02
PAIR_SEARCH_C2 = 0.010000
PAIR_SEARCH_SIGMA2 = 1.490050e-02
PAIR_SEARCH_MATCH_RTOL = 1e-4

# Heuristic-only flags (descriptive, not a scientific conclusion by
# themselves): if the fitting-grid shock error improves by at least this
# percentage from K-1 while the independent dense-grid shock error improves
# by less than the (much smaller) dense threshold, this K is flagged in the
# summary as a candidate for fitting-grid overfitting rather than genuine
# added representation capacity.
OVERFIT_FIT_IMPROVEMENT_MIN_PCT = 5.0
OVERFIT_DENSE_IMPROVEMENT_MAX_PCT = 1.0

OUTPUT_DIR = PROJECT_DIR / "outputs" / "pde_trained_k_gaussian_correction"


def build_dense_shock_baseline(model, device):
    """Independent, denser shock grid (2001 points on |x|<=0.02, t=1 only).
    Used ONLY for post-hoc validation of each K's already-fixed correction
    -- NEVER for greedy selection or coefficient fitting. Pure forward
    evaluation of the already-loaded, fixed model; no training, no
    optimizer, no parameter modification."""
    x_dense_np = np.linspace(-SHOCK_X_HALF_WIDTH, SHOCK_X_HALF_WIDTH, DENSE_SHOCK_N_X)
    y_dense = dsf.reference_values(x_dense_np, dsf.T_SNAPSHOT)

    x_dense_t = torch.tensor(x_dense_np.reshape(-1, 1), dtype=dsf.DTYPE)
    t_dense_t = torch.full_like(x_dense_t, dsf.T_SNAPSHOT)
    with torch.no_grad():
        _, u_rf_dense_t, _ = model.forward_components(x_dense_t, t_dense_t)
    u_rf_dense = u_rf_dense_t.detach().cpu().numpy().reshape(-1)

    return {
        "x_dense_shock": x_dense_np,
        "y_dense_shock": y_dense,
        "u_RF_dense_shock": u_rf_dense,
    }


def find_exact_library_value(target, library_values, rtol, label):
    """Locate the nearest entry in an EXISTING library array and verify it
    matches the target within rtol -- never silently substitutes an
    off-grid value, and never introduces a new candidate outside the
    existing C_VALUES/SIGMA_VALUES grids."""
    arr = np.asarray(library_values)
    idx = int(np.argmin(np.abs(arr - target)))
    val = float(arr[idx])
    rel_diff = abs(val - target) / abs(target) if target != 0 else abs(val)
    if rel_diff > rtol:
        raise RuntimeError(
            f"{label}: nearest library value {val} does not match target {target} "
            f"within rtol={rtol} (actual rel diff {rel_diff:.3e}). Refusing to proceed "
            "with an unverified pair-search geometry reference."
        )
    return val


def build_library(x_eval_full, x_eval_shock, x_dense_shock):
    """Full 525-candidate library (tfgc.C_VALUES x tfgc.SIGMA_VALUES, no
    left/right split) -- greedy forward selection is free to pick from the
    whole library at every step."""
    entries = []
    for c in tfgc.C_VALUES:
        for sigma in tfgc.SIGMA_VALUES:
            entries.append({
                "c": float(c), "sigma": float(sigma),
                "q_full": tfgc.gaussian(x_eval_full, c, sigma),
                "q_shock": tfgc.gaussian(x_eval_shock, c, sigma),
                "q_dense": tfgc.gaussian(x_dense_shock, c, sigma),
            })
    return entries


def find_library_index(library, c_target, sigma_target, rtol):
    for i, entry in enumerate(library):
        if abs(entry["c"] - c_target) / (abs(c_target) if c_target != 0 else 1.0) <= rtol and \
           abs(entry["sigma"] - sigma_target) / abs(sigma_target) <= rtol:
            return i
    raise RuntimeError(f"No library entry found matching c={c_target}, sigma={sigma_target} within rtol={rtol}.")


def stack_columns(indices, library, key):
    return np.column_stack([library[i][key] for i in indices])


def fit_and_score(selected_indices, library, e_shock):
    """Closed-form least-squares solve for ALL amplitudes jointly, via
    dsf.lstsq_solve -- NOT an optimizer, and NEVER holds any previously
    selected amplitude fixed."""
    Q = stack_columns(selected_indices, library, "q_shock")
    d, rank, s = dsf.lstsq_solve(Q, e_shock)
    residual = e_shock - Q @ d
    resid_norm = float(np.linalg.norm(residual))
    return d, int(rank), s, resid_norm


def greedy_select(library, e_shock, k_max):
    """Greedy forward basis selection with joint coefficient refitting.
    Duplicate (center,sigma) selection is structurally impossible: each
    step only considers indices not already in `selected`."""
    if k_max > len(library):
        raise ValueError(
            f"k_max={k_max} exceeds the candidate library size ({len(library)}); "
            "cannot select more distinct Gaussians than exist in the library."
        )
    selected = []
    remaining = set(range(len(library)))
    history = []
    for k in range(1, k_max + 1):
        best = None
        for j in remaining:
            candidate = selected + [j]
            d, rank, s, resid_norm = fit_and_score(candidate, library, e_shock)
            if best is None or resid_norm < best["resid_norm"]:
                best = {"j": j, "d": d, "rank": rank, "s": s, "resid_norm": resid_norm}
        selected.append(best["j"])
        remaining.discard(best["j"])
        cond = float(best["s"][0] / best["s"][-1]) if best["s"][-1] > 0 else float("inf")
        history.append({
            "K": k,
            "selected_indices": list(selected),
            "d": best["d"],
            "rank": best["rank"],
            "cond": cond,
            "newly_added_index": best["j"],
        })
    return history


def evaluate_selection(indices, d, library, baseline, dense_baseline):
    """Evaluate a FIXED (already-selected, already-fitted) correction on:
    (a) the canonical 200-pt fitting shock grid + full domain (global/
    smooth/shock via evaluate_error_metrics, reused unchanged);
    (b) the independent 2001-pt dense shock grid (shock-only, same
    evaluate_error_metrics formula, dense grid substituted for the shock
    arguments)."""
    x_eval_full = baseline["x_eval_full"]
    y_eval_full = baseline["y_eval_full"]
    u_RF_full = baseline["u_RF_full"]
    x_eval_shock = baseline["x_eval_shock"]
    y_eval_shock = baseline["y_eval_shock"]
    u_RF_shock = baseline["u_RF_shock"]

    if len(indices) == 0:
        correction_full = np.zeros_like(u_RF_full)
        correction_shock = np.zeros_like(u_RF_shock)
        correction_dense = np.zeros_like(dense_baseline["u_RF_dense_shock"])
    else:
        Q_full = stack_columns(indices, library, "q_full")
        Q_shock = stack_columns(indices, library, "q_shock")
        Q_dense = stack_columns(indices, library, "q_dense")
        correction_full = Q_full @ d
        correction_shock = Q_shock @ d
        correction_dense = Q_dense @ d

    u_pred_full = u_RF_full + correction_full
    u_pred_shock = u_RF_shock + correction_shock
    u_pred_dense = dense_baseline["u_RF_dense_shock"] + correction_dense

    metrics_fit = evaluate_error_metrics(x_eval_full, y_eval_full, u_pred_full, y_eval_shock, u_pred_shock)
    metrics_dense = evaluate_error_metrics(
        x_eval_full, y_eval_full, u_pred_full,
        dense_baseline["y_dense_shock"], u_pred_dense,
    )

    e_shock = y_eval_shock - u_RF_shock
    ss_tot_fit = float(np.dot(e_shock, e_shock))
    if len(indices) == 0 or ss_tot_fit == 0:
        r2_fit = float("nan")
    else:
        ss_res_fit = float(np.dot(e_shock - correction_shock, e_shock - correction_shock))
        r2_fit = 1.0 - ss_res_fit / ss_tot_fit

    e_dense = dense_baseline["y_dense_shock"] - dense_baseline["u_RF_dense_shock"]
    ss_tot_dense = float(np.dot(e_dense, e_dense))
    if len(indices) == 0 or ss_tot_dense == 0:
        r2_dense = float("nan")
    else:
        ss_res_dense = float(np.dot(e_dense - correction_dense, e_dense - correction_dense))
        r2_dense = 1.0 - ss_res_dense / ss_tot_dense

    return {
        "global_error": metrics_fit["global_relative_l2_error"],
        "smooth_error": metrics_fit["smooth_relative_l2_error"],
        "shock_error_fit": metrics_fit["shock_relative_l2_error"],
        "shock_error_dense": metrics_dense["shock_relative_l2_error"],
        "r2_fit": r2_fit,
        "r2_dense": r2_dense,
        "correction_full": correction_full,
        "correction_shock": correction_shock,
        "correction_dense": correction_dense,
    }


def pct_improvement(baseline_val, new_val):
    return 100 * (baseline_val - new_val) / baseline_val if baseline_val != 0 else float("nan")


def build_k_rows(history, library, baseline, dense_baseline, baseline_metrics, delta_x_fit):
    rows = []
    rf_only_eval = evaluate_selection([], None, library, baseline, dense_baseline)
    rows.append({
        "K": 0, "n_selected": 0,
        "centers": "", "sigmas": "", "amplitudes": "",
        "newly_added_c": "", "newly_added_sigma": "",
        "rank": "n/a", "cond": "n/a",
        "r2_fit": "n/a", "r2_dense": "n/a",
        "global_error": baseline_metrics["global_relative_l2_error"],
        "smooth_error": baseline_metrics["smooth_relative_l2_error"],
        "shock_error_fit": baseline_metrics["shock_relative_l2_error"],
        "shock_error_dense": rf_only_eval["shock_error_dense"],
        "subresolution_flag": "",
    })

    prev_shock_fit = baseline_metrics["shock_relative_l2_error"]
    prev_shock_dense = rows[0]["shock_error_dense"]
    rf_shock_fit = prev_shock_fit
    rf_shock_dense = prev_shock_dense

    for step in history:
        k = step["K"]
        indices = step["selected_indices"]
        d = step["d"]
        ev = evaluate_selection(indices, d, library, baseline, dense_baseline)

        newly_added_c = library[step["newly_added_index"]]["c"]
        newly_added_sigma = library[step["newly_added_index"]]["sigma"]
        subres = newly_added_sigma < 2 * delta_x_fit

        abs_from_prev_fit = prev_shock_fit - ev["shock_error_fit"]
        pct_from_prev_fit = pct_improvement(prev_shock_fit, ev["shock_error_fit"])
        abs_from_prev_dense = prev_shock_dense - ev["shock_error_dense"]
        pct_from_prev_dense = pct_improvement(prev_shock_dense, ev["shock_error_dense"])

        overfit_flag = (
            pct_from_prev_fit >= OVERFIT_FIT_IMPROVEMENT_MIN_PCT
            and pct_from_prev_dense < OVERFIT_DENSE_IMPROVEMENT_MAX_PCT
        )

        rows.append({
            "K": k, "n_selected": len(indices),
            "centers": ";".join(f"{library[i]['c']:.6f}" for i in indices),
            "sigmas": ";".join(f"{library[i]['sigma']:.6e}" for i in indices),
            "amplitudes": ";".join(f"{v:.6e}" for v in d),
            "newly_added_c": f"{newly_added_c:.6f}",
            "newly_added_sigma": f"{newly_added_sigma:.6e}",
            "rank": step["rank"],
            "cond": step["cond"],
            "r2_fit": ev["r2_fit"],
            "r2_dense": ev["r2_dense"],
            "global_error": ev["global_error"],
            "smooth_error": ev["smooth_error"],
            "shock_error_fit": ev["shock_error_fit"],
            "shock_error_dense": ev["shock_error_dense"],
            "shock_abs_improvement_from_prev_fit": abs_from_prev_fit,
            "shock_pct_improvement_from_prev_fit": pct_from_prev_fit,
            "shock_abs_improvement_from_prev_dense": abs_from_prev_dense,
            "shock_pct_improvement_from_prev_dense": pct_from_prev_dense,
            "shock_abs_improvement_from_rf_fit": rf_shock_fit - ev["shock_error_fit"],
            "shock_pct_improvement_from_rf_fit": pct_improvement(rf_shock_fit, ev["shock_error_fit"]),
            "shock_abs_improvement_from_rf_dense": rf_shock_dense - ev["shock_error_dense"],
            "shock_pct_improvement_from_rf_dense": pct_improvement(rf_shock_dense, ev["shock_error_dense"]),
            "subresolution_flag": "FLAG" if subres else "",
            "possible_overfit_flag": "FLAG" if overfit_flag else "",
            "_correction_dense": ev["correction_dense"],
            "_correction_shock": ev["correction_shock"],
        })
        prev_shock_fit = ev["shock_error_fit"]
        prev_shock_dense = ev["shock_error_dense"]

    return rows


CSV_HEADER = [
    "K", "n_selected", "centers", "sigmas", "amplitudes",
    "newly_added_c", "newly_added_sigma",
    "rank", "cond", "r2_fit", "r2_dense",
    "global_error", "smooth_error", "shock_error_fit", "shock_error_dense",
    "shock_abs_improvement_from_prev_fit", "shock_pct_improvement_from_prev_fit",
    "shock_abs_improvement_from_prev_dense", "shock_pct_improvement_from_prev_dense",
    "shock_abs_improvement_from_rf_fit", "shock_pct_improvement_from_rf_fit",
    "shock_abs_improvement_from_rf_dense", "shock_pct_improvement_from_rf_dense",
    "subresolution_flag", "possible_overfit_flag",
]


def write_csv(rows):
    path = OUTPUT_DIR / "pde_trained_k_gaussian_greedy.csv"
    with path.open("w") as f:
        f.write(",".join(CSV_HEADER) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(k, "")) for k in CSV_HEADER) + "\n")


def write_summary(rows, delta_x_fit, pair_row):
    lines = []
    lines.append(
        "GREEDY forward Gaussian basis selection with joint coefficient "
        "refitting (NOT claimed globally optimal). Selection uses ONLY the "
        f"canonical {SHOCK_N_X}-point fitting shock grid (|x|<={SHOCK_X_HALF_WIDTH}, "
        f"spacing Delta_x_fit={delta_x_fit:.6e}). An INDEPENDENT "
        f"{DENSE_SHOCK_N_X}-point dense shock grid is used ONLY to validate "
        "each K's already-fixed correction, never for selection/fitting."
    )
    lines.append("")

    for row in rows:
        k = row["K"]
        lines.append(f"--- K = {k} ---")
        if k == 0:
            lines.append("RF-only (no correction).")
        else:
            lines.append(f"centers  = {row['centers']}")
            lines.append(f"sigmas   = {row['sigmas']}")
            lines.append(f"amplitudes = {row['amplitudes']}")
            lines.append(f"newly added this step: c={row['newly_added_c']}, sigma={row['newly_added_sigma']}")
            lines.append(f"rank = {row['rank']}")
            lines.append(f"condition number = {row['cond']}")
            if k == 1:
                lines.append("  (K=1 condition number is trivially 1.0: single-column fit.)")
            lines.append(f"R^2 (fitting grid, {SHOCK_N_X}pt) = {row['r2_fit']}")
            lines.append(f"R^2 (dense grid, {DENSE_SHOCK_N_X}pt, independent) = {row['r2_dense']}")
        lines.append(f"global_error = {row['global_error']}")
        lines.append(f"smooth_error = {row['smooth_error']}")
        lines.append(f"shock_error (fitting grid) = {row['shock_error_fit']}")
        lines.append(f"shock_error (dense grid, independent) = {row['shock_error_dense']}")
        if k > 0:
            lines.append(
                f"shock improvement from K-1: fit={row['shock_pct_improvement_from_prev_fit']:+.3f}% "
                f"(abs {row['shock_abs_improvement_from_prev_fit']:.6f}), "
                f"dense={row['shock_pct_improvement_from_prev_dense']:+.3f}% "
                f"(abs {row['shock_abs_improvement_from_prev_dense']:.6f})"
            )
            lines.append(
                f"total shock improvement from RF-only: fit={row['shock_pct_improvement_from_rf_fit']:+.3f}%, "
                f"dense={row['shock_pct_improvement_from_rf_dense']:+.3f}%"
            )
            if row["subresolution_flag"]:
                lines.append(
                    f"WARNING: newly added sigma ({row['newly_added_sigma']}) < 2*Delta_x_fit "
                    f"({2*delta_x_fit:.6e}) -- this Gaussian is at or below the fitting grid's "
                    "spatial resolution; its apparent benefit on the 200-point fitting grid may "
                    "not reflect genuine representational capacity. Compare against the "
                    "dense-grid numbers above before trusting this step."
                )
            if row["possible_overfit_flag"]:
                lines.append(
                    f"WARNING (heuristic, descriptive only): fitting-grid shock improvement "
                    f"({row['shock_pct_improvement_from_prev_fit']:+.3f}%) exceeds "
                    f"{OVERFIT_FIT_IMPROVEMENT_MIN_PCT}% while the independent dense-grid shock "
                    f"improvement ({row['shock_pct_improvement_from_prev_dense']:+.3f}%) is below "
                    f"{OVERFIT_DENSE_IMPROVEMENT_MAX_PCT}%. This pattern is flagged as a candidate "
                    "for fitting-grid overfitting / saturation rather than genuine added "
                    "representation capacity -- NOT an automatic conclusion that this K is useless."
                )
        lines.append("")

    lines.append("=" * 70)
    lines.append("PAIR-SEARCH K=2 REFERENCE (test_pde_trained_two_gaussian_correction.py)")
    lines.append(
        "This is a SEPARATE, stronger controlled K=2 search: both Gaussian "
        "geometries were jointly searched without being constrained to "
        "retain any single-Gaussian winner. Geometry reused as exact "
        "existing library entries; amplitudes/metrics refit/recomputed "
        "fresh here (not hardcoded)."
    )
    lines.append(f"c1 = {pair_row['c1']:.6f}, sigma1 = {pair_row['sigma1']:.6e}, d1 = {pair_row['d1']:.6e}")
    lines.append(f"c2 = {pair_row['c2']:.6f}, sigma2 = {pair_row['sigma2']:.6e}, d2 = {pair_row['d2']:.6e}")
    lines.append(f"rank = {pair_row['rank']}, condition number = {pair_row['cond']}")
    lines.append(f"R^2 (fitting grid) = {pair_row['r2_fit']}, R^2 (dense grid) = {pair_row['r2_dense']}")
    lines.append(f"global_error = {pair_row['global_error']}")
    lines.append(f"smooth_error = {pair_row['smooth_error']}")
    lines.append(f"shock_error (fitting grid) = {pair_row['shock_error_fit']}")
    lines.append(f"shock_error (dense grid) = {pair_row['shock_error_dense']}")
    lines.append("")

    greedy_k2 = next(r for r in rows if r["K"] == 2)
    lines.append("--- Greedy K=2 vs pair-search K=2 (DO NOT CONFLATE) ---")
    lines.append(
        f"greedy K=2:      shock_error (fit) = {greedy_k2['shock_error_fit']}, "
        f"shock_error (dense) = {greedy_k2['shock_error_dense']}"
    )
    lines.append(
        f"pair-search K=2: shock_error (fit) = {pair_row['shock_error_fit']}, "
        f"shock_error (dense) = {pair_row['shock_error_dense']}"
    )
    lines.append(
        "Greedy K=2 is constrained to retain the greedy K=1 winner, so it is "
        "expected to be no better than (and typically worse than) the "
        "unconstrained pair-search K=2. The pair-search K=2 remains the "
        "stronger controlled K=2 reference; greedy K=2 is reported for "
        "consistency of the greedy K-curve only, not as a replacement."
    )
    lines.append("")
    lines.append(
        "NOTE ON INTERPRETATION: the 'best' K is NOT simply the K with the "
        "smallest fitting-grid shock error. Judge the SMALLEST USEFUL K using "
        "the dense-grid shock error/R^2, the incremental improvement from "
        "K-1 (both grids), the condition number, the selected widths, and "
        "the global/smooth error above -- see the possible-overfit warnings "
        "per K."
    )

    (OUTPUT_DIR / "pde_trained_k_gaussian_correction_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def plot_shock_error_vs_k(rows, pair_row):
    k_values = [r["K"] for r in rows]
    fit_errors = [r["shock_error_fit"] for r in rows]
    dense_errors = [r["shock_error_dense"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, fit_errors, "o-", label=f"greedy, fitting-grid ({SHOCK_N_X}pt)")
    ax.plot(k_values, dense_errors, "s--", label=f"greedy, dense-grid ({DENSE_SHOCK_N_X}pt, independent)")
    ax.scatter([2], [pair_row["shock_error_fit"]], marker="*", s=180, color="tab:red",
               label="pair-search K=2 (fitting-grid)", zorder=5)
    ax.scatter([2], [pair_row["shock_error_dense"]], marker="*", s=180, color="tab:orange",
               label="pair-search K=2 (dense-grid)", zorder=5)
    ax.set_xlabel("K")
    ax.set_ylabel("shock relative L2 error")
    ax.set_title("Shock error vs K (greedy forward selection)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "shock_error_vs_K.png")
    plt.close(fig)


def plot_global_error_vs_k(rows):
    k_values = [r["K"] for r in rows]
    global_errors = [r["global_error"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, global_errors, "o-", color="tab:green")
    ax.set_xlabel("K")
    ax.set_ylabel("global relative L2 error")
    ax.set_title("Global error vs K (greedy forward selection)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "global_error_vs_K.png")
    plt.close(fig)


def plot_r2_vs_k(rows, pair_row):
    k_values = [r["K"] for r in rows if r["K"] >= 1]
    r2_fit = [r["r2_fit"] for r in rows if r["K"] >= 1]
    r2_dense = [r["r2_dense"] for r in rows if r["K"] >= 1]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, r2_fit, "o-", label=f"greedy, fitting-grid ({SHOCK_N_X}pt)")
    ax.plot(k_values, r2_dense, "s--", label=f"greedy, dense-grid ({DENSE_SHOCK_N_X}pt, independent)")
    ax.scatter([2], [pair_row["r2_fit"]], marker="*", s=180, color="tab:red",
               label="pair-search K=2 (fitting-grid)", zorder=5)
    ax.scatter([2], [pair_row["r2_dense"]], marker="*", s=180, color="tab:orange",
               label="pair-search K=2 (dense-grid)", zorder=5)
    ax.set_xlabel("K")
    ax.set_ylabel("R^2 on shock residual")
    ax.set_title("Shock residual R^2 vs K (greedy forward selection)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "r2_vs_K.png")
    plt.close(fig)


def plot_condition_number_vs_k(rows):
    k_values = [r["K"] for r in rows if r["K"] >= 1]
    conds = [r["cond"] for r in rows if r["K"] >= 1]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, conds, "o-", color="tab:purple")
    ax.set_yscale("log")
    ax.set_xlabel("K")
    ax.set_ylabel("condition number (log scale)")
    ax.set_title("Condition number vs K (greedy forward selection)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "condition_number_vs_K.png")
    plt.close(fig)


def plot_corrected_solutions_vs_k(rows, baseline, dense_baseline):
    x_dense = dense_baseline["x_dense_shock"]
    y_dense = dense_baseline["y_dense_shock"]
    u_rf_dense = dense_baseline["u_RF_dense_shock"]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(x_dense, y_dense, "k-", linewidth=2.5, label="u_true")
    ax.plot(x_dense, u_rf_dense, "--", linewidth=1.5, label="u_RF (K=0)")
    for row in rows:
        if row["K"] == 0:
            continue
        u_corrected = u_rf_dense + row["_correction_dense"]
        ax.plot(x_dense, u_corrected, label=f"K={row['K']}")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t=1)")
    ax.set_title("Corrected shock solution vs K (evaluated on independent dense grid)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "corrected_solutions_vs_K.png")
    plt.close(fig)


def plot_greedy_vs_pairsearch(rows, pair_row, dense_baseline):
    x_dense = dense_baseline["x_dense_shock"]
    y_dense = dense_baseline["y_dense_shock"]
    u_rf_dense = dense_baseline["u_RF_dense_shock"]

    greedy_k2 = next(r for r in rows if r["K"] == 2)
    u_greedy_k2 = u_rf_dense + greedy_k2["_correction_dense"]
    u_pair_k2 = u_rf_dense + pair_row["correction_dense"]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(x_dense, y_dense, "k-", linewidth=2.5, label="u_true")
    ax.plot(x_dense, u_rf_dense, "--", label="u_RF (K=0)")
    ax.plot(x_dense, u_greedy_k2, ":", label="greedy K=2")
    ax.plot(x_dense, u_pair_k2, "-", label="pair-search K=2")
    ax.set_title("Greedy K=2 vs pair-search K=2 (independent dense shock grid)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "greedy_vs_pairsearch_K2.png")
    plt.close(fig)


def plot_final_components(rows, library, baseline, dense_baseline):
    final_row = rows[-1]
    if final_row["K"] == 0:
        return

    x_dense = dense_baseline["x_dense_shock"]
    e_dense = dense_baseline["y_dense_shock"] - dense_baseline["u_RF_dense_shock"]

    centers = [float(c) for c in final_row["centers"].split(";")]
    sigmas = [float(s) for s in final_row["sigmas"].split(";")]
    amplitudes = [float(a) for a in final_row["amplitudes"].split(";")]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(x_dense, e_dense, "k-", linewidth=2.5, label="e(x) = u_true - u_RF")
    total = np.zeros_like(x_dense)
    for i, (c, sigma, d) in enumerate(zip(centers, sigmas, amplitudes)):
        component = d * tfgc.gaussian(x_dense, c, sigma)
        total += component
        ax.plot(x_dense, component, "--", linewidth=1, label=f"d{i+1}*G{i+1} (c={c:.4g}, sigma={sigma:.4g})")
    ax.plot(x_dense, total, "-", linewidth=2, label=f"sum of K={final_row['K']} components")
    ax.set_title(f"Final K={final_row['K']} components vs residual (dense grid)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "final_K_components_vs_residual.png")
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

    dense_baseline = build_dense_shock_baseline(model, device)

    x_eval_shock = baseline["x_eval_shock"]
    delta_x_fit = (x_eval_shock.max() - x_eval_shock.min()) / (len(np.unique(x_eval_shock)) - 1)
    print(f"Delta_x_fit (canonical {SHOCK_N_X}-point shock grid spacing) = {delta_x_fit:.6e}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Building {len(tfgc.C_VALUES)}x{len(tfgc.SIGMA_VALUES)} candidate library...")
    library = build_library(baseline["x_eval_full"], baseline["x_eval_shock"], dense_baseline["x_dense_shock"])

    e_shock = baseline["y_eval_shock"] - baseline["u_RF_shock"]

    print(f"Running greedy forward selection, K_MAX={K_MAX}...")
    history = greedy_select(library, e_shock, K_MAX)

    rows = build_k_rows(history, library, baseline, dense_baseline, baseline_metrics, delta_x_fit)
    write_csv(rows)

    print("Computing pair-search K=2 reference (geometry reused, amplitudes/metrics refit fresh)...")
    c1 = find_exact_library_value(PAIR_SEARCH_C1, tfgc.C_VALUES, PAIR_SEARCH_MATCH_RTOL, "pair-search c1")
    sigma1 = find_exact_library_value(PAIR_SEARCH_SIGMA1, tfgc.SIGMA_VALUES, PAIR_SEARCH_MATCH_RTOL, "pair-search sigma1")
    c2 = find_exact_library_value(PAIR_SEARCH_C2, tfgc.C_VALUES, PAIR_SEARCH_MATCH_RTOL, "pair-search c2")
    sigma2 = find_exact_library_value(PAIR_SEARCH_SIGMA2, tfgc.SIGMA_VALUES, PAIR_SEARCH_MATCH_RTOL, "pair-search sigma2")
    idx1 = find_library_index(library, c1, sigma1, PAIR_SEARCH_MATCH_RTOL)
    idx2 = find_library_index(library, c2, sigma2, PAIR_SEARCH_MATCH_RTOL)
    pair_indices = [idx1, idx2]
    d_pair, rank_pair, s_pair, _ = fit_and_score(pair_indices, library, e_shock)
    cond_pair = float(s_pair[0] / s_pair[-1]) if s_pair[-1] > 0 else float("inf")
    ev_pair = evaluate_selection(pair_indices, d_pair, library, baseline, dense_baseline)
    pair_row = {
        "c1": c1, "sigma1": sigma1, "d1": float(d_pair[0]),
        "c2": c2, "sigma2": sigma2, "d2": float(d_pair[1]),
        "rank": rank_pair, "cond": cond_pair,
        "r2_fit": ev_pair["r2_fit"], "r2_dense": ev_pair["r2_dense"],
        "global_error": ev_pair["global_error"], "smooth_error": ev_pair["smooth_error"],
        "shock_error_fit": ev_pair["shock_error_fit"], "shock_error_dense": ev_pair["shock_error_dense"],
        "correction_dense": ev_pair["correction_dense"],
    }

    write_summary(rows, delta_x_fit, pair_row)

    plot_shock_error_vs_k(rows, pair_row)
    plot_global_error_vs_k(rows)
    plot_r2_vs_k(rows, pair_row)
    plot_condition_number_vs_k(rows)
    plot_corrected_solutions_vs_k(rows, baseline, dense_baseline)
    plot_greedy_vs_pairsearch(rows, pair_row, dense_baseline)
    plot_final_components(rows, library, baseline, dense_baseline)

    print("[test_pde_trained_k_gaussian_correction] done.")


if __name__ == "__main__":
    main()
