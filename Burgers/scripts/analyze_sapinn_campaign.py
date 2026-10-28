"""Read-only multi-seed statistical analysis of the SA-PINN Burgers
campaign (seed 0, plus however many of seeds 1-9 have completed so far --
this script works correctly with any number of available seeds per mode,
including just seed 0, and is meant to be re-run as the campaign
progresses). Deliberately kept separate from sapinn/train.py (training
code) and from scripts/run_sapinn_campaign.py (the orchestrator) -- this
script never trains anything, only reads existing results.
"""
import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

BURGERS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BURGERS_DIR))

from sapinn import config as sconfig  # noqa: E402

MODES = ["baseline", "fixed_weight", "adaptive", "balanced_10", "balanced_100"]
RESULTS_ROOT = BURGERS_DIR / "outputs" / "sapinn_burgers"
BUDGET = "paper"

COLORS = {
    "baseline": "tab:blue", "fixed_weight": "tab:orange", "adaptive": "tab:green",
    "balanced_10": "tab:red", "balanced_100": "tab:purple",
}

# Published reference (McClenny & Braga-Neto, arXiv:2009.04544, Section 4.1
# viscous Burgers benchmark): 10 UNSEEDED random restarts, Adam 10k + L-BFGS
# 10k. See sapinn/config.py for the full list of disclosed implementation
# differences (dtype, sampling method, N_b discrepancy, no seed in the
# reference, etc.) -- this is NOT a reproduction, only a magnitude check.
PUBLISHED_ADAPTIVE_MEAN = 4.80e-4
PUBLISHED_ADAPTIVE_STD = 1.01e-4
PUBLISHED_N_RESTARTS = 10


def discover_available_seeds(mode):
    mode_dir = RESULTS_ROOT / mode / BUDGET
    if not mode_dir.exists():
        return []
    seeds = []
    for child in sorted(mode_dir.iterdir()):
        if child.name.startswith("seed") and (child / "config.json").exists() and (child / "metrics.csv").exists():
            try:
                seeds.append(int(child.name[len("seed"):]))
            except ValueError:
                continue
    return sorted(seeds)


def load_final_row(mode, seed):
    path = RESULTS_ROOT / mode / BUDGET / f"seed{seed}" / "metrics.csv"
    with path.open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if k in ("phase", "nonfinite"):
                continue
            try:
                r[k] = float(v)
            except ValueError:
                r[k] = float("nan")
    adam = [r for r in rows if r["phase"] == "adam"]
    lbfgs = [r for r in rows if r["phase"] == "lbfgs"]
    adam_eval = [r for r in adam if not np.isnan(r["global_relative_l2_error"])]
    final = lbfgs[-1] if lbfgs else adam[-1]
    config_data = json.loads((RESULTS_ROOT / mode / BUDGET / f"seed{seed}" / "config.json").read_text())
    adam_wall = adam[-1]["wall_clock_s"]
    total_wall = final["wall_clock_s"]
    return {
        "mode": mode, "seed": seed,
        "global_relative_l2_error": final["global_relative_l2_error"],
        "smooth_relative_l2_error": final["smooth_relative_l2_error"],
        "shock_relative_l2_error": final["shock_relative_l2_error"],
        "global_l2_after_adam": adam_eval[-1]["global_relative_l2_error"] if adam_eval else float("nan"),
        "unweighted_pde_mse": final["unweighted_pde_mse"],
        "unweighted_ic_mse": final["unweighted_ic_mse"],
        "bc_mse": final["bc_mse"],
        "lbfgs_iters_completed": lbfgs[-1]["epoch"] if lbfgs else 0,
        "lbfgs_iters_requested": config_data["budget"]["lbfgs_iters"],
        "closure_evals": final["closure_evals"],
        "adam_wall_clock_s": adam_wall,
        "lbfgs_wall_clock_s": total_wall - adam_wall,
        "total_wall_clock_s": total_wall,
    }


def mean_std_median(values):
    values = [v for v in values if not np.isnan(v)]
    if not values:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "median": float("nan")}
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else float("nan"),
        "median": statistics.median(values),
    }


def main():
    out_dir = RESULTS_ROOT / "campaign_analysis"
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    per_seed_rows = []
    available_seeds = {}
    for mode in MODES:
        seeds = discover_available_seeds(mode)
        available_seeds[mode] = seeds
        print(f"{mode}: {len(seeds)} seed(s) available -> {seeds}")
        for seed in seeds:
            per_seed_rows.append(load_final_row(mode, seed))

    if not per_seed_rows:
        print("No completed runs found yet -- nothing to analyze.")
        return

    per_seed_header = list(per_seed_rows[0].keys())
    with (out_dir / "per_seed_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=per_seed_header)
        writer.writeheader()
        writer.writerows(per_seed_rows)

    # ------------------------------------------------------------------
    # Per-mode aggregate statistics
    # ------------------------------------------------------------------
    metric_keys = [
        "global_relative_l2_error", "shock_relative_l2_error", "smooth_relative_l2_error",
        "global_l2_after_adam", "unweighted_pde_mse", "unweighted_ic_mse", "bc_mse",
        "lbfgs_iters_completed", "total_wall_clock_s",
    ]
    summary_rows = []
    for mode in MODES:
        mode_rows = [r for r in per_seed_rows if r["mode"] == mode]
        row = {"mode": mode, "n_seeds": len(mode_rows), "seeds": [r["seed"] for r in mode_rows]}
        for key in metric_keys:
            stats = mean_std_median([r[key] for r in mode_rows])
            row[f"{key}_mean"] = stats["mean"]
            row[f"{key}_std"] = stats["std"]
            row[f"{key}_median"] = stats["median"]
        summary_rows.append(row)

    with (out_dir / "summary_statistics.json").open("w") as f:
        json.dump(summary_rows, f, indent=2, default=str)

    print()
    print("=== Per-mode summary (global relative L2) ===")
    for row in summary_rows:
        print(f"  {row['mode']:15s} n={row['n_seeds']}  "
              f"mean={row['global_relative_l2_error_mean']:.4e}  "
              f"std={row['global_relative_l2_error_std']:.4e}  "
              f"median={row['global_relative_l2_error_median']:.4e}")

    # ------------------------------------------------------------------
    # Paired per-seed differences between methods (only seeds present in BOTH)
    # ------------------------------------------------------------------
    by_mode_seed = {(r["mode"], r["seed"]): r for r in per_seed_rows}
    pairs_of_interest = [
        ("adaptive", "baseline"), ("adaptive", "fixed_weight"),
        ("balanced_10", "adaptive"), ("balanced_100", "adaptive"),
        ("balanced_100", "balanced_10"),
    ]
    paired_rows = []
    for mode_a, mode_b in pairs_of_interest:
        common_seeds = sorted(set(available_seeds.get(mode_a, [])) & set(available_seeds.get(mode_b, [])))
        diffs = []
        for seed in common_seeds:
            a = by_mode_seed[(mode_a, seed)]["global_relative_l2_error"]
            b = by_mode_seed[(mode_b, seed)]["global_relative_l2_error"]
            diffs.append(a - b)
        stats = mean_std_median(diffs)
        paired_rows.append({
            "mode_a": mode_a, "mode_b": mode_b, "n_common_seeds": len(common_seeds),
            "common_seeds": common_seeds,
            "mean_diff_global_l2_a_minus_b": stats["mean"], "std_diff": stats["std"],
            "a_better_count": sum(1 for d in diffs if d < 0), "b_better_count": sum(1 for d in diffs if d > 0),
        })
    with (out_dir / "paired_differences.json").open("w") as f:
        json.dump(paired_rows, f, indent=2, default=str)
    print()
    print("=== Paired per-seed differences (global relative L2, mode_a - mode_b) ===")
    for row in paired_rows:
        print(f"  {row['mode_a']} vs {row['mode_b']}: n_common={row['n_common_seeds']}, "
              f"mean_diff={row['mean_diff_global_l2_a_minus_b']}, "
              f"{row['mode_a']}_better={row['a_better_count']}, {row['mode_b']}_better={row['b_better_count']}")

    # ------------------------------------------------------------------
    # Published-benchmark comparison (adaptive only)
    # ------------------------------------------------------------------
    adaptive_rows = [r for r in per_seed_rows if r["mode"] == "adaptive"]
    adaptive_stats = mean_std_median([r["global_relative_l2_error"] for r in adaptive_rows])
    print()
    print("=== Comparison with published SA-PINN Burgers benchmark (adaptive mode only) ===")
    print(f"  Published: mean={PUBLISHED_ADAPTIVE_MEAN:.4e}, std={PUBLISHED_ADAPTIVE_STD:.4e}, "
          f"n_restarts={PUBLISHED_N_RESTARTS} (UNSEEDED)")
    print(f"  Ours:      mean={adaptive_stats['mean']:.4e}, std={adaptive_stats['std']:.4e}, "
          f"n_seeds={adaptive_stats['n']} (seeded, seeds={[r['seed'] for r in adaptive_rows]})")
    (out_dir / "published_comparison.json").write_text(json.dumps({
        "published_mean": PUBLISHED_ADAPTIVE_MEAN, "published_std": PUBLISHED_ADAPTIVE_STD,
        "published_n_restarts": PUBLISHED_N_RESTARTS,
        "ours_mean": adaptive_stats["mean"], "ours_std": adaptive_stats["std"], "ours_n": adaptive_stats["n"],
        "ours_seeds": [r["seed"] for r in adaptive_rows],
    }, indent=2))

    # ------------------------------------------------------------------
    # Plots (all gracefully handle n=1 per mode, e.g. only seed 0 so far)
    # ------------------------------------------------------------------
    # (a) Individual seeds + mean/std, per mode, global relative L2
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, mode in enumerate(MODES):
        mode_rows = [r for r in per_seed_rows if r["mode"] == mode]
        ys = [r["global_relative_l2_error"] for r in mode_rows]
        xs = [i + np.random.uniform(-0.08, 0.08) for _ in ys]  # jitter for visibility
        ax.scatter(xs, ys, color=COLORS[mode], alpha=0.6, label=f"{mode} (seeds)" if i == 0 else None, zorder=2)
        stats = mean_std_median(ys)
        if stats["n"] > 0:
            ax.errorbar([i], [stats["mean"]], yerr=[stats["std"]] if stats["n"] > 1 else None,
                        fmt="D", color="black", capsize=5, zorder=3,
                        label="mean +/- std" if i == 0 else None)
    ax.set_xticks(range(len(MODES)))
    ax.set_xticklabels(MODES, rotation=20)
    ax.set_yscale("log")
    ax.set_ylabel("global relative L2 error")
    ax.set_title("Individual-seed results and mean +/- sample std, per mode")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "individual_seeds_and_mean_std.png")
    plt.close(fig)

    # (b) Global vs shock-region accuracy scatter
    fig, ax = plt.subplots(figsize=(7, 6))
    for mode in MODES:
        mode_rows = [r for r in per_seed_rows if r["mode"] == mode]
        ax.scatter([r["global_relative_l2_error"] for r in mode_rows],
                   [r["shock_relative_l2_error"] for r in mode_rows],
                   color=COLORS[mode], label=mode)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("global relative L2 error")
    ax.set_ylabel("shock-region relative L2 error")
    ax.set_title("Global vs. shock-region accuracy, per seed")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "global_vs_shock_accuracy.png")
    plt.close(fig)

    # (c) Accuracy vs computational cost
    fig, ax = plt.subplots(figsize=(7, 6))
    for mode in MODES:
        mode_rows = [r for r in per_seed_rows if r["mode"] == mode]
        ax.scatter([r["total_wall_clock_s"] for r in mode_rows],
                   [r["global_relative_l2_error"] for r in mode_rows],
                   color=COLORS[mode], label=mode)
    ax.set_yscale("log")
    ax.set_xlabel("total wall-clock time (s, Adam+L-BFGS)")
    ax.set_ylabel("global relative L2 error")
    ax.set_title("Accuracy vs. computational cost, per seed")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "accuracy_vs_cost.png")
    plt.close(fig)

    print()
    print(f"Saved: per_seed_results.csv, summary_statistics.json, paired_differences.json, "
          f"published_comparison.json, plots/*.png -- all under {out_dir}")
    print("DONE.")


if __name__ == "__main__":
    main()
