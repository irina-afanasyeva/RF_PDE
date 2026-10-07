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

from burgers_rf.config import DTYPE, M_TEST, N_TEST_TIMES, NV  # noqa: E402
from burgers_rf.data import make_reference_values, make_test_grid  # noqa: E402
from burgers_rf.local_features import GaussianLocal  # noqa: E402
from burgers_rf.evaluation import evaluate_error_metrics  # noqa: E402
from sweep_local_width import build_shock_grid  # noqa: E402
import test_two_stage_two_gaussian_local_feature as t2g  # noqa: E402 -- reused unchanged:
# load_frozen_rf_model, find_exact_library_value, compute_oracle_amplitudes,
# u_fn_factory, verify_pretraining_state, reconstruct_interior_points,
# snapshot_frozen_tensors, check_frozen_invariance, train_stage2,
# evaluate_fixed_two_gaussian. t2g.WEIGHT_DECAY is overridden as a module
# attribute at runtime per case (the file on disk is never modified);
# train_stage2's `weight_decay=WEIGHT_DECAY` is a global lookup resolved at
# call time, so this correctly changes the optimizer's behavior per case.

# WEIGHT-DECAY SENSITIVITY SWEEP of the frozen-RF two-Gaussian experiment.
# Reuses test_two_stage_two_gaussian_local_feature.py's functions directly;
# does NOT modify that file. For every case: same frozen RF checkpoint,
# same fixed Gaussian geometry, same d1=d2=0 initialization, same canonical
# seed/interior points (re-derived fresh via t2g.reconstruct_interior_points
# immediately before each case, resetting the global RNG identically so
# per-epoch IC/BC draws are bit-identical across cases), same LR/EPOCHS/
# IC_BC_WEIGHT/optimizer type/loss definitions/evaluation grids. The ONLY
# varying factor is weight_decay.
#
# This is a diagnostic of amplitude suppression under the current training
# objective, NOT an attempt to tune the final model. We do NOT conclude in
# advance that weight decay is the cause.

WEIGHT_DECAY_VALUES = [1.0, 0.1, 0.01, 0.0]

REPRO_D1 = 0.035453350169778575
REPRO_D2 = -0.03832831801950355
REPRO_GLOBAL = 0.26946486173938333
REPRO_SMOOTH = 0.243773446446857
REPRO_SHOCK = 0.8169198499822581
REPRO_T1_SHOCK = 0.7170982076598021
REPRO_RTOL = 1e-4

OUTPUT_DIR = PROJECT_DIR / "outputs" / "two_stage_two_gaussian_weight_decay_sweep"


def rel_diff(a, b):
    return abs(a - b) / abs(b)


def run_one_case(model, checkpoint, c1, sigma1, c2, sigma2, G1, G2, weight_decay, device,
                  x_test, t_test, y_true, x_shock, t_shock, y_true_shock):
    x_train, t_train = t2g.reconstruct_interior_points(checkpoint, device)

    d1 = torch.nn.Parameter(torch.zeros(1))
    d2 = torch.nn.Parameter(torch.zeros(1))
    u_fn = t2g.u_fn_factory(model, G1, G2, d1, d2)

    print(f"--- weight_decay={weight_decay} ---")
    t2g.verify_pretraining_state(model, checkpoint, device, u_fn, d1, d2)

    t2g.WEIGHT_DECAY = weight_decay
    history, grad_d1_initial, grad_d2_initial, d1_after_first_step, d2_after_first_step = t2g.train_stage2(
        u_fn, d1, d2, x_train, t_train, device
    )

    d1_final, d2_final = d1.item(), d2.item()

    with torch.no_grad():
        y_pred = u_fn(x_test, t_test).cpu().numpy().reshape(-1)
        y_pred_shock = u_fn(x_shock, t_shock).cpu().numpy().reshape(-1)
    canonical_metrics = evaluate_error_metrics(x_test, y_true, y_pred, y_true_shock, y_pred_shock)

    with torch.no_grad():
        u_rf_test = model(x_test, t_test)
        u_rf_shock_canon = model(x_shock, t_shock)
        local_test = t_test * (d1_final * G1(x_test) + d2_final * G2(x_test))
        local_shock_canon = t_shock * (d1_final * G1(x_shock) + d2_final * G2(x_shock))
    ratio_full = float(torch.linalg.norm(local_test) / torch.linalg.norm(u_rf_test))
    ratio_shock = float(torch.linalg.norm(local_shock_canon) / torch.linalg.norm(u_rf_shock_canon))

    return {
        "weight_decay": weight_decay,
        "history": history,
        "grad_d1_initial": grad_d1_initial, "grad_d2_initial": grad_d2_initial,
        "d1_after_first_step": d1_after_first_step, "d2_after_first_step": d2_after_first_step,
        "d1_final": d1_final, "d2_final": d2_final,
        "canonical_metrics": canonical_metrics,
        "ratio_full": ratio_full, "ratio_shock": ratio_shock,
    }


def verify_reproduction_gate(result):
    """The weight_decay=1.0 case MUST reproduce the completed Task-2
    server result within REPRO_RTOL. Raises (stopping before any other
    case is interpreted) if it does not."""
    checks = {
        "d1": rel_diff(result["d1_final"], REPRO_D1),
        "d2": rel_diff(result["d2_final"], REPRO_D2),
        "global": rel_diff(result["canonical_metrics"]["global_relative_l2_error"], REPRO_GLOBAL),
        "smooth": rel_diff(result["canonical_metrics"]["smooth_relative_l2_error"], REPRO_SMOOTH),
        "shock": rel_diff(result["canonical_metrics"]["shock_relative_l2_error"], REPRO_SHOCK),
        "t1_shock": rel_diff(result["metrics_pde_t1"]["shock_relative_l2_error"], REPRO_T1_SHOCK),
    }
    print("weight_decay=1.0 reproduction gate (vs. completed Task-2 server result):")
    for k, v in checks.items():
        print(f"  {k}: relative difference = {v:.3e} (tolerance {REPRO_RTOL:.1e})")
    failed = {k: v for k, v in checks.items() if v > REPRO_RTOL}
    if failed:
        raise RuntimeError(
            f"weight_decay=1.0 REPRODUCTION GATE FAILED: {failed}. Refusing to run/interpret "
            "the remaining weight_decay sweep cases."
        )
    print("Reproduction gate PASSED.")
    return checks


def write_sweep_csv(results, oracle_info):
    d1_oracle, d2_oracle = oracle_info["d1_oracle"], oracle_info["d2_oracle"]
    header = [
        "weight_decay", "d1_final", "d2_final",
        "d1_pde_over_oracle", "d2_pde_over_oracle",
        "abs_d1_pde_over_oracle", "abs_d2_pde_over_oracle",
        "global_error", "smooth_error", "shock_error",
        "t1_global_error", "t1_smooth_error", "t1_shock_error",
        "final_pde_loss", "final_ic_loss", "final_bc_loss",
        "ratio_full", "ratio_shock",
        "grad_d1_initial", "grad_d2_initial",
        "d1_after_first_step", "d2_after_first_step",
    ]
    path = OUTPUT_DIR / "weight_decay_sweep_summary.csv"
    with path.open("w") as f:
        f.write(",".join(header) + "\n")
        for r in results:
            h_last = r["history"][-1]
            row = {
                "weight_decay": r["weight_decay"],
                "d1_final": r["d1_final"], "d2_final": r["d2_final"],
                "d1_pde_over_oracle": r["d1_final"] / d1_oracle if d1_oracle != 0 else float("nan"),
                "d2_pde_over_oracle": r["d2_final"] / d2_oracle if d2_oracle != 0 else float("nan"),
                "abs_d1_pde_over_oracle": abs(r["d1_final"]) / abs(d1_oracle) if d1_oracle != 0 else float("nan"),
                "abs_d2_pde_over_oracle": abs(r["d2_final"]) / abs(d2_oracle) if d2_oracle != 0 else float("nan"),
                "global_error": r["canonical_metrics"]["global_relative_l2_error"],
                "smooth_error": r["canonical_metrics"]["smooth_relative_l2_error"],
                "shock_error": r["canonical_metrics"]["shock_relative_l2_error"],
                "t1_global_error": r["metrics_pde_t1"]["global_relative_l2_error"],
                "t1_smooth_error": r["metrics_pde_t1"]["smooth_relative_l2_error"],
                "t1_shock_error": r["metrics_pde_t1"]["shock_relative_l2_error"],
                "final_pde_loss": h_last["pde_loss"], "final_ic_loss": h_last["ic_loss"], "final_bc_loss": h_last["bc_loss"],
                "ratio_full": r["ratio_full"], "ratio_shock": r["ratio_shock"],
                "grad_d1_initial": r["grad_d1_initial"], "grad_d2_initial": r["grad_d2_initial"],
                "d1_after_first_step": r["d1_after_first_step"], "d2_after_first_step": r["d2_after_first_step"],
            }
            f.write(",".join(str(row[k]) for k in header) + "\n")


def write_case_history_csv(result):
    header = ["epoch", "d1", "d2", "pde_loss", "ic_loss", "bc_loss", "total_loss"]
    path = OUTPUT_DIR / f"training_history_wd_{result['weight_decay']}.csv"
    with path.open("w") as f:
        f.write(",".join(header) + "\n")
        for row in result["history"]:
            f.write(",".join(str(row[k]) for k in header) + "\n")


def write_summary(results, oracle_info, repro_checks, frozen_diffs):
    d1_oracle, d2_oracle = oracle_info["d1_oracle"], oracle_info["d2_oracle"]
    case0 = results[0]

    lines = []
    lines.append(f"WEIGHT_DECAY_VALUES = {WEIGHT_DECAY_VALUES}")
    lines.append(f"d1_oracle = {d1_oracle}, d2_oracle = {d2_oracle}")
    lines.append("")

    lines.append("=== weight_decay=1.0 reproduction gate ===")
    for k, v in repro_checks.items():
        lines.append(f"  {k}: relative difference = {v:.3e} (tolerance {REPRO_RTOL:.1e})")
    lines.append("Gate PASSED (script would have stopped otherwise).")
    lines.append("")

    lines.append("=== Initial-gradient identity check across weight_decay cases ===")
    lines.append(
        "Because weight_decay*param=0 at d1=d2=0 regardless of weight_decay's "
        "value, the initial data-loss gradients should be IDENTICAL across "
        "all four cases (weight decay modifies the gradient only inside "
        "optimizer.step(), never during backward())."
    )
    max_diff_grad_d1 = max(abs(r["grad_d1_initial"] - case0["grad_d1_initial"]) for r in results)
    max_diff_grad_d2 = max(abs(r["grad_d2_initial"] - case0["grad_d2_initial"]) for r in results)
    lines.append(f"max |grad_d1_initial(case) - grad_d1_initial(wd=1.0)| = {max_diff_grad_d1} (expected 0.0)")
    lines.append(f"max |grad_d2_initial(case) - grad_d2_initial(wd=1.0)| = {max_diff_grad_d2} (expected 0.0)")
    lines.append("")

    lines.append("=== First-optimizer-step identity check across weight_decay cases ===")
    lines.append(
        "Since weight_decay*0=0 at the first step too (param is still exactly "
        "0 at the moment the first update is computed), d1,d2 after epoch 0 "
        "should ALSO be IDENTICAL across all four weight_decay values -- "
        "divergence should only begin from epoch 1 onward, once d1,d2 become "
        "nonzero."
    )
    max_diff_d1_step1 = max(abs(r["d1_after_first_step"] - case0["d1_after_first_step"]) for r in results)
    max_diff_d2_step1 = max(abs(r["d2_after_first_step"] - case0["d2_after_first_step"]) for r in results)
    lines.append(f"max |d1_after_first_step(case) - d1_after_first_step(wd=1.0)| = {max_diff_d1_step1} (expected 0.0)")
    lines.append(f"max |d2_after_first_step(case) - d2_after_first_step(wd=1.0)| = {max_diff_d2_step1} (expected 0.0)")
    lines.append("")

    lines.append("=== Frozen-RF invariance check across the ENTIRE sweep (max |after-before|, expected 0.0) ===")
    for k, v in frozen_diffs.items():
        lines.append(f"  {k}: {v}")
    lines.append("")

    lines.append("=== Per-case results ===")
    for r in results:
        lines.append(f"--- weight_decay = {r['weight_decay']} ---")
        lines.append(f"final d1 = {r['d1_final']}, final d2 = {r['d2_final']}")
        if d1_oracle != 0:
            lines.append(f"d1_PDE/d1_oracle = {r['d1_final']/d1_oracle}, |d1_PDE|/|d1_oracle| = {abs(r['d1_final'])/abs(d1_oracle)}")
        if d2_oracle != 0:
            lines.append(f"d2_PDE/d2_oracle = {r['d2_final']/d2_oracle}, |d2_PDE|/|d2_oracle| = {abs(r['d2_final'])/abs(d2_oracle)}")
        lines.append(f"canonical: {r['canonical_metrics']}")
        lines.append(f"t=1-only: {r['metrics_pde_t1']}")
        h_last = r["history"][-1]
        lines.append(f"final_pde_loss = {h_last['pde_loss']}, final_ic_loss = {h_last['ic_loss']}, final_bc_loss = {h_last['bc_loss']}")
        lines.append(f"ratio_full = {r['ratio_full']}, ratio_shock = {r['ratio_shock']}")
        lines.append("")

    lines.append("=== Interpretive scaffold (questions the numbers above let you answer; not asserted here) ===")
    lines.append("A. Does reducing weight decay allow amplitudes to grow? Compare |d1_final|,|d2_final| across the table above as weight_decay decreases.")
    lines.append("B. Does that growth move them toward the oracle amplitudes? Compare |d_PDE|/|d_oracle| across the table above.")
    lines.append("C. Does solution error improve? Compare canonical/t=1 shock error across the table above.")
    lines.append("D. What happens to the PDE objective? Compare final_pde_loss across the table above.")
    lines.append(
        "This is a diagnostic of amplitude suppression under the current "
        "objective, not an attempt to tune the final model. No conclusion "
        "about weight decay being THE cause is asserted here."
    )

    (OUTPUT_DIR / "weight_decay_sweep_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def plot_final_d_vs_weight_decay(results, d1_oracle, d2_oracle):
    positions = list(range(len(results)))
    labels = [str(r["weight_decay"]) for r in results]
    d1_values = [r["d1_final"] for r in results]
    d2_values = [r["d2_final"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(positions, d1_values, "o-", color="tab:blue", label="d1_final")
    ax.plot(positions, d2_values, "o-", color="tab:orange", label="d2_final")
    ax.axhline(d1_oracle, color="tab:blue", linestyle="--", linewidth=1, label=f"d1_oracle={d1_oracle:.4g}")
    ax.axhline(d2_oracle, color="tab:orange", linestyle="--", linewidth=1, label=f"d2_oracle={d2_oracle:.4g}")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel("weight_decay")
    ax.set_ylabel("amplitude")
    ax.set_title("Final d1,d2 vs weight_decay")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "final_d_vs_weight_decay.png")
    plt.close(fig)


def plot_amplitude_fraction_vs_weight_decay(results, d1_oracle, d2_oracle):
    positions = list(range(len(results)))
    labels = [str(r["weight_decay"]) for r in results]
    frac_d1 = [abs(r["d1_final"]) / abs(d1_oracle) for r in results]
    frac_d2 = [abs(r["d2_final"]) / abs(d2_oracle) for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(positions, frac_d1, "o-", label="|d1_PDE|/|d1_oracle|")
    ax.plot(positions, frac_d2, "o-", label="|d2_PDE|/|d2_oracle|")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel("weight_decay")
    ax.set_ylabel("amplitude fraction")
    ax.set_title("Amplitude fraction vs weight_decay")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "amplitude_fraction_vs_weight_decay.png")
    plt.close(fig)


def plot_shock_error_vs_weight_decay(results):
    positions = list(range(len(results)))
    labels = [str(r["weight_decay"]) for r in results]
    canonical_shock = [r["canonical_metrics"]["shock_relative_l2_error"] for r in results]
    t1_shock = [r["metrics_pde_t1"]["shock_relative_l2_error"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(positions, canonical_shock, "o-", label="canonical shock error")
    ax.plot(positions, t1_shock, "s--", label="t=1 shock error")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel("weight_decay")
    ax.set_ylabel("shock relative L2 error")
    ax.set_title("Shock error vs weight_decay")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "shock_error_vs_weight_decay.png")
    plt.close(fig)


def plot_d_trajectories(results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for r in results:
        epochs = [row["epoch"] for row in r["history"]]
        d1_values = [row["d1"] for row in r["history"]]
        d2_values = [row["d2"] for row in r["history"]]
        axes[0].plot(epochs, d1_values, label=f"weight_decay={r['weight_decay']}")
        axes[1].plot(epochs, d2_values, label=f"weight_decay={r['weight_decay']}")
    axes[0].set_title("d1 vs epoch")
    axes[1].set_title("d2 vs epoch")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "d_trajectories_vs_epoch.png")
    plt.close(fig)


def main():
    torch.set_default_dtype(DTYPE)
    device = "cpu"

    model, checkpoint = t2g.load_frozen_rf_model(device)

    c1 = t2g.find_exact_library_value(t2g.C1_NOMINAL, t2g.tfgc.C_VALUES, t2g.GEOMETRY_MATCH_RTOL, "c1")
    sigma1 = t2g.find_exact_library_value(t2g.SIGMA1_NOMINAL, t2g.tfgc.SIGMA_VALUES, t2g.GEOMETRY_MATCH_RTOL, "sigma1")
    c2 = t2g.find_exact_library_value(t2g.C2_NOMINAL, t2g.tfgc.C_VALUES, t2g.GEOMETRY_MATCH_RTOL, "c2")
    sigma2 = t2g.find_exact_library_value(t2g.SIGMA2_NOMINAL, t2g.tfgc.SIGMA_VALUES, t2g.GEOMETRY_MATCH_RTOL, "sigma2")
    print(f"Fixed geometry: c1={c1}, sigma1={sigma1}, c2={c2}, sigma2={sigma2}")

    oracle_info = t2g.compute_oracle_amplitudes(model, device, c1, sigma1, c2, sigma2)
    print(f"d1_oracle={oracle_info['d1_oracle']}, d2_oracle={oracle_info['d2_oracle']}")

    G1 = GaussianLocal(center=c1, width=sigma1)
    G2 = GaussianLocal(center=c2, width=sigma2)

    frozen_before = t2g.snapshot_frozen_tensors(model)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    nv = checkpoint.get("NV", NV)
    x_test, t_test = make_test_grid(M_TEST, N_TEST_TIMES, device)
    y_true = make_reference_values(t_test, x_test, nv)
    x_shock, t_shock = build_shock_grid(t_test, device)
    y_true_shock = make_reference_values(t_shock, x_shock, nv)

    results = []
    repro_checks = None
    for i, wd in enumerate(WEIGHT_DECAY_VALUES):
        result = run_one_case(
            model, checkpoint, c1, sigma1, c2, sigma2, G1, G2, wd, device,
            x_test, t_test, y_true, x_shock, t_shock, y_true_shock,
        )
        metrics_pde_t1, _, _ = t2g.evaluate_fixed_two_gaussian(
            oracle_info["baseline"], c1, sigma1, result["d1_final"], c2, sigma2, result["d2_final"]
        )
        result["metrics_pde_t1"] = metrics_pde_t1
        write_case_history_csv(result)

        if i == 0:
            repro_checks = verify_reproduction_gate(result)

        results.append(result)

    frozen_diffs = t2g.check_frozen_invariance(model, frozen_before)

    write_sweep_csv(results, oracle_info)
    write_summary(results, oracle_info, repro_checks, frozen_diffs)

    plot_final_d_vs_weight_decay(results, oracle_info["d1_oracle"], oracle_info["d2_oracle"])
    plot_amplitude_fraction_vs_weight_decay(results, oracle_info["d1_oracle"], oracle_info["d2_oracle"])
    plot_shock_error_vs_weight_decay(results)
    plot_d_trajectories(results)

    print("[sweep_two_stage_two_gaussian_weight_decay] done.")


if __name__ == "__main__":
    main()
