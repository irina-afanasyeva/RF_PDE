"""Controlled one-seed SA-RF comparison for the 2026-11-04 research
meeting: RF baseline vs. adaptive SA-RF vs. balanced_10/balanced_100,
all at the parameter-matched prototype configuration, seed=0, with the
best preliminary coefficient LR (0.001) found by the LR sweep.

Priority order (A and B are required; C and D run only if a cumulative
soft time budget is not yet exceeded):
    A. baseline     (unit weights, no ascent)
    B. adaptive     (lambda_f=100, lambda_0~U(0,1), ascent during Adam)
    C. balanced_10  (lambda_f=lambda_0=10, frozen)        -- if time permits
    D. balanced_100 (lambda_f=lambda_0=100, frozen)       -- if time permits

Common settings (identical across all four runs): seed=0, Nx=159, Nt=19
(3,021 RF coefficients), N_F=10,000, N_0=100, N_B_PER_SIDE=25 (50 TOTAL
boundary points -- NOT yet the paper-aligned 200-total experiment, see
printed note below), coef_lr=0.001, attention LR=config.ADAM_LR=0.005,
budget="controlled" (1000 Adam + 100 L-BFGS).

Writes to NEW, non-destructive output directories
(outputs/sa_rf/one_seed_comparison/{mode}/) -- never touches
outputs/sa_rf/smoke/ or outputs/sa_rf/lr_sweep/.
"""
import json
import sys
import time
from pathlib import Path

BURGERS_DIR = Path(__file__).resolve().parents[1]
if str(BURGERS_DIR) not in sys.path:
    sys.path.insert(0, str(BURGERS_DIR))

from sa_rf import config as sa_rf_config  # noqa: E402
from sa_rf.train import run  # noqa: E402

OUTPUT_ROOT = BURGERS_DIR / "outputs" / "sa_rf" / "one_seed_comparison"
BUDGET = "controlled"
SEED = 0
COEF_LR = 0.001
PRIORITY_ORDER = ["baseline", "adaptive", "balanced_10", "balanced_100"]
SOFT_TIME_BUDGET_SECONDS = 30 * 60  # stop launching further (lower-priority) runs past this


def main():
    print(f"N_B_PER_SIDE={sa_rf_config.N_B_PER_SIDE} -- PER SIDE, i.e. "
          f"{2 * sa_rf_config.N_B_PER_SIDE} TOTAL boundary points. This is NOT yet the "
          f"paper-aligned 200-total-boundary-point experiment (that remains a separate, "
          f"not-yet-run SA-PINN-side task); it matches the completed smoke tests/LR sweep "
          f"for direct comparability.")
    print(f"coef_lr={COEF_LR} (best preliminary value from the LR sweep); "
          f"attention optimizer LR={sa_rf_config.ADAM_LR} (unchanged, config default).")
    print(f"Estimated runtime per run: ~4.8 min (from measured 'short'-budget sweep timings); "
          f"budget={BUDGET} -> {sa_rf_config.BUDGETS[BUDGET]}")
    print()

    results = {}
    sweep_start = time.time()
    for mode in PRIORITY_ORDER:
        elapsed_so_far = time.time() - sweep_start
        if mode in ("balanced_10", "balanced_100") and elapsed_so_far > SOFT_TIME_BUDGET_SECONDS:
            print(f"SKIPPING {mode}: soft time budget ({SOFT_TIME_BUDGET_SECONDS}s) exceeded "
                  f"({elapsed_so_far:.0f}s elapsed) -- stopping here per 'if time permits'.")
            break

        output_dir = OUTPUT_ROOT / mode
        if (output_dir / "config.json").exists():
            raise FileExistsError(f"{output_dir} already has a completed run -- refusing to overwrite.")

        print(f"=== [{elapsed_so_far:.0f}s elapsed] starting mode={mode} ===")
        metrics_rows, run_config = run(mode, BUDGET, seed=SEED, output_dir=output_dir, coef_lr=COEF_LR)
        results[mode] = {"metrics_rows": metrics_rows, "run_config": run_config}

    print()
    print("=" * 110)
    print(f"SA-RF ONE-SEED CONTROLLED COMPARISON (budget={BUDGET}, seed={SEED}, coef_lr={COEF_LR}, "
          f"attention_lr={sa_rf_config.ADAM_LR})")
    print("=" * 110)
    for mode in PRIORITY_ORDER:
        if mode not in results:
            print(f"\n--- mode={mode}: NOT RUN (skipped) ---")
            continue
        rows = results[mode]["metrics_rows"]
        run_config = results[mode]["run_config"]
        initial_row = next(r for r in rows if r["phase"] == "initial")
        last_row = rows[-1]

        print(f"\n--- mode={mode} ---")
        print(f"  initial: total={initial_row['total_objective']:.6e}  "
              f"pde={initial_row['weighted_pde_loss']:.4e}  ic={initial_row['weighted_ic_loss']:.4e}  "
              f"bc={initial_row['bc_mse']:.4e}  (unweighted pde_mse={initial_row['unweighted_pde_mse']:.4e}, "
              f"ic_mse={initial_row['unweighted_ic_mse']:.4e})")
        print(f"  final ({last_row['phase']}, epoch {last_row['epoch']}): "
              f"total={last_row['total_objective']:.6e}  pde={last_row['weighted_pde_loss']:.4e}  "
              f"ic={last_row['weighted_ic_loss']:.4e}  bc={last_row['bc_mse']:.4e}  "
              f"(unweighted pde_mse={last_row['unweighted_pde_mse']:.4e}, "
              f"ic_mse={last_row['unweighted_ic_mse']:.4e})")
        print(f"  global relative L2: {initial_row['global_relative_l2_error']:.4e} -> "
              f"{last_row['global_relative_l2_error']:.4e}")
        print(f"  shock relative L2:  {initial_row['shock_relative_l2_error']:.4e} -> "
              f"{last_row['shock_relative_l2_error']:.4e}")
        print(f"  RF coef grad norm: {initial_row['coef_grad_norm']:.4e} -> {last_row['coef_grad_norm']:.4e} "
              f"(coef norm final: {last_row['coef_norm']:.4e})")
        print(f"  lambda_f: {initial_row['lambda_f_mean']:.4e} -> {last_row['lambda_f_mean']:.4e} "
              f"(std final={last_row['lambda_f_std']:.4e}, grad_norm final={last_row['lambda_f_grad_norm']:.4e})")
        print(f"  lambda_0: {initial_row['lambda_0_mean']:.4e} -> {last_row['lambda_0_mean']:.4e} "
              f"(std final={last_row['lambda_0_std']:.4e}, grad_norm final={last_row['lambda_0_grad_norm']:.4e})")
        print(f"  runtime: {last_row['wall_clock_s']:.2f}s, "
              f"nonfinite anywhere: {any(r['nonfinite'] for r in rows)}, warnings: {run_config['warnings']}")

    total_elapsed = time.time() - sweep_start
    print(f"\nTotal wall-clock for this script: {total_elapsed:.1f}s ({total_elapsed/60:.2f} min)")

    summary_path = OUTPUT_ROOT / "one_seed_comparison_summary.json"
    summary_path.write_text(json.dumps(
        {mode: {"initial_row": next(r for r in results[mode]["metrics_rows"] if r["phase"] == "initial"),
                "final_row": results[mode]["metrics_rows"][-1],
                "run_config": results[mode]["run_config"]}
         for mode in results},
        indent=2, default=str,
    ))
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
