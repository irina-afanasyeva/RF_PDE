"""Longer-training SA-RF comparison for the compute server.

Three configurations, all at seed=0, Nx=159/Nt=19 (3,021 RF
coefficients), N_F=10,000, N_0=100, N_B_PER_SIDE=25 (50 total boundary
points), coef_lr=0.001, attention LR=config.ADAM_LR=0.005, budget="long"
(5,000 Adam + 1,000 L-BFGS -- new budget added to sa_rf/config.py), and
device=CUDA-if-available / dtype=float64 (sa_rf.train.run()'s existing
defaults, unchanged):

    1. baseline    -- unit attention weights, no ascent.
    2. adaptive_B  -- lambda_f_init=1,  lambda_0_init=Uniform(0,1).
    3. adaptive_C  -- lambda_f_init=10, lambda_0_init=10.

Reuses sa_rf.train.run() completely unmodified (beyond the new "long"
budget entry and the already-existing lambda_f_init/lambda_0_init
override added for the attention-init sweep) -- no new training logic.
Each configuration gets a fresh model initialization but IDENTICAL RF
features (Wx/Wt/bx/bt) and collocation points, since those are
constructed from the same seed before any mode/attention branching in
run() (see sa_rf/train.py).

RESUMABLE / NON-DESTRUCTIVE, same convention as the other sa_rf drivers:
each configuration's output directory is checked for an existing
completed run (config.json present) before launching -- already-done
configurations are skipped, never overwritten, so re-running this script
after an interruption continues where it left off. Writes to a NEW
directory (outputs/sa_rf/long_training/), never touching
outputs/sa_rf/{smoke,lr_sweep,one_seed_comparison,attention_init_sweep}/.

Convergence metrics (the full per-logged-epoch history) and final
checkpoints are already saved by run() itself (metrics.csv, final.pt) --
nothing new needed there. This script additionally writes one
consolidated comparison summary JSON across all three configurations.

NOT RUN locally -- intended for the compute server only.
"""
import json
import sys
from pathlib import Path

BURGERS_DIR = Path(__file__).resolve().parents[1]
if str(BURGERS_DIR) not in sys.path:
    sys.path.insert(0, str(BURGERS_DIR))

from sa_rf import config as sa_rf_config  # noqa: E402
from sa_rf.train import run  # noqa: E402

OUTPUT_ROOT = BURGERS_DIR / "outputs" / "sa_rf" / "long_training"
BUDGET = "long"
SEED = 0
COEF_LR = 0.001

CONFIGURATIONS = [
    ("baseline", "baseline", {}),
    ("adaptive_B", "adaptive", dict(lambda_f_init=1.0, lambda_0_init=None)),
    ("adaptive_C", "adaptive", dict(lambda_f_init=10.0, lambda_0_init=10.0)),
]


def _load_completed(output_dir):
    import csv
    run_config = json.loads((output_dir / "config.json").read_text())
    with (output_dir / "metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if k == "phase":
                continue
            try:
                r[k] = float(v)
            except ValueError:
                r[k] = float("nan")
    return rows, run_config


def main():
    print(f"seed={SEED}, coef_lr={COEF_LR}, attention_lr={sa_rf_config.ADAM_LR}, "
          f"budget={BUDGET} -> {sa_rf_config.BUDGETS[BUDGET]}")
    print(f"Nx={sa_rf_config.NX_MATCHED}, Nt={sa_rf_config.NT_MATCHED} "
          f"({sa_rf_config.NX_MATCHED * sa_rf_config.NT_MATCHED} RF coefficients), "
          f"N_F={sa_rf_config.N_F}, N_0={sa_rf_config.N_0}, "
          f"N_B_PER_SIDE={sa_rf_config.N_B_PER_SIDE} ({2 * sa_rf_config.N_B_PER_SIDE} total boundary points)")
    print()

    results = {}
    for name, mode, overrides in CONFIGURATIONS:
        output_dir = OUTPUT_ROOT / name
        if (output_dir / "config.json").exists():
            print(f"=== {name}: SKIPPING (already completed at {output_dir}) ===")
            rows, run_config = _load_completed(output_dir)
            results[name] = {"metrics_rows": rows, "run_config": run_config}
            continue

        print(f"=== {name}: starting (mode={mode}, overrides={overrides}) ===")
        metrics_rows, run_config = run(
            mode, BUDGET, seed=SEED, output_dir=output_dir, coef_lr=COEF_LR, **overrides,
        )
        results[name] = {"metrics_rows": metrics_rows, "run_config": run_config}

    print()
    print("=" * 110)
    print(f"SA-RF LONG-TRAINING COMPARISON SUMMARY (seed={SEED}, budget={BUDGET})")
    print("=" * 110)
    consolidated = {}
    for name, mode, overrides in CONFIGURATIONS:
        rows = results[name]["metrics_rows"]
        run_config = results[name]["run_config"]
        initial_row = next(r for r in rows if r["phase"] == "initial")
        last_row = rows[-1]

        print(f"\n--- {name} (mode={mode}) ---")
        print(f"  global relative L2:  {initial_row['global_relative_l2_error']:.6e} -> "
              f"{last_row['global_relative_l2_error']:.6e}")
        print(f"  smooth relative L2:  {initial_row['smooth_relative_l2_error']:.6e} -> "
              f"{last_row['smooth_relative_l2_error']:.6e}")
        print(f"  shock relative L2:   {initial_row['shock_relative_l2_error']:.6e} -> "
              f"{last_row['shock_relative_l2_error']:.6e}")
        print(f"  unweighted PDE MSE:  {initial_row['unweighted_pde_mse']:.6e} -> "
              f"{last_row['unweighted_pde_mse']:.6e}")
        print(f"  unweighted IC MSE:   {initial_row['unweighted_ic_mse']:.6e} -> "
              f"{last_row['unweighted_ic_mse']:.6e}")
        print(f"  BC MSE:              {initial_row['bc_mse']:.6e} -> {last_row['bc_mse']:.6e}")
        print(f"  lambda_f: {initial_row['lambda_f_mean']:.4e} -> {last_row['lambda_f_mean']:.4e}")
        print(f"  lambda_0: {initial_row['lambda_0_mean']:.4e} -> {last_row['lambda_0_mean']:.4e}")
        print(f"  runtime: {last_row['wall_clock_s']:.2f}s, "
              f"nonfinite anywhere: {any(r['nonfinite'] for r in rows)}, warnings: {run_config['warnings']}")

        consolidated[name] = {"initial_row": initial_row, "final_row": last_row, "run_config": run_config}

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_ROOT / "long_training_summary.json"
    summary_path.write_text(json.dumps(consolidated, indent=2, default=str))
    print(f"\nConsolidated summary written to {summary_path}")


if __name__ == "__main__":
    main()
