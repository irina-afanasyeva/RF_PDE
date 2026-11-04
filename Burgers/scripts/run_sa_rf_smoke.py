"""Phase 6: short, inexpensive smoke tests for SA-RF's three controlled
comparison modes, at the parameter-matched prototype configuration
(Nx=159, Nt=19 -> 3,021 RF coefficients; N_F=10,000, N_0=100,
N_B_PER_SIDE=25 -- matching SA-PINN's current attention-parameter
budget). Uses the "short" budget (300 Adam + 50 L-BFGS iterations) --
NOT the paper-scale 10,000/10,000 budget. Writes to a NEW,
non-destructive output directory (outputs/sa_rf/smoke/); does not touch
any existing experiment output.
"""
import json
import sys
from pathlib import Path

BURGERS_DIR = Path(__file__).resolve().parents[1]
if str(BURGERS_DIR) not in sys.path:
    sys.path.insert(0, str(BURGERS_DIR))

from sa_rf import config as sa_rf_config  # noqa: E402
from sa_rf.train import MODES, run  # noqa: E402

OUTPUT_ROOT = BURGERS_DIR / "outputs" / "sa_rf" / "smoke"
BUDGET = "short"
SEED = 0


def main():
    results = {}
    for mode in MODES:
        output_dir = OUTPUT_ROOT / mode
        if (output_dir / "config.json").exists():
            raise FileExistsError(
                f"{output_dir} already has a completed run -- refusing to overwrite. "
                "Remove it manually first if a fresh smoke test is really wanted."
            )
        metrics_rows, run_config = run(mode, BUDGET, seed=SEED, output_dir=output_dir)
        results[mode] = {"metrics_rows": metrics_rows, "run_config": run_config}

    print()
    print("=" * 100)
    print("SA-RF SHORT SMOKE TEST SUMMARY (budget=short: 300 Adam + 50 L-BFGS, seed=0)")
    print("=" * 100)
    for mode in MODES:
        rows = results[mode]["metrics_rows"]
        run_config = results[mode]["run_config"]
        first_adam = next(r for r in rows if r["phase"] == "adam")
        last_row = rows[-1]
        print(f"\n--- mode={mode} ---")
        print(f"  parameters: {run_config['n_rf_coefficients']} RF coefficients, "
              f"{run_config['n_attention_parameters']} attention parameters "
              f"({run_config['n_f']} PDE + {run_config['n_0']} IC)")
        print(f"  initial total loss (epoch 0):  {first_adam['total_objective']:.6e} "
              f"(pde={first_adam['weighted_pde_loss']:.4e}, ic={first_adam['weighted_ic_loss']:.4e}, "
              f"bc={first_adam['bc_mse']:.4e})")
        print(f"  final total loss ({last_row['phase']}, epoch {last_row['epoch']}): "
              f"{last_row['total_objective']:.6e} "
              f"(pde={last_row['weighted_pde_loss']:.4e}, ic={last_row['weighted_ic_loss']:.4e}, "
              f"bc={last_row['bc_mse']:.4e})")
        print(f"  unweighted PDE MSE: {first_adam['unweighted_pde_mse']:.4e} -> {last_row['unweighted_pde_mse']:.4e}")
        print(f"  unweighted IC MSE:  {first_adam['unweighted_ic_mse']:.4e} -> {last_row['unweighted_ic_mse']:.4e}")
        print(f"  global relative L2: {first_adam['global_relative_l2_error']:.4e} -> "
              f"{last_row['global_relative_l2_error']:.4e}")
        print(f"  shock relative L2:  {first_adam['shock_relative_l2_error']:.4e} -> "
              f"{last_row['shock_relative_l2_error']:.4e}")
        print(f"  RF coefficient grad norm (final): {last_row['coef_grad_norm']:.4e}, "
              f"coef norm (final): {last_row['coef_norm']:.4e}")
        print(f"  lambda_f: mean={last_row['lambda_f_mean']:.4e} std={last_row['lambda_f_std']:.4e} "
              f"grad_norm={last_row['lambda_f_grad_norm']:.4e}")
        print(f"  lambda_0: mean={last_row['lambda_0_mean']:.4e} std={last_row['lambda_0_std']:.4e} "
              f"grad_norm={last_row['lambda_0_grad_norm']:.4e}")
        print(f"  runtime: {last_row['wall_clock_s']:.2f}s, nonfinite anywhere: "
              f"{any(r['nonfinite'] for r in rows)}, warnings: {run_config['warnings']}")

    summary_path = OUTPUT_ROOT / "smoke_summary.json"
    summary_path.write_text(json.dumps(
        {mode: {"final_row": results[mode]["metrics_rows"][-1], "run_config": results[mode]["run_config"]}
         for mode in MODES},
        indent=2, default=str,
    ))
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
