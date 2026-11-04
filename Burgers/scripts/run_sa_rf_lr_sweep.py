"""Controlled SA-RF learning-rate sensitivity experiment (coefficient
optimizer only), for modes fixed_weight and adaptive.

Sweeps coef_lr in {0.005, 0.001, 0.0005} -- the RF-coefficient Adam
optimizer's learning rate -- while holding EVERYTHING else fixed:
  - attention ascent-optimizer learning rate: config.ADAM_LR (unchanged,
    never swept -- see sa_rf/train.py's run(coef_lr=...) docstring).
  - RF features (Nx=159, Nt=19, SIGMA_X/SIGMA_T): config defaults.
  - initial RF coefficients: identical given the same seed (model
    construction happens before any mode/coef_lr-dependent branching).
  - initial attention values: identical given the same seed and generator
    (lambda_f=100 constant, lambda_0~U(0,1) draw).
  - collocation points: identical given the same seed (sampled once,
    before model construction).
  - seed=0, budget="short" (300 Adam + 50 L-BFGS), for both modes.

This script does not modify sa_rf/, burgers_rf/, or sapinn/ beyond the
already-made, justified coef_lr parameter addition to sa_rf/train.py.
Writes to NEW, non-destructive output directories
(outputs/sa_rf/lr_sweep/{mode}/lr_{lr}/) -- never touches
outputs/sa_rf/smoke/ or any other existing result.
"""
import json
import sys
from pathlib import Path

BURGERS_DIR = Path(__file__).resolve().parents[1]
if str(BURGERS_DIR) not in sys.path:
    sys.path.insert(0, str(BURGERS_DIR))

from sa_rf import config as sa_rf_config  # noqa: E402
from sa_rf.train import run  # noqa: E402

OUTPUT_ROOT = BURGERS_DIR / "outputs" / "sa_rf" / "lr_sweep"
BUDGET = "short"
SEED = 0
MODES = ("fixed_weight", "adaptive")
COEF_LRS = (0.005, 0.001, 0.0005)


def _lr_tag(lr):
    return str(lr).replace(".", "p")


def main():
    print(f"N_B semantics check: sa_rf.config.N_B_PER_SIDE = {sa_rf_config.N_B_PER_SIDE} "
          f"-- this is PER SIDE (25 points at x=-1 AND 25 points at x=+1), "
          f"i.e. {2 * sa_rf_config.N_B_PER_SIDE} TOTAL boundary points, matching "
          f"SA-PINN's current code value (sapinn.config.N_B_PER_SIDE=25, N_B_TOTAL=50), "
          f"NOT the paper's stated 200 total.")
    print(f"Attention optimizer LR held fixed at config.ADAM_LR={sa_rf_config.ADAM_LR} "
          f"across the entire sweep -- only the RF-coefficient optimizer's LR varies.")
    print()

    results = {}
    for mode in MODES:
        for lr in COEF_LRS:
            output_dir = OUTPUT_ROOT / mode / f"lr_{_lr_tag(lr)}"
            if (output_dir / "config.json").exists():
                raise FileExistsError(
                    f"{output_dir} already has a completed run -- refusing to overwrite."
                )
            print(f"=== mode={mode}, coef_lr={lr} ===")
            metrics_rows, run_config = run(
                mode, BUDGET, seed=SEED, output_dir=output_dir, coef_lr=lr,
            )
            assert run_config["coef_lr"] == lr
            assert run_config["adam_lr"] == sa_rf_config.ADAM_LR  # attention LR unchanged
            results[(mode, lr)] = {"metrics_rows": metrics_rows, "run_config": run_config}

    print()
    print("=" * 110)
    print("SA-RF COEFFICIENT-LEARNING-RATE SWEEP SUMMARY "
          f"(budget={BUDGET}, seed={SEED}, attention LR fixed at {sa_rf_config.ADAM_LR})")
    print("=" * 110)
    for mode in MODES:
        for lr in COEF_LRS:
            rows = results[(mode, lr)]["metrics_rows"]
            run_config = results[(mode, lr)]["run_config"]
            adam_rows = [r for r in rows if r["phase"] == "adam"]
            first_adam = adam_rows[0]
            early_rows = adam_rows[:6]  # first few logged points (epochs 0,10,20,30,40,50 at LOG_EVERY=10)
            last_row = rows[-1]

            print(f"\n--- mode={mode}, coef_lr={lr} ---")
            print(f"  initial: total={first_adam['total_objective']:.6e}  "
                  f"pde={first_adam['weighted_pde_loss']:.4e}  ic={first_adam['weighted_ic_loss']:.4e}  "
                  f"bc={first_adam['bc_mse']:.4e}  "
                  f"(unweighted pde_mse={first_adam['unweighted_pde_mse']:.4e}, "
                  f"ic_mse={first_adam['unweighted_ic_mse']:.4e})")
            print(f"  final ({last_row['phase']}, epoch {last_row['epoch']}): "
                  f"total={last_row['total_objective']:.6e}  "
                  f"pde={last_row['weighted_pde_loss']:.4e}  ic={last_row['weighted_ic_loss']:.4e}  "
                  f"bc={last_row['bc_mse']:.4e}  "
                  f"(unweighted pde_mse={last_row['unweighted_pde_mse']:.4e}, "
                  f"ic_mse={last_row['unweighted_ic_mse']:.4e})")
            print(f"  global relative L2: {first_adam['global_relative_l2_error']:.4e} -> "
                  f"{last_row['global_relative_l2_error']:.4e}")
            print(f"  shock relative L2:  {first_adam['shock_relative_l2_error']:.4e} -> "
                  f"{last_row['shock_relative_l2_error']:.4e}")
            print(f"  RF coefficient grad norm: {first_adam['coef_grad_norm']:.4e} -> "
                  f"{last_row['coef_grad_norm']:.4e}  (coef norm final: {last_row['coef_norm']:.4e})")
            print(f"  lambda_f: init_mean={first_adam['lambda_f_mean']:.4e} -> "
                  f"final_mean={last_row['lambda_f_mean']:.4e} (std final={last_row['lambda_f_std']:.4e}, "
                  f"grad_norm final={last_row['lambda_f_grad_norm']:.4e})")
            print(f"  lambda_0: init_mean={first_adam['lambda_0_mean']:.4e} -> "
                  f"final_mean={last_row['lambda_0_mean']:.4e} (std final={last_row['lambda_0_std']:.4e}, "
                  f"grad_norm final={last_row['lambda_0_grad_norm']:.4e})")
            print(f"  early total-loss trajectory (first logged epochs): "
                  + ", ".join(f"ep{r['epoch']}={r['total_objective']:.3e}" for r in early_rows))
            print(f"  runtime: {last_row['wall_clock_s']:.2f}s, "
                  f"nonfinite anywhere: {any(r['nonfinite'] for r in rows)}, "
                  f"warnings: {run_config['warnings']}")

    summary_path = OUTPUT_ROOT / "lr_sweep_summary.json"
    summary_path.write_text(json.dumps(
        {f"{mode}__lr_{_lr_tag(lr)}": {
            "final_row": results[(mode, lr)]["metrics_rows"][-1],
            "run_config": results[(mode, lr)]["run_config"],
        } for mode in MODES for lr in COEF_LRS},
        indent=2, default=str,
    ))
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
