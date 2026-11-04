"""Controlled SA-RF attention-INITIALIZATION sensitivity sweep.

Follows the completed one-seed server comparison (seed=0, 1000 Adam + 100
L-BFGS), which found adaptive SA-RF has much smaller PDE MSE but worse
IC/BC errors than baseline/balanced -- hypothesized to be driven by the
initial lambda_f=100 dominating the IC/BC terms before gradient ascent
can correct for it. Three new "adaptive" runs test this by varying ONLY
the attention initialization:

    A. lambda_f_init=10,  lambda_0_init=None (Uniform(0,1) draw, as usual)
    B. lambda_f_init=1,   lambda_0_init=None (Uniform(0,1) draw, as usual)
    C. lambda_f_init=10,  lambda_0_init=10   (all IC attention weights =10)

All other settings are IDENTICAL to the completed comparison: mode=
"adaptive", seed=0, Nx=159/Nt=19 (3,021 RF coefficients), N_F=10,000,
N_0=100, N_B_PER_SIDE=25 (50 total), coef_lr=0.001, attention LR=
config.ADAM_LR=0.005, budget="controlled" (1000 Adam + 100 L-BFGS),
device=CUDA-if-available (config.get_device()'s default), dtype=float64
(config.DTYPE's default). Attention parameters remain trainable
(gradient ascent) during Adam and frozen during L-BFGS in every variant
-- this is sa_rf.train.run()'s existing, unmodified "adaptive" mode
behavior; only the INITIAL values differ.

RESUMABLE / NON-DESTRUCTIVE: each variant's output directory is checked
for an existing completed run (config.json present) before launching --
already-completed variants are skipped, never overwritten, so re-running
this script after a partial/interrupted sweep continues where it left
off. Writes to a NEW directory (outputs/sa_rf/attention_init_sweep/),
never touching outputs/sa_rf/one_seed_comparison/ or any other existing
result.

NOT RUN by this script's author -- intended to be executed on the server
only (see docs/meetings and the task's explicit "do not launch locally").
"""
import json
import sys
from pathlib import Path

BURGERS_DIR = Path(__file__).resolve().parents[1]
if str(BURGERS_DIR) not in sys.path:
    sys.path.insert(0, str(BURGERS_DIR))

from sa_rf import config as sa_rf_config  # noqa: E402
from sa_rf.train import run  # noqa: E402

OUTPUT_ROOT = BURGERS_DIR / "outputs" / "sa_rf" / "attention_init_sweep"
BUDGET = "controlled"
SEED = 0
COEF_LR = 0.001
MODE = "adaptive"

EXPERIMENTS = [
    ("A_lambda_f10_lambda0_uniform", dict(lambda_f_init=10.0, lambda_0_init=None)),
    ("B_lambda_f1_lambda0_uniform", dict(lambda_f_init=1.0, lambda_0_init=None)),
    ("C_lambda_f10_lambda0_10", dict(lambda_f_init=10.0, lambda_0_init=10.0)),
]


def main():
    print(f"Mode={MODE}, seed={SEED}, coef_lr={COEF_LR}, attention_lr={sa_rf_config.ADAM_LR}, "
          f"budget={BUDGET} -> {sa_rf_config.BUDGETS[BUDGET]}")
    print(f"Nx={sa_rf_config.NX_MATCHED}, Nt={sa_rf_config.NT_MATCHED} "
          f"({sa_rf_config.NX_MATCHED * sa_rf_config.NT_MATCHED} RF coefficients), "
          f"N_F={sa_rf_config.N_F}, N_0={sa_rf_config.N_0}, "
          f"N_B_PER_SIDE={sa_rf_config.N_B_PER_SIDE} ({2 * sa_rf_config.N_B_PER_SIDE} total boundary points)")
    print()

    results = {}
    for name, overrides in EXPERIMENTS:
        output_dir = OUTPUT_ROOT / name
        if (output_dir / "config.json").exists():
            print(f"=== {name}: SKIPPING (already completed at {output_dir}) ===")
            run_config = json.loads((output_dir / "config.json").read_text())
            import csv
            with (output_dir / "metrics.csv").open() as f:
                rows = list(csv.DictReader(f))
            for r in rows:
                for k, v in r.items():
                    if k in ("phase",):
                        continue
                    try:
                        r[k] = float(v)
                    except ValueError:
                        r[k] = float("nan")
            results[name] = {"metrics_rows": rows, "run_config": run_config}
            continue

        print(f"=== {name}: starting (lambda_f_init={overrides['lambda_f_init']}, "
              f"lambda_0_init={overrides['lambda_0_init']}) ===")
        metrics_rows, run_config = run(
            MODE, BUDGET, seed=SEED, output_dir=output_dir, coef_lr=COEF_LR, **overrides,
        )
        assert run_config["lambda_f_init"] == overrides["lambda_f_init"]
        expected_lambda_0 = overrides["lambda_0_init"] if overrides["lambda_0_init"] is not None else "uniform_0_1"
        assert run_config["lambda_0_init"] == expected_lambda_0
        results[name] = {"metrics_rows": metrics_rows, "run_config": run_config}

    print()
    print("=" * 110)
    print(f"SA-RF ATTENTION-INITIALIZATION SWEEP SUMMARY (mode={MODE}, seed={SEED}, budget={BUDGET})")
    print("=" * 110)
    consolidated = {}
    for name, _ in EXPERIMENTS:
        rows = results[name]["metrics_rows"]
        run_config = results[name]["run_config"]
        initial_row = next(r for r in rows if r["phase"] == "initial")
        last_row = rows[-1]

        print(f"\n--- {name} (lambda_f_init={run_config['lambda_f_init']}, "
              f"lambda_0_init={run_config['lambda_0_init']}) ---")
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
        print(f"  weighted PDE loss:   {initial_row['weighted_pde_loss']:.6e} -> "
              f"{last_row['weighted_pde_loss']:.6e}")
        print(f"  weighted IC loss:    {initial_row['weighted_ic_loss']:.6e} -> "
              f"{last_row['weighted_ic_loss']:.6e}")
        print(f"  lambda_f: {initial_row['lambda_f_mean']:.4e} -> {last_row['lambda_f_mean']:.4e} "
              f"(std final={last_row['lambda_f_std']:.4e}, grad_norm final={last_row['lambda_f_grad_norm']:.4e})")
        print(f"  lambda_0: {initial_row['lambda_0_mean']:.4e} -> {last_row['lambda_0_mean']:.4e} "
              f"(std final={last_row['lambda_0_std']:.4e}, grad_norm final={last_row['lambda_0_grad_norm']:.4e})")
        print(f"  runtime: {last_row['wall_clock_s']:.2f}s, "
              f"nonfinite anywhere: {any(r['nonfinite'] for r in rows)}, warnings: {run_config['warnings']}")

        consolidated[name] = {"initial_row": initial_row, "final_row": last_row, "run_config": run_config}

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_ROOT / "attention_init_sweep_summary.json"
    summary_path.write_text(json.dumps(consolidated, indent=2, default=str))
    print(f"\nConsolidated summary written to {summary_path}")


if __name__ == "__main__":
    main()
