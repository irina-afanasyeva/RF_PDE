import copy
import gc
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(PROJECT_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from burgers_rf.config import (  # noqa: E402
    DTYPE, EPOCHS, LR, M_BD, M_INT, M_TEST, M_TRAIN, N_TEST_TIMES, NV,
    SEED, SIGMA_T, SIGMA_X, WEIGHT_DECAY, get_device, set_seed,
)
from burgers_rf.data import g, make_reference_values, make_test_grid, sample_training_points  # noqa: E402
from burgers_rf.model import BurgersRF  # noqa: E402
from burgers_rf.physics import boundary_residual, burgers_residual, initial_residual, train  # noqa: E402
from burgers_rf.evaluation import evaluate_error_metrics  # noqa: E402
from sweep_local_width import build_shock_grid  # noqa: E402
import sweep_rf_basis_size as sweep  # noqa: E402 -- reused unchanged: build_complete_checkpoint
import test_pde_trained_gaussian_correction as ptgc  # noqa: E402 -- reused unchanged: build_search_baseline
import test_fixed_gaussian_correction as tfgc  # noqa: E402 -- reused unchanged: solution_ylim

# EXPERIMENT 2 -- fixed Nx/Nt-ratio sweep (2026-10-21 advisor meeting prep,
# replaces the single-case test_rf_nt20.py). Question: should Nt increase
# together with Nx? Preserves the canonical ratio Nx/Nt=50 exactly across a
# small, cheap-to-expensive range of total coefficient counts. BurgersRF
# construction and physics.train()/loss_fn are reused completely
# unmodified; this is a direct generalization of sweep_rf_basis_size.py's
# own run_one_nx to vary (Nx,Nt) jointly instead of Nx alone, plus the
# fuller metric set (IC/PDE/BC loss, t=1 metrics) already established in
# test_rf_nt20.py/sweep_rf_random_seeds.py.
#
# METHODOLOGY / RNG NOTE (same as sweep_rf_basis_size.py, deliberately NOT
# changed): for EVERY config, set_seed(SEED) -> sample_training_points(...)
# is called immediately before BurgersRF(...) construction. Because this
# reset happens fresh at the start of every iteration, each config's
# collocation points (x_train, t_train) are drawn from the IDENTICAL
# RNG position regardless of what any other config in the sweep did before
# or after it -- in particular, the canonical (500,10) case's trajectory is
# NOT affected by its position in the sweep order. Collocation points are
# therefore BIT-IDENTICAL across all six configs. However, BurgersRF.__init__
# draws an (Nx,Nt)-dependent number of random values for Wx/Wt/bx/bt, so the
# RNG stream position immediately after construction differs across
# configs -- the per-epoch boundary_residual/initial_residual draws inside
# physics.train() therefore DIFFER across configs from the very first
# training epoch onward. This is the same unavoidable, already-documented
# consequence described in sweep_rf_basis_size.py's own RNG note; no new
# sampling methodology is introduced here.
#
# Configurations run smallest-to-largest so a partial run (meeting
# interruption) still leaves useful completed results. summary.csv is
# rewritten after EVERY completed configuration, not only at the end.
#
# EXTENSION (2026-10-21, same session): added 7 larger configurations,
# same Nx/Nt=50 ratio, same methodology -- no scientific/architecture
# change. The script now RESUMES from an existing outputs/summary.csv:
# already-completed configs (by exact (Nx,Nt) match) are detected and
# skipped, never retrained; their rows are carried forward unchanged.

CONFIGS = [
    (50, 1), (100, 2), (250, 5), (500, 10), (750, 15), (1000, 20),
    (1250, 25), (1500, 30), (2000, 40), (2500, 50), (3000, 60), (3500, 70), (4000, 80),
]
CANONICAL_CONFIG = (500, 10)

CANONICAL_GLOBAL = 0.27149430095802507
CANONICAL_SMOOTH = 0.2443827084159696
CANONICAL_SHOCK = 0.8390487998889169
VERIFY_RTOL = 1e-4

IC_GRID_N_X = 1000
OUTPUT_DIR = PROJECT_DIR / "outputs" / "rf_nx_nt_fixed_ratio_sweep"

CSV_HEADER = [
    "Nx", "Nt", "n_coefficients",
    "global_relative_l2_error", "smooth_relative_l2_error", "shock_relative_l2_error",
    "ic_relative_l2_error", "final_pde_loss", "final_ic_loss", "final_bc_loss",
    "t1_global_relative_l2_error", "t1_smooth_relative_l2_error", "t1_shock_relative_l2_error",
]


def relative_l2(y_true, y_pred):
    return float(np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true))


def run_one_config(nx, nt, device):
    """IDENTICAL procedure to sweep_rf_basis_size.py's run_one_nx, generalized
    to vary (Nx,Nt) jointly. set_seed(SEED) is called fresh here, exactly as
    in that script, so this config's trajectory is independent of sweep
    order (see module docstring)."""
    set_seed(SEED)
    x_train, t_train = sample_training_points(M_TRAIN, device)

    print(f"=== Nx={nx}, Nt={nt} (Nx*Nt={nx * nt}) ===")
    model = BurgersRF(nx, nt, SIGMA_X, SIGMA_T, use_local=False, device=device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    train(model, optimizer, x_train, t_train, M_BD, M_INT, NV, g, device, epochs=EPOCHS)

    residual = burgers_residual(model, x_train, t_train, NV)
    final_pde_loss = torch.mean(residual**2).item()
    residual_bd = boundary_residual(model, M_BD, device)
    final_bc_loss = torch.mean(residual_bd**2).item()
    residual_int = initial_residual(model, M_INT, g, device)
    final_ic_loss = torch.mean(residual_int**2).item()

    x_test, t_test = make_test_grid(M_TEST, N_TEST_TIMES, device)
    y_true = make_reference_values(t_test, x_test, NV)
    x_shock, t_shock = build_shock_grid(t_test, device)
    y_true_shock = make_reference_values(t_shock, x_shock, NV)
    y_pred = model(x_test, t_test).cpu().detach().numpy().reshape(-1)
    y_pred_shock = model(x_shock, t_shock).cpu().detach().numpy().reshape(-1)
    canonical_metrics = evaluate_error_metrics(x_test, y_true, y_pred, y_true_shock, y_pred_shock)

    x_ic = torch.linspace(-1, 1, IC_GRID_N_X).reshape(-1, 1).to(device)
    t_ic = torch.zeros_like(x_ic).to(device)
    y_ic_true = g(x_ic).detach().cpu().numpy().reshape(-1)
    y_ic_pred = model(x_ic, t_ic).cpu().detach().numpy().reshape(-1)
    ic_relative_l2 = relative_l2(y_ic_true, y_ic_pred)

    # t=1-only metrics via an independent CPU copy (established device-safe
    # pattern) -- the live `model` is never modified by this, and this
    # happens AFTER training is complete, so it cannot delay/affect it.
    cpu_model = copy.deepcopy(model).to("cpu")
    for attr_name in ("Wx", "Wt", "bx", "bt"):
        cpu_value = getattr(cpu_model, attr_name).detach().cpu().clone()
        delattr(cpu_model, attr_name)
        setattr(cpu_model, attr_name, cpu_value)
    baseline = ptgc.build_search_baseline(cpu_model, "cpu")
    t1_metrics = evaluate_error_metrics(
        baseline["x_eval_full"], baseline["y_eval_full"], baseline["u_RF_full"],
        baseline["y_eval_shock"], baseline["u_RF_shock"],
    )

    metadata = {
        "Nt": nt, "SIGMA_X": SIGMA_X, "SIGMA_T": SIGMA_T, "NV": NV, "DTYPE": str(DTYPE),
        "seed": SEED, "M_TRAIN": M_TRAIN, "M_BD": M_BD, "M_INT": M_INT,
        "EPOCHS": EPOCHS, "LR": LR, "WEIGHT_DECAY": WEIGHT_DECAY,
        "global_relative_l2_error": canonical_metrics["global_relative_l2_error"],
        "smooth_relative_l2_error": canonical_metrics["smooth_relative_l2_error"],
        "shock_relative_l2_error": canonical_metrics["shock_relative_l2_error"],
        "ic_relative_l2_error": ic_relative_l2,
        "final_pde_loss": final_pde_loss, "final_ic_loss": final_ic_loss, "final_bc_loss": final_bc_loss,
    }
    checkpoint = sweep.build_complete_checkpoint(model, nx, metadata)

    row = {
        "Nx": nx, "Nt": nt, "n_coefficients": nx * nt,
        "global_relative_l2_error": canonical_metrics["global_relative_l2_error"],
        "smooth_relative_l2_error": canonical_metrics["smooth_relative_l2_error"],
        "shock_relative_l2_error": canonical_metrics["shock_relative_l2_error"],
        "ic_relative_l2_error": ic_relative_l2,
        "final_pde_loss": final_pde_loss, "final_ic_loss": final_ic_loss, "final_bc_loss": final_bc_loss,
        "t1_global_relative_l2_error": t1_metrics["global_relative_l2_error"],
        "t1_smooth_relative_l2_error": t1_metrics["smooth_relative_l2_error"],
        "t1_shock_relative_l2_error": t1_metrics["shock_relative_l2_error"],
    }
    return row, checkpoint, baseline


def verify_reproduction_gate(row):
    def rel_diff(a, b):
        return abs(a - b) / abs(b)

    checks = {
        "global": rel_diff(row["global_relative_l2_error"], CANONICAL_GLOBAL),
        "smooth": rel_diff(row["smooth_relative_l2_error"], CANONICAL_SMOOTH),
        "shock": rel_diff(row["shock_relative_l2_error"], CANONICAL_SHOCK),
    }
    print("(Nx,Nt)=(500,10) reproduction gate (vs. canonical RF-only baseline):")
    for k, v in checks.items():
        print(f"  {k}: relative difference = {v:.3e} (tolerance {VERIFY_RTOL:.1e})")
    failed = {k: v for k, v in checks.items() if v > VERIFY_RTOL}
    if failed:
        raise RuntimeError(
            f"(500,10) REPRODUCTION GATE FAILED: {failed}. Refusing to continue "
            "interpreting the rest of the Nx/Nt sweep."
        )
    print("Reproduction gate PASSED.")


_CSV_INT_COLUMNS = {"Nx", "Nt", "n_coefficients"}


def load_existing_results():
    """Parse an existing summary.csv (same format write_csv produces) into
    the same row-dict structure used throughout this script. Returns []
    if no summary.csv exists yet (fresh run). Used ONLY to detect already-
    completed configurations for resume -- never retrains or modifies
    them."""
    path = OUTPUT_DIR / "summary.csv"
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        header = f.readline().strip().split(",")
        if header != CSV_HEADER:
            raise RuntimeError(
                f"Existing {path} header {header} does not match the expected "
                f"{CSV_HEADER}. Refusing to resume from a mismatched summary file."
            )
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            row = {
                k: (int(v) if k in _CSV_INT_COLUMNS else float(v))
                for k, v in zip(CSV_HEADER, parts)
            }
            rows.append(row)
    return rows


def write_csv(rows):
    path = OUTPUT_DIR / "summary.csv"
    with path.open("w") as f:
        f.write(",".join(CSV_HEADER) + "\n")
        for row in rows:
            f.write(",".join(str(row[k]) for k in CSV_HEADER) + "\n")


def write_summary(rows):
    lines = [
        "Fixed Nx/Nt-ratio sweep (Nx/Nt=50, canonical ratio), seed=42, epochs=1000.",
        f"Configurations: {CONFIGS}",
        "",
        "RNG note: set_seed(SEED) -> sample_training_points(...) is called fresh "
        "immediately before BurgersRF(...) construction for EVERY config, so "
        "collocation points (x_train,t_train) are BIT-IDENTICAL across all six "
        "configs and independent of sweep order. BurgersRF.__init__ draws an "
        "(Nx,Nt)-dependent number of random values for Wx/Wt/bx/bt, so the "
        "per-epoch boundary/initial residual draws inside physics.train() DIFFER "
        "across configs from the first training epoch onward. No new sampling "
        "methodology was introduced -- this is the same behavior as "
        "sweep_rf_basis_size.py.",
        "",
    ]
    for row in rows:
        lines.append(f"--- Nx={row['Nx']}, Nt={row['Nt']} (coeff={row['n_coefficients']}) ---")
        for k in CSV_HEADER[3:]:
            lines.append(f"  {k} = {row[k]}")
    (OUTPUT_DIR / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def plot_error_vs_coefficient_count(rows):
    coeffs = [row["n_coefficients"] for row in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(coeffs, [r["global_relative_l2_error"] for r in rows], "o-", label="global relative L2")
    ax.plot(coeffs, [r["shock_relative_l2_error"] for r in rows], "s-", label="shock relative L2")
    ax.set_xlabel("Nx*Nt (trainable coefficients)")
    ax.set_ylabel("relative L2 error")
    ax.set_xscale("log")
    ax.set_title("RF-only error vs. coefficient count (fixed ratio Nx/Nt=50)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "error_vs_coefficient_count.png")
    plt.close(fig)


def plot_error_vs_nx(rows):
    nx_values = [row["Nx"] for row in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(nx_values, [r["global_relative_l2_error"] for r in rows], "o-", label="global relative L2")
    ax.plot(nx_values, [r["shock_relative_l2_error"] for r in rows], "s-", label="shock relative L2")
    ax.set_xlabel("Nx (Nt = Nx/50)")
    ax.set_ylabel("relative L2 error")
    ax.set_xscale("log")
    ax.set_title("RF-only error vs. Nx, fixed ratio Nx/Nt=50")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "error_vs_nx.png")
    plt.close(fig)


def plot_solution(nx, nt, baseline, suffix, key_x, key_y, key_u, metric_value, metric_name):
    x = baseline[key_x]
    y = baseline[key_y]
    u = baseline[key_u]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(x, y, "k-", linewidth=2, label="u_true")
    ax.plot(x, u, "--", label="u_RF")
    ax.set_ylim(tfgc.solution_ylim(y, u))
    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t=1)")
    ax.set_title(f"RF solution, Nx={nx}, Nt={nt}, t=1\n{metric_name} = {metric_value:.6f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"solution_{suffix}_nx{nx}_nt{nt}.png")
    plt.close(fig)


def print_table(rows):
    header = ["Nx", "Nt", "coeff", "global", "smooth", "shock", "IC", "pde_loss", "t1_global", "t1_smooth", "t1_shock"]
    print(" | ".join(f"{h:>11}" for h in header))
    for row in rows:
        vals = [
            str(row["Nx"]), str(row["Nt"]), str(row["n_coefficients"]),
            f"{row['global_relative_l2_error']:.6f}", f"{row['smooth_relative_l2_error']:.6f}",
            f"{row['shock_relative_l2_error']:.6f}", f"{row['ic_relative_l2_error']:.6f}",
            f"{row['final_pde_loss']:.6e}",
            f"{row['t1_global_relative_l2_error']:.6f}", f"{row['t1_smooth_relative_l2_error']:.6f}",
            f"{row['t1_shock_relative_l2_error']:.6f}",
        ]
        print(" | ".join(f"{v:>11}" for v in vals))


def _is_oom_error(exc):
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def main():
    torch.set_default_dtype(DTYPE)
    device = get_device()
    print(device)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # RESUME: load already-completed configurations from the existing
    # summary.csv (if any) -- their results are carried forward EXACTLY as
    # saved and are NEVER retrained.
    rows = load_existing_results()
    completed_configs = {(row["Nx"], row["Nt"]) for row in rows}
    if rows:
        print(f"Resuming: {len(rows)} already-completed configuration(s) found: "
              f"{sorted(completed_configs)}")
        for row in rows:
            missing = not (OUTPUT_DIR / f"checkpoint_nx{row['Nx']}_nt{row['Nt']}.pt").exists()
            if missing:
                print(f"  WARNING: summary.csv has a row for Nx={row['Nx']},Nt={row['Nt']} "
                      "but its checkpoint file is missing on this machine.")
        canonical_row = next((r for r in rows if (r["Nx"], r["Nt"]) == CANONICAL_CONFIG), None)
        if canonical_row is not None:
            verify_reproduction_gate(canonical_row)  # integrity check on the carried-forward result

    for nx, nt in CONFIGS:
        if (nx, nt) in completed_configs:
            print(f"Skipping already-completed config Nx={nx}, Nt={nt} (found in existing summary.csv).")
            continue

        relative_size = (nx * nt) / (CANONICAL_CONFIG[0] * CANONICAL_CONFIG[1])
        print(f"=== starting Nx={nx}, Nt={nt} (Nx*Nt={nx * nt}), "
              f"{relative_size:.2f}x the canonical 500x10 coefficient count ===")

        try:
            row, checkpoint, baseline = run_one_config(nx, nt, device)
        except RuntimeError as exc:
            if _is_oom_error(exc):
                print(f"CUDA OUT OF MEMORY at Nx={nx}, Nt={nt}: {exc}")
                print("Stopping the sweep here -- NOT proceeding to larger configurations. "
                      "All earlier results remain in summary.csv/summary.txt and their "
                      "checkpoints; scientific settings were not altered.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                break
            raise

        if (nx, nt) == CANONICAL_CONFIG:
            verify_reproduction_gate(row)

        torch.save(checkpoint, OUTPUT_DIR / f"checkpoint_nx{nx}_nt{nt}.pt")
        rows.append(row)
        write_csv(rows)  # rewritten after EVERY completed config

        plot_solution(nx, nt, baseline, "full", "x_eval_full", "y_eval_full", "u_RF_full",
                      row["t1_global_relative_l2_error"], "t=1 global relative L2")
        plot_solution(nx, nt, baseline, "shock", "x_eval_shock", "y_eval_shock", "u_RF_shock",
                      row["t1_shock_relative_l2_error"], "t=1 shock relative L2")

        # Explicit cleanup between configurations: run_one_config's model/
        # optimizer/training tensors are function-local and already out of
        # scope at this point; this just returns the now-free CUDA blocks
        # to the allocator's pool before the next (larger) config runs.
        del row, checkpoint, baseline
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_summary(rows)
    plot_error_vs_coefficient_count(rows)
    plot_error_vs_nx(rows)
    print_table(rows)

    print("[sweep_rf_nx_nt_fixed_ratio] done.")


if __name__ == "__main__":
    main()
