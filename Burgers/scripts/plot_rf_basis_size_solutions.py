import sys
from pathlib import Path

import torch
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(PROJECT_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from burgers_rf.config import DTYPE  # noqa: E402
from burgers_rf.evaluation import evaluate_error_metrics  # noqa: E402
import sweep_rf_basis_size as sweep  # noqa: E402 -- reused unchanged: reconstruct_model_from_checkpoint,
# OUTPUT_DIR, NX_VALUES. This script does NOT retrain, does NOT modify
# checkpoints, and does NOT touch sweep_rf_basis_size.py's own outputs
# (summary.csv, summary.txt, error_vs_nx.png, checkpoint_nx*.pt).
import test_pde_trained_gaussian_correction as ptgc  # noqa: E402 -- reused unchanged: build_search_baseline
import test_fixed_gaussian_correction as tfgc  # noqa: E402 -- reused unchanged: solution_ylim

# POST-HOC solution-diagnostic plots for the RF basis-size sweep
# (sweep_rf_basis_size.py). NO training, NO optimizer, NO modification of
# any checkpoint -- each Nx's complete checkpoint (explicit Wx/Wt/bx/bt/
# rf_linear_weight tensors, no RNG replay) is loaded and reconstructed via
# sweep.reconstruct_model_from_checkpoint, then evaluated purely via
# forward passes.
#
# EVALUATION TIME: build_search_baseline evaluates at t=1 only (same
# T_SNAPSHOT=1.0 / EVAL_FULL_N=3001 / canonical 200-point shock grid
# convention used by every other post-hoc solution-comparison diagnostic in
# this project), identical across all three Nx values.
#
# METRIC LABELING (diagnostic distinction, not a new canonical metric):
# the canonical multi-time global/shock relative L2 (already saved in each
# checkpoint / summary.csv, computed over N_TEST_TIMES test times) is NOT
# used to label the t=1 plots. Instead, the relative L2 error is computed
# directly from the SAME t=1 arrays the plots show (via
# evaluate_error_metrics on build_search_baseline's own x_eval_full/
# y_eval_full/u_RF_full and x_eval_shock/y_eval_shock/u_RF_shock), so the
# number in each title exactly matches the curve being plotted. Both the
# canonical multi-time numbers and the t=1 numbers are printed side by side
# in the comparison table for transparency -- this script does NOT modify
# summary.csv or summary.txt.

OUTPUT_DIR = sweep.OUTPUT_DIR  # same existing directory; no new path invented


def load_reconstructed_model(nx, device="cpu"):
    checkpoint_path = OUTPUT_DIR / f"checkpoint_nx{nx}.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = sweep.reconstruct_model_from_checkpoint(checkpoint, device=device)
    return model, checkpoint


def compute_t1_metrics(baseline):
    """Relative L2 error computed directly from the SAME t=1 arrays used
    for plotting (via evaluate_error_metrics, reused unmodified) -- NOT the
    canonical multi-time metric stored in the checkpoint."""
    return evaluate_error_metrics(
        baseline["x_eval_full"], baseline["y_eval_full"], baseline["u_RF_full"],
        baseline["y_eval_shock"], baseline["u_RF_shock"],
    )


def plot_solution_full(nx, baseline, t1_global_error):
    x_full = baseline["x_eval_full"]
    y_full = baseline["y_eval_full"]
    u_rf_full = baseline["u_RF_full"]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(x_full, y_full, "k-", linewidth=2, label="u_true")
    ax.plot(x_full, u_rf_full, "--", label="u_RF")
    ax.set_ylim(tfgc.solution_ylim(y_full, u_rf_full))
    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t=1)")
    ax.set_title(f"RF solution, Nx={nx}, t=1\nGlobal relative L2 = {t1_global_error:.6f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"solution_full_nx{nx}.png")
    plt.close(fig)


def plot_solution_shock(nx, baseline, t1_shock_error):
    x_shock = baseline["x_eval_shock"]
    y_shock = baseline["y_eval_shock"]
    u_rf_shock = baseline["u_RF_shock"]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(x_shock, y_shock, "k-", linewidth=2, label="u_true")
    ax.plot(x_shock, u_rf_shock, "--", label="u_RF")
    ax.set_ylim(tfgc.solution_ylim(y_shock, u_rf_shock))
    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t=1)")
    ax.set_title(f"RF shock region, Nx={nx}, t=1\nShock relative L2 = {t1_shock_error:.6f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"solution_shock_nx{nx}.png")
    plt.close(fig)


def print_comparison_table(rows):
    header = ["Nx", "canonical_global", "canonical_shock", "t1_global", "t1_shock"]
    print(" | ".join(f"{h:>18}" for h in header))
    for row in rows:
        vals = [
            str(row["Nx"]),
            f"{row['canonical_global']:.6f}", f"{row['canonical_shock']:.6f}",
            f"{row['t1_global']:.6f}", f"{row['t1_shock']:.6f}",
        ]
        print(" | ".join(f"{v:>18}" for v in vals))


def main():
    # Must be set before any grid is built: sweep_local_width.build_shock_grid
    # calls torch.linspace with no explicit dtype, so it silently follows
    # whatever torch's ambient default dtype is at that moment. Every other
    # script in this project sets this as the first line of main() for
    # exactly this reason -- omitting it here caused a float64/float32
    # cartesian_prod dtype mismatch in build_shock_grid on a fresh
    # interpreter (where no prior code had already set it), and would also
    # have silently downcast the reconstructed model's RF linear weight to
    # float32 via copy_().
    torch.set_default_dtype(DTYPE)
    device = "cpu"
    rows = []

    for nx in sweep.NX_VALUES:
        print(f"=== Nx={nx} ===")
        model, checkpoint = load_reconstructed_model(nx, device)

        baseline = ptgc.build_search_baseline(model, device)
        t1_metrics = compute_t1_metrics(baseline)

        plot_solution_full(nx, baseline, t1_metrics["global_relative_l2_error"])
        plot_solution_shock(nx, baseline, t1_metrics["shock_relative_l2_error"])

        rows.append({
            "Nx": nx,
            "canonical_global": checkpoint["global_relative_l2_error"],
            "canonical_shock": checkpoint["shock_relative_l2_error"],
            "t1_global": t1_metrics["global_relative_l2_error"],
            "t1_shock": t1_metrics["shock_relative_l2_error"],
        })

    print_comparison_table(rows)
    print("[plot_rf_basis_size_solutions] done.")


if __name__ == "__main__":
    main()
