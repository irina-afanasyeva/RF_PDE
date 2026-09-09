"""
Diagnostic-only script for the analytic Burgers reference solution.

Does not touch model/training code: only imports u_true (via
make_reference_values) and NV from the existing Burgers package. No
training is run.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from burgers_rf.config import NV  # noqa: E402
from burgers_rf.data import make_reference_values  # noqa: E402

N_X_FINE = 1000
TIMES = [0.5, 0.75, 1.0]
TOP_K = 20  # how many steepest points to report per time

OUTPUT_DIR = PROJECT_DIR / "outputs" / "diagnostics"


def build_fine_grid(n_x, times):
    """Same cartesian-product convention as data.make_test_grid, but with
    an arbitrary list of times instead of a linspace."""
    x_fine = torch.linspace(-1, 1, n_x)
    t_sel = torch.tensor(times, dtype=x_fine.dtype)
    x_t_product = torch.cartesian_prod(t_sel, x_fine)
    x_grid = x_t_product[:, [1]]
    t_grid = x_t_product[:, [0]]
    return x_grid, t_grid, x_fine


def main():
    x_grid, t_grid, x_fine = build_fine_grid(N_X_FINE, TIMES)

    y_true = make_reference_values(t_grid, x_grid, NV)
    y_true = y_true.reshape(len(TIMES), N_X_FINE)
    x_fine_np = x_fine.numpy()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, (ax_u, ax_grad) = plt.subplots(1, 2, figsize=(12, 5))

    summary_lines = []
    for i, t in enumerate(TIMES):
        u = y_true[i]
        du_dx = np.gradient(u, x_fine_np)
        abs_du_dx = np.abs(du_dx)

        ax_u.plot(x_fine_np, u, label=f"t={t}")
        ax_grad.plot(x_fine_np, abs_du_dx, label=f"t={t}")

        peak_idx = np.argmax(abs_du_dx)
        peak_x = x_fine_np[peak_idx]
        peak_val = abs_du_dx[peak_idx]

        top_idx = np.argsort(abs_du_dx)[::-1][:TOP_K]
        top_idx_sorted = top_idx[np.argsort(x_fine_np[top_idx])]

        summary_lines.append(f"t = {t}")
        summary_lines.append(f"  argmax |du/dx| at x = {peak_x:.6f}, value = {peak_val:.6f}")
        summary_lines.append(f"  top-{TOP_K} steepest points (sorted by x):")
        for idx in top_idx_sorted:
            summary_lines.append(
                f"    x = {x_fine_np[idx]: .6f}   u = {u[idx]: .6f}   |du/dx| = {abs_du_dx[idx]:.6f}"
            )
        summary_lines.append("")

        data_path = OUTPUT_DIR / f"reference_gradient_t{t}.csv"
        with data_path.open("w") as f:
            f.write("x,u_true,abs_du_dx\n")
            for xv, uv, gv in zip(x_fine_np, u, abs_du_dx):
                f.write(f"{xv},{uv},{gv}\n")

    ax_u.set_xlabel("x")
    ax_u.set_ylabel("u_true(x, t)")
    ax_u.set_title("Reference solution")
    ax_u.legend()

    ax_grad.set_xlabel("x")
    ax_grad.set_ylabel("|du/dx| (finite-difference estimate)")
    ax_grad.set_title("Gradient magnitude")
    ax_grad.legend()

    fig.tight_layout()
    plot_path = OUTPUT_DIR / "reference_gradient_diagnostic.png"
    fig.savefig(plot_path)

    summary_path = OUTPUT_DIR / "reference_gradient_diagnostic.txt"
    summary_text = "\n".join(summary_lines)
    summary_path.write_text(summary_text)

    print(summary_text)
    print(f"Saved plot to {plot_path}")
    print(f"Saved per-time data (full x-grid) to {OUTPUT_DIR}/reference_gradient_t<t>.csv")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
