import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from burgers_rf.config import (  # noqa: E402
    DTYPE,
    EPOCHS,
    LR,
    M_BD,
    M_INT,
    M_TEST,
    M_TRAIN,
    N_TEST_TIMES,
    NT,
    NV,
    NX,
    SEED,
    SIGMA_T,
    SIGMA_X,
    WEIGHT_DECAY,
    get_device,
    set_seed,
)
from burgers_rf.data import g, make_reference_values, make_test_grid, sample_training_points  # noqa: E402
from burgers_rf.model import BurgersRF  # noqa: E402
from burgers_rf.physics import train  # noqa: E402
from burgers_rf.evaluation import evaluate_error_metrics  # noqa: E402
from sweep_local_width import build_shock_grid  # noqa: E402

# TODO Task 1, step 1: reproduce the seed-42 RF-only PDE-trained Burgers
# baseline (use_local=False) using the EXISTING, unmodified training path
# (physics.train / physics.loss_fn, no new loss/optimizer/sampling), and
# persist a checkpoint + raw residual data for later analysis. This script
# does NOT fit a Gaussian, add a local feature, or interpret/change the
# model based on the residual -- diagnostic only.
#
# Exactly ONE baseline training realization is run (not looped over
# N_TRIALS), so the saved checkpoint/predictions correspond unambiguously
# to a single seed-42 run regardless of any future change to config.N_TRIALS.

DIAG_N_X = 4001
DIAG_TIMES = [0.25, 0.50, 0.75, 1.00]
SHOCK_HALF_WIDTH = 0.02

OUTPUT_DIR = PROJECT_DIR / "outputs" / "pde_trained_rf_residual"


def solution_ylim(*arrays, margin=0.05):
    vals = np.concatenate(arrays)
    lo, hi = float(vals.min()), float(vals.max())
    pad = margin * (hi - lo) if hi > lo else 1.0
    return lo - pad, hi + pad


def plot_solution(t_val, x, u_true, u_rf, mask, suffix, title_suffix):
    xs, ut, ur = x[mask], u_true[mask], u_rf[mask]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ut, "k-", linewidth=2, label="u_true")
    ax.plot(xs, ur, "--", label="u_RF")
    ax.set_ylim(solution_ylim(ut, ur))
    ax.set_xlabel("x")
    ax.set_ylabel(f"u(x, t={t_val})")
    ax.set_title(f"PDE-trained RF baseline: t={t_val}, {title_suffix}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"solution_t{t_val}_{suffix}.png")
    plt.close(fig)


def plot_residual(t_val, x, residual, mask, suffix, title_suffix):
    xs, res = x[mask], residual[mask]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, res, "r-", label="residual = u_true - u_RF")
    ax.axhline(0.0, color="k", linewidth=0.8)
    ax.set_ylim(solution_ylim(res))
    ax.set_xlabel("x")
    ax.set_ylabel("residual")
    ax.set_title(f"PDE-trained RF baseline residual: t={t_val}, {title_suffix}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"residual_t{t_val}_{suffix}.png")
    plt.close(fig)


def residual_extrema(x, residual):
    pos_mask = residual > 0
    neg_mask = residual < 0

    if pos_mask.any():
        idx_pos = int(np.argmax(np.where(pos_mask, residual, -np.inf)))
        max_pos_val, max_pos_x = float(residual[idx_pos]), float(x[idx_pos])
    else:
        max_pos_val, max_pos_x = float("nan"), float("nan")

    if neg_mask.any():
        idx_neg = int(np.argmin(np.where(neg_mask, residual, np.inf)))
        min_neg_val, min_neg_x = float(residual[idx_neg]), float(x[idx_neg])
    else:
        min_neg_val, min_neg_x = float("nan"), float("nan")

    idx_abs = int(np.argmax(np.abs(residual)))
    max_abs_val, max_abs_x = float(residual[idx_abs]), float(x[idx_abs])

    return {
        "max_positive_residual": max_pos_val,
        "max_positive_x": max_pos_x,
        "min_negative_residual": min_neg_val,
        "min_negative_x": min_neg_x,
        "max_abs_residual": max_abs_val,
        "max_abs_x": max_abs_x,
    }


def main():
    torch.set_default_dtype(DTYPE)
    device = get_device()
    print(device)

    # Exactly one baseline training realization, seed=42.
    set_seed(SEED)
    x_train, t_train = sample_training_points(M_TRAIN, device)

    model = BurgersRF(NX, NT, SIGMA_X, SIGMA_T, use_local=False, device=device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    train(model, optimizer, x_train, t_train, M_BD, M_INT, NV, g, device, epochs=EPOCHS)

    # Canonical global/smooth/shock metrics, same grids/convention as every
    # other controlled experiment in this study.
    x_test, t_test = make_test_grid(M_TEST, N_TEST_TIMES, device)
    y_true = make_reference_values(t_test, x_test, NV)
    x_shock, t_shock = build_shock_grid(t_test, device)
    y_true_shock = make_reference_values(t_shock, x_shock, NV)

    y_pred = model(x_test, t_test).cpu().detach().numpy().reshape(-1)
    y_pred_shock = model(x_shock, t_shock).cpu().detach().numpy().reshape(-1)
    metrics = evaluate_error_metrics(x_test, y_true, y_pred, y_true_shock, y_pred_shock)
    print(f"metrics: {metrics}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Checkpoint: model.state_dict() already contains Wx, Wt, bx, bt (fixed
    # RF frequencies/biases) AND model.0.weight (trained c_ij) -- verified
    # directly before implementing this script. Architecture/config values
    # are also stored explicitly so the checkpoint is self-describing and
    # does not silently depend on config.py's state at load time.
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "seed": SEED,
        "use_local": False,
        "NX": NX,
        "NT": NT,
        "SIGMA_X": SIGMA_X,
        "SIGMA_T": SIGMA_T,
        "NV": NV,
        "DTYPE": str(DTYPE),
        "M_TRAIN": M_TRAIN,
        "M_BD": M_BD,
        "M_INT": M_INT,
        "EPOCHS": EPOCHS,
        "LR": LR,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "global_relative_l2_error": metrics["global_relative_l2_error"],
        "smooth_relative_l2_error": metrics["smooth_relative_l2_error"],
        "shock_relative_l2_error": metrics["shock_relative_l2_error"],
    }
    torch.save(checkpoint, OUTPUT_DIR / "checkpoint.pt")

    # Dense diagnostic grid at the four selected times -- separate from the
    # canonical metric grids above, used only for residual visualization.
    x_diag = torch.linspace(-1, 1, DIAG_N_X).reshape(-1, 1).to(device)
    x_diag_np = x_diag.detach().cpu().numpy().reshape(-1)
    shock_mask = np.abs(x_diag_np) <= SHOCK_HALF_WIDTH
    full_mask = np.ones_like(x_diag_np, dtype=bool)

    csv_rows = []
    summary_lines = [
        f"global_relative_l2_error = {metrics['global_relative_l2_error']}",
        f"smooth_relative_l2_error = {metrics['smooth_relative_l2_error']}",
        f"shock_relative_l2_error = {metrics['shock_relative_l2_error']}",
        "",
    ]

    for t_val in DIAG_TIMES:
        t_diag = torch.full_like(x_diag, t_val)
        y_true_diag = make_reference_values(t_diag, x_diag, NV)

        _, u_rf_diag, _ = model.forward_components(x_diag, t_diag)
        u_rf_diag_np = u_rf_diag.detach().cpu().numpy().reshape(-1)

        residual = y_true_diag - u_rf_diag_np

        for xv, uv_true, uv_rf, rv in zip(x_diag_np, y_true_diag, u_rf_diag_np, residual):
            csv_rows.append((t_val, xv, uv_true, uv_rf, rv))

        plot_solution(t_val, x_diag_np, y_true_diag, u_rf_diag_np, full_mask, "full", "full domain")
        plot_solution(t_val, x_diag_np, y_true_diag, u_rf_diag_np, shock_mask, "shock", f"shock zoom |x|<={SHOCK_HALF_WIDTH}")
        plot_residual(t_val, x_diag_np, residual, full_mask, "full", "full domain")
        plot_residual(t_val, x_diag_np, residual, shock_mask, "shock", f"shock zoom |x|<={SHOCK_HALF_WIDTH}")

        extrema = residual_extrema(x_diag_np, residual)
        summary_lines.append(f"t = {t_val}")
        for k, v in extrema.items():
            summary_lines.append(f"  {k} = {v}")
        summary_lines.append("")

    csv_path = OUTPUT_DIR / "residual_grid.csv"
    with csv_path.open("w") as f:
        f.write("t,x,u_true,u_rf,residual\n")
        for row in csv_rows:
            f.write(",".join(str(v) for v in row) + "\n")

    (OUTPUT_DIR / "baseline_summary.txt").write_text("\n".join(summary_lines) + "\n")
    print("\n".join(summary_lines))
    print("[test_pde_trained_rf_baseline] done.")


if __name__ == "__main__":
    main()
