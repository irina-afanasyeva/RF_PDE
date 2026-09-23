import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from burgers_rf.config import (  # noqa: E402
    DTYPE,
    LOCAL_CENTER,
    M_TRAIN,
    NT,
    NV,
    NX,
    SEED,
    SIGMA_T,
    SIGMA_X,
    set_seed,
)
from burgers_rf.data import make_reference_values, sample_training_points  # noqa: E402
from burgers_rf.model import BurgersRF  # noqa: E402
from burgers_rf.local_features import GaussianLocal  # noqa: E402
from sweep_local_width import build_shock_grid  # noqa: E402

# Direct spatial least-squares diagnostic: removes PDE training, Adam, IC/BC
# penalties, and autograd entirely. Given the EXACT seed=42 spatial RF
# realization (Wx, bx) used by every controlled Burgers experiment, fit a
# single fixed-time snapshot U(x, t=1) directly via linear algebra, with and
# without one Gaussian local column, and compare the coefficient the fit
# itself prefers against what PDE training actually learns (near-zero).
#
# Two fitting formulations are both preserved:
#   Experiment A: the professor's whiteboard minimum-norm formulation
#       minimize ||theta||_2  subject to  A theta = y
#     with M_fit < Nx=500 (underdetermined).
#   Experiment B: dense overdetermined least squares, M_fit=4001 >> Nx.
# Both are solved via numpy.linalg.lstsq (SVD-based), never an explicit
# matrix inverse -- this single routine correctly reduces to the min-norm
# solution when the system is underdetermined-consistent, and to ordinary
# least squares when overdetermined.
#
# TARGET / REFERENCE SOLUTION: this repository contains only ONE reference
# solution source -- the Cole-Hopf / Gauss-Hermite quadrature evaluation
# (burgers_rf.data.u_true / make_reference_values). There is no separate
# numerical PDE solver anywhere in this codebase. We do not invent one:
# "true" and "numerical" solution are the same source here, and that is
# stated explicitly in the script's own output.

T_SNAPSHOT = 1.0
SIGMAS_LOCAL = [0.0025, 0.02, 0.20]  # local Gaussian widths under test (NOT SIGMA_X/SIGMA_T)

# Experiment A: M_fit deliberately chosen well below Nx=500 (50%), leaving a
# >=250-dimensional null space for the RF-only system so the minimum-norm
# solution is a genuine interpolation regime, not a degenerate few-point fit.
M_FIT_A = 250

# Experiment B: dense, deterministic, overdetermined (~8x oversampling of Nx).
M_FIT_B = 4001

# Full-domain evaluation grid: deliberately a different point count from
# both fitting grids, so evaluation never coincides with fitting points.
EVAL_FULL_N = 3001

OUTPUT_DIR = PROJECT_DIR / "outputs" / "direct_spatial_fit"

EXPERIMENTS = ["experiment_A_min_norm", "experiment_B_dense_ls"]


def get_rf_params(device):
    """Reproduce the EXACT seed=42 spatial RF realization used by every
    controlled Burgers experiment: sample_training_points must be called
    (and its output discarded) before model construction, because it
    advances the RNG to the exact position every real experiment relies on."""
    set_seed(SEED)
    sample_training_points(M_TRAIN, device)  # burn RNG exactly as every real experiment does
    model = BurgersRF(NX, NT, SIGMA_X, SIGMA_T, device=device)
    return model.Wx.detach(), model.bx.detach()


def phi_x_matrix(x_np, Wx_t, bx_t):
    x_t = torch.tensor(x_np.reshape(-1, 1), dtype=DTYPE)
    phi = torch.cos(x_t @ Wx_t + bx_t)
    return phi.detach().numpy()


def gaussian_column(x_np, center, sigma_local):
    x_t = torch.tensor(x_np.reshape(-1, 1), dtype=DTYPE)
    feature = GaussianLocal(center=center, width=sigma_local)
    return feature(x_t).detach().numpy().reshape(-1)


def reference_values(x_np, t_value):
    x_t = torch.tensor(x_np.reshape(-1, 1), dtype=DTYPE)
    t_t = torch.full_like(x_t, t_value)
    return make_reference_values(t_t, x_t, NV)


def relative_l2(y_true, y_pred):
    return float(np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true))


def lstsq_solve(A, y):
    """SVD-based least-squares/pseudoinverse solve. For an underdetermined
    consistent system this returns the minimum-norm solution; for an
    overdetermined system, ordinary least squares. Never forms an inverse."""
    theta, _residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)
    return theta, rank, s


def fit_and_evaluate(experiment_name, x_fit, sigma_local, Wx_t, bx_t, x_eval_full, x_eval_shock, y_eval_full, y_eval_shock):
    y_fit = reference_values(x_fit, T_SNAPSHOT)

    A_fit_RF = phi_x_matrix(x_fit, Wx_t, bx_t)
    g_fit = gaussian_column(x_fit, LOCAL_CENTER, sigma_local)
    A_fit_aug = np.hstack([A_fit_RF, g_fit.reshape(-1, 1)])

    c_rf, rank_rf, s_rf = lstsq_solve(A_fit_RF, y_fit)
    theta_aug, rank_aug, s_aug = lstsq_solve(A_fit_aug, y_fit)
    c_aug = theta_aug[:-1]
    d = float(theta_aug[-1])

    A_eval_full_RF = phi_x_matrix(x_eval_full, Wx_t, bx_t)
    A_eval_shock_RF = phi_x_matrix(x_eval_shock, Wx_t, bx_t)
    g_eval_full = gaussian_column(x_eval_full, LOCAL_CENTER, sigma_local)
    g_eval_shock = gaussian_column(x_eval_shock, LOCAL_CENTER, sigma_local)

    y_pred_full_rf = A_eval_full_RF @ c_rf
    y_pred_shock_rf = A_eval_shock_RF @ c_rf

    rf_part_full = A_eval_full_RF @ c_aug
    rf_part_shock = A_eval_shock_RF @ c_aug
    local_full = d * g_eval_full
    local_shock = d * g_eval_shock
    y_pred_full_aug = rf_part_full + local_full
    y_pred_shock_aug = rf_part_shock + local_shock

    E_RF_full = relative_l2(y_eval_full, y_pred_full_rf)
    E_RF_shock = relative_l2(y_eval_shock, y_pred_shock_rf)
    E_aug_full = relative_l2(y_eval_full, y_pred_full_aug)
    E_aug_shock = relative_l2(y_eval_shock, y_pred_shock_aug)

    # ratio_full/ratio_shock use the RF PART OF THE AUGMENTED FIT as the
    # denominator (matching burgers_rf.diagnostics' convention of comparing
    # against the same model's own RF component, not a separately-fit model).
    ratio_full = float(np.linalg.norm(local_full) / np.linalg.norm(rf_part_full))
    ratio_shock = float(np.linalg.norm(local_shock) / np.linalg.norm(rf_part_shock))

    delta_global = E_RF_full - E_aug_full
    delta_shock = E_RF_shock - E_aug_shock
    pct_global = 100 * delta_global / E_RF_full if E_RF_full != 0 else float("nan")
    pct_shock = 100 * delta_shock / E_RF_shock if E_RF_shock != 0 else float("nan")

    cond_rf = float(s_rf[0] / s_rf[rank_rf - 1]) if rank_rf > 0 else float("nan")
    cond_aug = float(s_aug[0] / s_aug[rank_aug - 1]) if rank_aug > 0 else float("nan")

    result = {
        "experiment": experiment_name,
        "sigma_local": sigma_local,
        "M_fit": len(x_fit),
        "E_RF_full": E_RF_full,
        "E_RF_shock": E_RF_shock,
        "norm_c_RF": float(np.linalg.norm(c_rf)),
        "E_aug_full": E_aug_full,
        "E_aug_shock": E_aug_shock,
        "norm_c_aug": float(np.linalg.norm(c_aug)),
        "d": d,
        "norm_dG": float(np.linalg.norm(local_full)),
        "ratio_full": ratio_full,
        "ratio_shock": ratio_shock,
        "delta_global": delta_global,
        "delta_shock": delta_shock,
        "pct_global": pct_global,
        "pct_shock": pct_shock,
        "rank_A_RF": int(rank_rf),
        "rank_A_aug": int(rank_aug),
        "largest_sv_RF": float(s_rf[0]),
        "smallest_retained_sv_RF": float(s_rf[rank_rf - 1]) if rank_rf > 0 else float("nan"),
        "cond_RF": cond_rf,
        "largest_sv_aug": float(s_aug[0]),
        "smallest_retained_sv_aug": float(s_aug[rank_aug - 1]) if rank_aug > 0 else float("nan"),
        "cond_aug": cond_aug,
    }

    plot_data = {
        "x_eval_full": x_eval_full,
        "y_eval_full": y_eval_full,
        "y_pred_full_rf": y_pred_full_rf,
        "rf_part_full": rf_part_full,
        "local_full": local_full,
        "y_pred_full_aug": y_pred_full_aug,
        "x_eval_shock": x_eval_shock,
        "y_eval_shock": y_eval_shock,
        "y_pred_shock_rf": y_pred_shock_rf,
        "rf_part_shock": rf_part_shock,
        "local_shock": local_shock,
        "y_pred_shock_aug": y_pred_shock_aug,
    }
    return result, plot_data


SUMMARY_HEADER = [
    "sigma_local", "M_fit",
    "E_RF_full", "E_RF_shock", "norm_c_RF",
    "E_aug_full", "E_aug_shock", "norm_c_aug", "d", "norm_dG",
    "ratio_full", "ratio_shock",
    "delta_global", "delta_shock", "pct_global", "pct_shock",
    "rank_A_RF", "rank_A_aug",
    "largest_sv_RF", "smallest_retained_sv_RF", "cond_RF",
    "largest_sv_aug", "smallest_retained_sv_aug", "cond_aug",
]


def write_summary(experiment_name, rows):
    csv_path = OUTPUT_DIR / f"{experiment_name}_summary.csv"
    with csv_path.open("w") as f:
        f.write(",".join(SUMMARY_HEADER) + "\n")
        for row in rows:
            f.write(",".join(str(row[key]) for key in SUMMARY_HEADER) + "\n")

    txt_path = OUTPUT_DIR / f"{experiment_name}_summary.txt"
    with txt_path.open("w") as f:
        for row in rows:
            f.write(f"sigma_local = {row['sigma_local']}\n")
            for key in SUMMARY_HEADER[1:]:
                f.write(f"  {key} = {row[key]}\n")
            f.write("\n")


def shared_ylim(plot_data_by_sigma, keys):
    vals = []
    for sigma_local in SIGMAS_LOCAL:
        pd = plot_data_by_sigma[sigma_local]
        for key in keys:
            vals.append(pd[key])
    all_vals = np.concatenate(vals)
    return float(all_vals.min()), float(all_vals.max())


def plot_full_domain(experiment_name, sigma_local, pd, ylim):
    fig, ax = plt.subplots(figsize=(9, 6))
    x = pd["x_eval_full"]
    ax.plot(x, pd["y_eval_full"], label="reference (Cole-Hopf / Gauss-Hermite)", linewidth=2, color="k")
    ax.plot(x, pd["y_pred_full_rf"], label="RF-only fit", linestyle="--")
    ax.plot(x, pd["rf_part_full"], label="RF part of augmented fit", linestyle="-.")
    ax.plot(x, pd["local_full"], label="local contribution d*G", linestyle=":")
    ax.plot(x, pd["y_pred_full_aug"], label="RF+Gaussian total", linestyle="-")
    ax.set_xlabel("x")
    ax.set_ylabel(f"u(x, t={T_SNAPSHOT})")
    ax.set_ylim(ylim)
    ax.set_title(f"{experiment_name}: sigma_local={sigma_local}, full domain")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{experiment_name}_sigma{sigma_local}_full.png")
    plt.close(fig)


def plot_shock_zoom(experiment_name, sigma_local, pd, ylim):
    fig, ax = plt.subplots(figsize=(9, 6))
    x = pd["x_eval_shock"]
    ax.plot(x, pd["y_eval_shock"], label="reference", linewidth=2, color="k")
    ax.plot(x, pd["y_pred_shock_rf"], label="RF-only fit", linestyle="--")
    ax.plot(x, pd["rf_part_shock"], label="RF part of augmented fit", linestyle="-.")
    ax.plot(x, pd["local_shock"], label="local contribution d*G", linestyle=":")
    ax.plot(x, pd["y_pred_shock_aug"], label="RF+Gaussian total", linestyle="-")
    ax.set_xlabel("x")
    ax.set_ylabel(f"u(x, t={T_SNAPSHOT})")
    ax.set_ylim(ylim)
    ax.set_title(f"{experiment_name}: sigma_local={sigma_local}, shock zoom |x|<=0.02")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{experiment_name}_sigma{sigma_local}_shock.png")
    plt.close(fig)


def plot_compact_summary(experiment_name, rows):
    sigmas = [r["sigma_local"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].plot(sigmas, [r["E_RF_full"] for r in rows], marker="o", label="E_RF (full)")
    axes[0].plot(sigmas, [r["E_aug_full"] for r in rows], marker="o", label="E_RF+G (full)")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("sigma_local")
    axes[0].set_ylabel("relative L2 error")
    axes[0].set_title("Full-domain error")
    axes[0].legend()

    axes[1].plot(sigmas, [r["E_RF_shock"] for r in rows], marker="o", label="E_RF (shock)")
    axes[1].plot(sigmas, [r["E_aug_shock"] for r in rows], marker="o", label="E_RF+G (shock)")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("sigma_local")
    axes[1].set_ylabel("relative L2 error")
    axes[1].set_title("Shock-region error")
    axes[1].legend()

    axes[2].plot(sigmas, [r["d"] for r in rows], marker="o", label="fitted d")
    axes[2].plot(sigmas, [r["ratio_shock"] for r in rows], marker="o", label="local/RF shock ratio")
    axes[2].set_xscale("log")
    axes[2].set_yscale("symlog", linthresh=1e-6)
    axes[2].set_xlabel("sigma_local")
    axes[2].set_title("Local coefficient / shock ratio")
    axes[2].legend()

    fig.suptitle(f"{experiment_name}: compact summary")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{experiment_name}_compact_summary.png")
    plt.close(fig)


def main():
    torch.set_default_dtype(DTYPE)
    device = "cpu"  # pure linear algebra; RNG draws are CPU-based regardless of training device

    Wx_t, bx_t = get_rf_params(device)
    print(f"Reproduced spatial RF realization: Wx shape={tuple(Wx_t.shape)}, bx shape={tuple(bx_t.shape)}")

    print(
        f"[Experiment A] M_fit={M_FIT_A} chosen deliberately: comfortably below Nx={NX} "
        f"({100 * M_FIT_A / NX:.0f}% of Nx) so the RF-only system ({M_FIT_A} equations, {NX} "
        f"unknowns) is underdetermined with a >={NX - M_FIT_A}-dimensional null space, giving a "
        f"genuine minimum-norm interpolation regime -- not a degenerate few-point fit."
    )
    x_fit_A = np.linspace(-1, 1, M_FIT_A)
    x_fit_B = np.linspace(-1, 1, M_FIT_B)
    print(f"[Experiment B] M_fit={M_FIT_B} (dense, overdetermined, ~{M_FIT_B / NX:.1f}x oversampling of Nx={NX}).")

    x_eval_full = np.linspace(-1, 1, EVAL_FULL_N)
    t_probe = torch.tensor([[T_SNAPSHOT]], dtype=DTYPE)
    x_shock_t, _ = build_shock_grid(t_probe, device)
    x_eval_shock = x_shock_t.detach().cpu().numpy().reshape(-1)
    print(f"Evaluation grids: full-domain N={EVAL_FULL_N}, shock N={len(x_eval_shock)} "
          f"(canonical grid, reused from sweep_local_width.build_shock_grid) -- both distinct "
          f"from the fitting grids (M_fit_A={M_FIT_A}, M_fit_B={M_FIT_B}).")

    y_eval_full = reference_values(x_eval_full, T_SNAPSHOT)
    y_eval_shock = reference_values(x_eval_shock, T_SNAPSHOT)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fit_grids = {"experiment_A_min_norm": x_fit_A, "experiment_B_dense_ls": x_fit_B}

    for experiment_name in EXPERIMENTS:
        x_fit = fit_grids[experiment_name]
        rows = []
        plot_data_by_sigma = {}
        for sigma_local in SIGMAS_LOCAL:
            result, plot_data = fit_and_evaluate(
                experiment_name, x_fit, sigma_local, Wx_t, bx_t,
                x_eval_full, x_eval_shock, y_eval_full, y_eval_shock,
            )
            rows.append(result)
            plot_data_by_sigma[sigma_local] = plot_data
            print(f"{experiment_name} sigma_local={sigma_local}: {result}")

        write_summary(experiment_name, rows)

        full_ylim = shared_ylim(
            plot_data_by_sigma,
            ["y_eval_full", "y_pred_full_rf", "rf_part_full", "local_full", "y_pred_full_aug"],
        )
        shock_ylim = shared_ylim(
            plot_data_by_sigma,
            ["y_eval_shock", "y_pred_shock_rf", "rf_part_shock", "local_shock", "y_pred_shock_aug"],
        )
        for sigma_local in SIGMAS_LOCAL:
            plot_full_domain(experiment_name, sigma_local, plot_data_by_sigma[sigma_local], full_ylim)
            plot_shock_zoom(experiment_name, sigma_local, plot_data_by_sigma[sigma_local], shock_ylim)

        plot_compact_summary(experiment_name, rows)

    print(
        "\nNOTE: this repository contains only ONE reference solution source -- the "
        "Cole-Hopf / Gauss-Hermite quadrature evaluation (burgers_rf.data.u_true / "
        "make_reference_values). There is no separate numerical PDE solver anywhere "
        "in this codebase; 'true' and 'numerical' solution are the same source here."
    )
    print("[test_direct_spatial_fit] done.")


if __name__ == "__main__":
    main()
