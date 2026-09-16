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
    LOCAL_CENTER,
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
from burgers_rf.diagnostics import save_component_diagnostics  # noqa: E402
from sweep_local_width import SHOCK_N_X, SHOCK_X_HALF_WIDTH, build_shock_grid  # noqa: E402

# Small-sigma width sweep. Each sigma is an INDEPENDENT controlled run
# (set_seed -> sample -> fresh model -> fresh optimizer -> train), matching
# the original width-sweep convention. The canonical 200-point shock grid
# (imported unmodified from sweep_local_width) is used for the error metrics
# so shock_relative_l2_error stays comparable with every previous result.
# A separate, odd 2001-point fine grid is used ONLY to visualize/measure the
# local component at tiny sigma, never for the canonical metric.
SIGMAS = [1e-4, 5e-4, 2.5e-3, 1.25e-2, 6.25e-2, 3.125e-1]
FINE_SHOCK_N_X = 2001

OUTPUT_DIR = PROJECT_DIR / "outputs" / "small_sigma_sweep"


def build_fine_shock_grid(t_test, device):
    """Diagnostic-only fine grid over the same shock region, odd point
    count so x=0 is (numerically) included. NOT used for any error metric."""
    shock_times = torch.unique(t_test.detach().cpu())
    x_fine = torch.linspace(-SHOCK_X_HALF_WIDTH, SHOCK_X_HALF_WIDTH, FINE_SHOCK_N_X)
    product = torch.cartesian_prod(shock_times, x_fine)
    x_fine_grid = product[:, [1]].to(device)
    t_fine_grid = product[:, [0]].to(device)
    return x_fine_grid, t_fine_grid


def verify_setup(x_shock, x_fine):
    assert torch.get_default_dtype() == DTYPE, "dtype must match config.DTYPE (float64)"

    shock_x_unique = torch.unique(x_shock.detach().cpu()).numel()
    assert shock_x_unique == SHOCK_N_X, (
        f"canonical shock grid must remain unchanged: expected {SHOCK_N_X} x-points, got {shock_x_unique}"
    )

    fine_x_unique = torch.unique(x_fine.detach().cpu())
    assert fine_x_unique.numel() == FINE_SHOCK_N_X, (
        f"fine grid must have {FINE_SHOCK_N_X} x-points, got {fine_x_unique.numel()}"
    )
    assert float(fine_x_unique.abs().min()) < 1e-12, "fine grid must include x=0"

    print(
        f"Setup verification passed: dtype={torch.get_default_dtype()}, "
        f"canonical shock grid unchanged ({SHOCK_N_X} points), "
        f"fine grid has {FINE_SHOCK_N_X} points including x=0."
    )


def compute_components_numpy(model, x, t):
    u_total, u_rf, u_local = model.forward_components(x, t)
    return (
        u_total.detach().cpu().numpy().reshape(-1),
        u_rf.detach().cpu().numpy().reshape(-1),
        u_local.detach().cpu().numpy().reshape(-1),
    )


def save_fine_shock_plot(case_name, x_fine, t_fine, y_true_fine, u_total, u_rf, u_local):
    """Special fine-shock component plot, implemented locally (not via
    save_component_diagnostics) so no redundant full-domain plot is made."""
    t_np = t_fine.detach().cpu().numpy().reshape(-1)
    n_times = len(np.unique(t_np))
    n_x = len(t_np) // n_times

    x2 = x_fine.detach().cpu().numpy().reshape(n_times, n_x)
    t2 = t_np.reshape(n_times, n_x)
    y_true2 = y_true_fine.reshape(n_times, n_x)
    u_total2 = u_total.reshape(n_times, n_x)
    u_rf2 = u_rf.reshape(n_times, n_x)
    u_local2 = u_local.reshape(n_times, n_x)

    panel_idx = sorted(set([0, n_times // 2, n_times - 1]))
    fig, axes = plt.subplots(1, len(panel_idx), figsize=(5 * len(panel_idx), 4))
    if len(panel_idx) == 1:
        axes = [axes]

    for ax, i in zip(axes, panel_idx):
        ax.plot(x2[i], y_true2[i], label="u_true", linewidth=2)
        ax.plot(x2[i], u_rf2[i], label="u_RF", linestyle="--")
        ax.plot(x2[i], u_local2[i], label="u_local", linestyle=":")
        ax.plot(x2[i], u_total2[i], label="u_total", linestyle="-.")
        ax.set_title(f"t = {t2[i, 0]:.3f}")
        ax.set_xlabel("x")
        ax.legend()

    fig.suptitle(f"{case_name}: fine shock-region components (N_x={n_x})")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"components_{case_name}_fine_shock.png")
    plt.close(fig)


def write_summary(rows):
    header = [
        "sigma",
        "global_relative_l2_error",
        "smooth_relative_l2_error",
        "shock_relative_l2_error",
        "local_coef_norm",
        "ratio_full",
        "ratio_shock",
        "ratio_shock_fine",
        "collocation_total",
        "collocation_within_sigma",
        "collocation_within_3sigma",
    ]
    csv_path = OUTPUT_DIR / "small_sigma_sweep_summary.csv"
    with csv_path.open("w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(row[key]) for key in header) + "\n")

    txt_path = OUTPUT_DIR / "small_sigma_sweep_summary.txt"
    with txt_path.open("w") as f:
        for row in rows:
            f.write(f"sigma = {row['sigma']}\n")
            for key in header[1:]:
                f.write(f"  {key} = {row[key]}\n")
            f.write("\n")


def run_case(sigma, x_test, t_test, y_true, x_shock, t_shock, y_true_shock, x_fine, t_fine, y_true_fine, device):
    # Independent controlled run: reseed, resample, fresh model, fresh optimizer.
    set_seed(SEED)
    x_train, t_train = sample_training_points(M_TRAIN, device)

    x_train_np = x_train.detach().cpu().numpy().reshape(-1)
    total_count = x_train_np.shape[0]
    count_1sigma = int((np.abs(x_train_np) <= sigma).sum())
    count_3sigma = int((np.abs(x_train_np) <= 3 * sigma).sum())

    print(f"=== sigma={sigma} ===")
    print(f"dtype: {torch.get_default_dtype()}")
    print(f"total collocation count: {total_count}")
    print(f"count |x| <= sigma: {count_1sigma}")
    print(f"count |x| <= 3*sigma: {count_3sigma}")

    model = BurgersRF(
        NX,
        NT,
        SIGMA_X,
        SIGMA_T,
        use_local=True,
        local_type="gaussian",
        local_center=LOCAL_CENTER,
        local_width=sigma,
        device=device,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    train(model, optimizer, x_train, t_train, M_BD, M_INT, NV, g, device, epochs=EPOCHS)

    y_pred = model(x_test, t_test).cpu().detach().numpy().reshape(-1)
    y_pred_shock = model(x_shock, t_shock).cpu().detach().numpy().reshape(-1)
    metrics = evaluate_error_metrics(x_test, y_true, y_pred, y_true_shock, y_pred_shock)

    case_name = f"sigma{sigma}"
    diag_stats = save_component_diagnostics(
        model, case_name, x_test, t_test, y_true, x_shock, t_shock, y_true_shock, OUTPUT_DIR
    )

    u_total_fine, u_rf_fine, u_local_fine = compute_components_numpy(model, x_fine, t_fine)
    ratio_shock_fine = float(np.linalg.norm(u_local_fine) / np.linalg.norm(u_rf_fine))
    save_fine_shock_plot(case_name, x_fine, t_fine, y_true_fine, u_total_fine, u_rf_fine, u_local_fine)

    return {
        "sigma": sigma,
        "global_relative_l2_error": metrics["global_relative_l2_error"],
        "smooth_relative_l2_error": metrics["smooth_relative_l2_error"],
        "shock_relative_l2_error": metrics["shock_relative_l2_error"],
        "local_coef_norm": diag_stats["coef_norm"],
        "ratio_full": diag_stats["ratio_full"],
        "ratio_shock": diag_stats["ratio_shock"],
        "ratio_shock_fine": ratio_shock_fine,
        "collocation_total": total_count,
        "collocation_within_sigma": count_1sigma,
        "collocation_within_3sigma": count_3sigma,
    }


def main():
    torch.set_default_dtype(DTYPE)
    device = get_device()
    print(device)

    x_test, t_test = make_test_grid(M_TEST, N_TEST_TIMES, device)
    y_true = make_reference_values(t_test, x_test, NV)

    x_shock, t_shock = build_shock_grid(t_test, device)
    y_true_shock = make_reference_values(t_shock, x_shock, NV)

    x_fine, t_fine = build_fine_shock_grid(t_test, device)
    y_true_fine = make_reference_values(t_fine, x_fine, NV)

    verify_setup(x_shock, x_fine)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for sigma in SIGMAS:
        row = run_case(
            sigma, x_test, t_test, y_true, x_shock, t_shock, y_true_shock, x_fine, t_fine, y_true_fine, device
        )
        rows.append(row)
        write_summary(rows)
        print(f"sigma={sigma} done: {row}")

    print("[small_sigma_sweep] done.")


if __name__ == "__main__":
    main()
