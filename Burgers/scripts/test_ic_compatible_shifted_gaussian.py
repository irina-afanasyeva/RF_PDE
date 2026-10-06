import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from burgers_rf.config import (  # noqa: E402
    DTYPE,
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
from burgers_rf.physics import boundary_residual, burgers_residual, initial_residual, train  # noqa: E402
from burgers_rf.evaluation import evaluate_error_metrics  # noqa: E402
from burgers_rf.diagnostics import save_component_diagnostics  # noqa: E402
from sweep_local_width import build_shock_grid  # noqa: E402

# Controlled comparison: local_time_mode="ic_compatible" (fixed for both
# cases) with a centered, narrow Gaussian (CONTROL, matching the previously
# tested configuration) vs. the shifted, post-hoc-useful Gaussian found by
# analyzing the actual PDE-trained RF residual at t=1 (TEST). physics.train
# / loss_fn / IC_BC_WEIGHT are reused completely unmodified -- only the
# local feature's center/width differ between the two cases. Local
# coefficients remain zero-initialized in both cases; the post-hoc oracle
# amplitude d_star_shock=-0.6578595 is NOT used anywhere in this script.
EPOCHS = 1000
LOCAL_TIME_MODE = "ic_compatible"
CASES = [
    ("control_centered_sigma0.0025", 0.0, 0.0025),
    ("test_shifted_sigma0.007646", 0.014, 0.007646112),
]
IC_GRID_N_X = 1000

OUTPUT_DIR = PROJECT_DIR / "outputs" / "ic_compatible_shifted_gaussian"


def relative_l2(y_true, y_pred):
    return float(np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true))


def run_case(label, center, width, x_test, t_test, y_true, x_shock, t_shock, y_true_shock, x_ic, t_ic, y_ic_true, device):
    set_seed(SEED)
    x_train, t_train = sample_training_points(M_TRAIN, device)

    print(f"=== case={label} (center={center}, width={width}) ===")
    model = BurgersRF(
        NX,
        NT,
        SIGMA_X,
        SIGMA_T,
        use_local=True,
        local_type="gaussian",
        local_center=center,
        local_width=width,
        local_time_mode=LOCAL_TIME_MODE,
        device=device,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # physics.train / loss_fn / IC_BC_WEIGHT reused completely unmodified.
    train(model, optimizer, x_train, t_train, M_BD, M_INT, NV, g, device, epochs=EPOCHS)

    # Post-hoc unweighted component losses (physics.py's own residual
    # functions, reused unmodified; BD/IC points freshly drawn, matching
    # their normal per-call resampling behavior).
    residual = burgers_residual(model, x_train, t_train, NV)
    pde_loss_final = torch.mean(residual**2).item()
    residual_bd = boundary_residual(model, M_BD, device)
    bc_loss_final = torch.mean(residual_bd**2).item()
    residual_int = initial_residual(model, M_INT, g, device)
    ic_loss_final = torch.mean(residual_int**2).item()

    y_pred = model(x_test, t_test).cpu().detach().numpy().reshape(-1)
    y_pred_shock = model(x_shock, t_shock).cpu().detach().numpy().reshape(-1)
    metrics = evaluate_error_metrics(x_test, y_true, y_pred, y_true_shock, y_pred_shock)

    y_ic_pred = model(x_ic, t_ic).cpu().detach().numpy().reshape(-1)
    ic_relative_l2 = relative_l2(y_ic_true, y_ic_pred)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    diag_stats = save_component_diagnostics(
        model, label, x_test, t_test, y_true, x_shock, t_shock, y_true_shock, OUTPUT_DIR
    )

    d_values = model.local_coefficients.detach().cpu().numpy().flatten().tolist()

    row = {
        "label": label,
        "center": center,
        "width": width,
        "global_relative_l2_error": metrics["global_relative_l2_error"],
        "smooth_relative_l2_error": metrics["smooth_relative_l2_error"],
        "shock_relative_l2_error": metrics["shock_relative_l2_error"],
        "ic_relative_l2_error": ic_relative_l2,
        "local_coef_norm": diag_stats["coef_norm"],
        "ratio_full": diag_stats["ratio_full"],
        "ratio_shock": diag_stats["ratio_shock"],
        "final_pde_loss": pde_loss_final,
        "final_unweighted_ic_loss": ic_loss_final,
        "final_unweighted_bc_loss": bc_loss_final,
        "d_values": d_values,
    }
    print(f"result: {row}")
    print(f"learned d_j: {d_values}")
    return row


def write_summary(rows):
    header = [
        "label",
        "center",
        "width",
        "global_relative_l2_error",
        "smooth_relative_l2_error",
        "shock_relative_l2_error",
        "ic_relative_l2_error",
        "local_coef_norm",
        "ratio_full",
        "ratio_shock",
        "final_pde_loss",
        "final_unweighted_ic_loss",
        "final_unweighted_bc_loss",
    ]
    csv_path = OUTPUT_DIR / "ic_compatible_shifted_gaussian_summary.csv"
    with csv_path.open("w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(row[key]) for key in header) + "\n")

    txt_path = OUTPUT_DIR / "ic_compatible_shifted_gaussian_summary.txt"
    with txt_path.open("w") as f:
        for row in rows:
            f.write(f"label = {row['label']}\n")
            for key in header[1:]:
                f.write(f"  {key} = {row[key]}\n")
            f.write(f"  d_values = {row['d_values']}\n")
            f.write("\n")


def main():
    torch.set_default_dtype(DTYPE)
    device = get_device()
    print(device)

    x_test, t_test = make_test_grid(M_TEST, N_TEST_TIMES, device)
    y_true = make_reference_values(t_test, x_test, NV)

    x_shock, t_shock = build_shock_grid(t_test, device)
    y_true_shock = make_reference_values(t_shock, x_shock, NV)

    x_ic = torch.linspace(-1, 1, IC_GRID_N_X).reshape(-1, 1).to(device)
    t_ic = torch.zeros_like(x_ic).to(device)
    y_ic_true = g(x_ic).detach().cpu().numpy().reshape(-1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for label, center, width in CASES:
        row = run_case(label, center, width, x_test, t_test, y_true, x_shock, t_shock, y_true_shock, x_ic, t_ic, y_ic_true, device)
        rows.append(row)
        write_summary(rows)

    print("[ic_compatible_shifted_gaussian] done.")


if __name__ == "__main__":
    main()
