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
from burgers_rf.physics import boundary_residual, burgers_residual, initial_residual  # noqa: E402
from burgers_rf.evaluation import evaluate_error_metrics  # noqa: E402
from burgers_rf.diagnostics import save_component_diagnostics  # noqa: E402
from sweep_local_width import build_shock_grid  # noqa: E402

# Quick controlled check: does the IC loss weight suppress the learned local
# Gaussian coefficients? Reuses burgers_residual/boundary_residual/
# initial_residual UNMODIFIED from physics.py; only the loss combination and
# train loop are duplicated locally (to allow independent IC/BC weights),
# per instruction not to refactor physics.py before the meeting.
SIGMA = 0.0025
EPOCHS = 1000
IC_WEIGHTS = [1000, 100, 10, 0]
BC_WEIGHT = 1000
IC_GRID_N_X = 1000

OUTPUT_DIR = PROJECT_DIR / "outputs" / "ic_weight_sigma0.0025"


def loss_fn_separate(model, x, t, m_bd, m_int, nv, g, device, ic_weight, bc_weight):
    residual = burgers_residual(model, x, t, nv)
    residual_bd = boundary_residual(model, m_bd, device)
    residual_int = initial_residual(model, m_int, g, device)

    pde_loss = torch.mean(residual**2)
    ic_loss = torch.mean(residual_int**2)
    bc_loss = torch.mean(residual_bd**2)

    total = pde_loss + ic_weight * ic_loss + bc_weight * bc_loss
    return total, pde_loss, ic_loss, bc_loss


def train_ic_bc_separate(model, optimizer, x, t, m_bd, m_int, nv, g, device, ic_weight, bc_weight, epochs):
    pde_loss_val = ic_loss_val = bc_loss_val = None
    for epoch in range(epochs):
        optimizer.zero_grad()
        total, pde_loss, ic_loss, bc_loss = loss_fn_separate(
            model, x, t, m_bd, m_int, nv, g, device, ic_weight, bc_weight
        )
        total.backward(retain_graph=True)
        optimizer.step()
        pde_loss_val, ic_loss_val, bc_loss_val = pde_loss.item(), ic_loss.item(), bc_loss.item()
        if epoch % 1000 == 0:
            print(f"  epoch {epoch}, total loss {total.item():.6e}")
    return pde_loss_val, ic_loss_val, bc_loss_val


def relative_l2(y_true, y_pred):
    return float(np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true))


def run_case(ic_weight, x_test, t_test, y_true, x_shock, t_shock, y_true_shock, x_ic, t_ic, y_ic_true, device):
    set_seed(SEED)
    x_train, t_train = sample_training_points(M_TRAIN, device)

    print(f"=== IC_WEIGHT={ic_weight}, BC_WEIGHT={BC_WEIGHT} ===")
    model = BurgersRF(
        NX,
        NT,
        SIGMA_X,
        SIGMA_T,
        use_local=True,
        local_type="gaussian",
        local_center=LOCAL_CENTER,
        local_width=SIGMA,
        device=device,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    pde_loss_final, ic_loss_final, bc_loss_final = train_ic_bc_separate(
        model, optimizer, x_train, t_train, M_BD, M_INT, NV, g, device, ic_weight, BC_WEIGHT, EPOCHS
    )

    y_pred = model(x_test, t_test).cpu().detach().numpy().reshape(-1)
    y_pred_shock = model(x_shock, t_shock).cpu().detach().numpy().reshape(-1)
    metrics = evaluate_error_metrics(x_test, y_true, y_pred, y_true_shock, y_pred_shock)

    y_ic_pred = model(x_ic, t_ic).cpu().detach().numpy().reshape(-1)
    ic_relative_l2 = relative_l2(y_ic_true, y_ic_pred)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    case_name = f"ic_weight{ic_weight}"
    diag_stats = save_component_diagnostics(
        model, case_name, x_test, t_test, y_true, x_shock, t_shock, y_true_shock, OUTPUT_DIR
    )

    row = {
        "ic_weight": ic_weight,
        "bc_weight": BC_WEIGHT,
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
    }
    print(f"result: {row}")
    return row


def write_summary(rows):
    header = [
        "ic_weight",
        "bc_weight",
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
    csv_path = OUTPUT_DIR / "ic_weight_summary.csv"
    with csv_path.open("w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(row[key]) for key in header) + "\n")

    txt_path = OUTPUT_DIR / "ic_weight_summary.txt"
    with txt_path.open("w") as f:
        for row in rows:
            f.write(f"IC_WEIGHT = {row['ic_weight']} (BC_WEIGHT = {row['bc_weight']})\n")
            for key in header[2:]:
                f.write(f"  {key} = {row[key]}\n")
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
    for ic_weight in IC_WEIGHTS:
        row = run_case(ic_weight, x_test, t_test, y_true, x_shock, t_shock, y_true_shock, x_ic, t_ic, y_ic_true, device)
        rows.append(row)
        write_summary(rows)

    print("[ic_weight_sigma0.0025] done.")


if __name__ == "__main__":
    main()
