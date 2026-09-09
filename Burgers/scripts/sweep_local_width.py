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
    EPOCHS,
    LR,
    LOCAL_CENTER,
    LOCAL_TYPE,
    M_BD,
    M_INT,
    M_TEST,
    M_TRAIN,
    N_TEST_TIMES,
    N_TRIALS,
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
from test import plot_solution_slices, relative_l2  # noqa: E402

# Each case only varies USE_LOCAL / LOCAL_WIDTH; every other hyperparameter
# stays fixed at its config.py value.
CASES = [
    {"name": "baseline", "use_local": False, "local_width": None},
    {"name": "gaussian_w0.02", "use_local": True, "local_width": 0.02},
    {"name": "gaussian_w0.05", "use_local": True, "local_width": 0.05},
    {"name": "gaussian_w0.10", "use_local": True, "local_width": 0.10},
    {"name": "gaussian_w0.20", "use_local": True, "local_width": 0.20},
    {"name": "gaussian_w0.30", "use_local": True, "local_width": 0.30},
    {"name": "gaussian_w0.40", "use_local": True, "local_width": 0.40},
    {"name": "gaussian_w0.50", "use_local": True, "local_width": 0.50},
]

# Fixed shock-region evaluation grid, from the analytic-reference gradient
# diagnostic: |x| <= 0.02, evaluated on a dedicated fine grid (the global
# M_TEST=100 test grid is too coarse for this narrow region). Test times are
# taken from the already-built global t_test tensor so they are guaranteed
# to match the global evaluation exactly.
SHOCK_X_HALF_WIDTH = 0.02
SHOCK_N_X = 200


def build_shock_grid(t_test, device):
    shock_times = torch.unique(t_test.detach().cpu())
    x_shock = torch.linspace(-SHOCK_X_HALF_WIDTH, SHOCK_X_HALF_WIDTH, SHOCK_N_X)
    shock_product = torch.cartesian_prod(shock_times, x_shock)
    x_shock_grid = shock_product[:, [1]].to(device)
    t_shock_grid = shock_product[:, [0]].to(device)
    return x_shock_grid, t_shock_grid


def run_case(case, x_test, t_test, y_true, x_shock, t_shock, y_true_shock, device):
    use_local = case["use_local"]
    local_width = case["local_width"]

    # Reset the seed right before sampling/model construction so every case
    # draws the same RF initialization and the same stochastic sampling
    # sequence (USE_LOCAL does not consume any extra random numbers).
    set_seed(SEED)
    x_train, t_train = sample_training_points(M_TRAIN, device)

    errors = []
    shock_errors = []
    y_pred = None
    for _ in range(N_TRIALS):
        model = BurgersRF(
            NX,
            NT,
            SIGMA_X,
            SIGMA_T,
            use_local=use_local,
            local_type=LOCAL_TYPE,
            local_center=LOCAL_CENTER,
            local_width=local_width if local_width is not None else 0.05,
            device=device,
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        train(model, optimizer, x_train, t_train, M_BD, M_INT, NV, g, device, epochs=EPOCHS)

        y_pred = model(x_test, t_test)
        y_pred = y_pred.cpu().detach().numpy().reshape(-1)
        errors.append(relative_l2(y_true, y_pred))

        # Additional evaluation only, on the already-trained model — no retraining.
        y_pred_shock = model(x_shock, t_shock)
        y_pred_shock = y_pred_shock.cpu().detach().numpy().reshape(-1)
        shock_errors.append(relative_l2(y_true_shock, y_pred_shock))

    mean_error = np.mean(errors)
    mean_shock_error = np.mean(shock_errors)

    if use_local:
        model_type = "gaussian"
        output_dir = PROJECT_DIR / "outputs" / "gaussian"
        base_filename = (
            f"gaussian_s{LOCAL_CENTER}_sigma{local_width}_trials{N_TRIALS}_epochs{EPOCHS}"
        )
    else:
        model_type = "baseline"
        output_dir = PROJECT_DIR / "outputs" / "baseline"
        base_filename = f"baseline_trials{N_TRIALS}_epochs{EPOCHS}"

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, _ = plot_solution_slices(x_test, y_true, y_pred)
    fig.savefig(output_dir / f"{base_filename}.png")

    summary_path = output_dir / f"{base_filename}.txt"
    with summary_path.open("w") as summary_file:
        summary_file.write(f"model type: {model_type}\n")
        summary_file.write(f"USE_LOCAL: {use_local}\n")
        summary_file.write(f"LOCAL_TYPE: {LOCAL_TYPE}\n")
        if use_local:
            summary_file.write(f"LOCAL_CENTER: {LOCAL_CENTER}\n")
            summary_file.write(f"LOCAL_WIDTH: {local_width}\n")
        summary_file.write(f"N_TRIALS: {N_TRIALS}\n")
        summary_file.write(f"EPOCHS: {EPOCHS}\n")
        summary_file.write(f"individual trial errors: {errors}\n")
        summary_file.write(f"mean relative L2 error: {mean_error}\n")
        summary_file.write(f"shock region: |x| <= {SHOCK_X_HALF_WIDTH}, {SHOCK_N_X} spatial points\n")
        summary_file.write(f"individual trial shock errors: {shock_errors}\n")
        summary_file.write(f"mean shock relative L2 error: {mean_shock_error}\n")

    return mean_error, mean_shock_error


def main():
    torch.set_default_dtype(DTYPE)
    device = get_device()
    print(device)

    x_test, t_test = make_test_grid(M_TEST, N_TEST_TIMES, device)
    y_true = make_reference_values(t_test, x_test, NV)

    x_shock, t_shock = build_shock_grid(t_test, device)
    y_true_shock = make_reference_values(t_shock, x_shock, NV)

    results = []
    for case in CASES:
        mean_error, mean_shock_error = run_case(
            case, x_test, t_test, y_true, x_shock, t_shock, y_true_shock, device
        )
        results.append((case["name"], mean_error, mean_shock_error))
        print(
            f"{case['name']}: mean relative L2 error = {mean_error}, "
            f"mean shock relative L2 error = {mean_shock_error}"
        )

    print("\ncase                 mean_relative_l2_error    mean_shock_relative_l2_error")
    for name, mean_error, mean_shock_error in results:
        print(f"{name:<20} {mean_error:<26} {mean_shock_error}")


if __name__ == "__main__":
    main()
