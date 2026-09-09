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
from sweep_local_width import SHOCK_N_X, SHOCK_X_HALF_WIDTH, build_shock_grid  # noqa: E402

# Controlled comparison: a single fixed-width Gaussian local enrichment vs. a
# sum of fixed-width Gaussians (same center, independent trainable temporal
# coefficients per width). Known controlled baseline (USE_LOCAL=False,
# SEED=42): global L2 = 0.27149430095802507, shock L2 = 0.8390487998889169.
CASES = [
    {
        "name": "single_sigma0.20",
        "local_type": "gaussian",
        "local_width": 0.20,
        "local_widths": None,
    },
    {
        "name": "gaussian_sum_0.05_0.10_0.20",
        "local_type": "gaussian_sum",
        "local_width": None,
        "local_widths": [0.05, 0.10, 0.20],
    },
]

OUTPUT_DIR = PROJECT_DIR / "outputs" / "gaussian_sum"


def run_case(case, x_test, t_test, y_true, x_shock, t_shock, y_true_shock, device):
    local_type = case["local_type"]
    local_width = case["local_width"]
    local_widths = case["local_widths"]

    # Reset the seed right before sampling/model construction so every case
    # draws the same RF initialization and the same stochastic sampling
    # sequence, matching the width-sweep reproducibility convention.
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
            use_local=True,
            local_type=local_type,
            local_center=LOCAL_CENTER,
            local_width=local_width if local_width is not None else 0.05,
            local_widths=local_widths,
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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base_filename = f"{case['name']}_trials{N_TRIALS}_epochs{EPOCHS}"

    fig, _ = plot_solution_slices(x_test, y_true, y_pred)
    fig.savefig(OUTPUT_DIR / f"{base_filename}.png")

    summary_path = OUTPUT_DIR / f"{base_filename}.txt"
    with summary_path.open("w") as summary_file:
        summary_file.write(f"case: {case['name']}\n")
        summary_file.write(f"local_type: {local_type}\n")
        summary_file.write(f"LOCAL_CENTER: {LOCAL_CENTER}\n")
        if local_type == "gaussian":
            summary_file.write(f"local_width: {local_width}\n")
        else:
            summary_file.write(f"local_widths: {local_widths}\n")
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

    print("\ncase                              mean_relative_l2_error    mean_shock_relative_l2_error")
    for name, mean_error, mean_shock_error in results:
        print(f"{name:<33} {mean_error:<26} {mean_shock_error}")


if __name__ == "__main__":
    main()
