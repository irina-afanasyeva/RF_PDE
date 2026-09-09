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
    LOCAL_TYPE,
    LOCAL_WIDTH,
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
    USE_LOCAL,
    WEIGHT_DECAY,
    get_device,
    set_seed,
)
from burgers_rf.data import g, make_reference_values, make_test_grid, sample_training_points  # noqa: E402
from burgers_rf.model import BurgersRF  # noqa: E402
from burgers_rf.physics import train  # noqa: E402
from test import plot_solution_slices, relative_l2  # noqa: E402


def main():
    set_seed(SEED)
    torch.set_default_dtype(DTYPE)
    device = get_device()
    print(device)

    x_train, t_train = sample_training_points(M_TRAIN, device)
    x_test, t_test = make_test_grid(M_TEST, N_TEST_TIMES, device)
    y_true = make_reference_values(t_test, x_test, NV)

    errors = []
    y_pred = None

    for _ in range(N_TRIALS):
        model = BurgersRF(
            NX,
            NT,
            SIGMA_X,
            SIGMA_T,
            use_local=USE_LOCAL,
            local_type=LOCAL_TYPE,
            local_center=LOCAL_CENTER,
            local_width=LOCAL_WIDTH,
            device=device,
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        train(model, optimizer, x_train, t_train, M_BD, M_INT, NV, g, device, epochs=EPOCHS)

        y_pred = model(x_test, t_test)
        y_pred = y_pred.cpu().detach().numpy().reshape(-1)
        e = relative_l2(y_true, y_pred)
        errors.append(e)

    print(errors)
    mean_error = np.mean(errors)
    print(mean_error)

    if USE_LOCAL and LOCAL_TYPE == "gaussian":
        model_type = "gaussian"
        output_dir = PROJECT_DIR / "outputs" / "gaussian"
        base_filename = (
            f"gaussian_s{LOCAL_CENTER}_sigma{LOCAL_WIDTH}_trials{N_TRIALS}_epochs{EPOCHS}"
        )
    else:
        model_type = "baseline"
        output_dir = PROJECT_DIR / "outputs" / "baseline"
        base_filename = f"baseline_trials{N_TRIALS}_epochs{EPOCHS}"

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plot_solution_slices(x_test, y_true, y_pred)
    fig.savefig(output_dir / f"{base_filename}.png")

    summary_path = output_dir / f"{base_filename}.txt"
    with summary_path.open("w") as summary_file:
        summary_file.write(f"model type: {model_type}\n")
        summary_file.write(f"USE_LOCAL: {USE_LOCAL}\n")
        summary_file.write(f"LOCAL_TYPE: {LOCAL_TYPE}\n")
        if USE_LOCAL and LOCAL_TYPE == "gaussian":
            summary_file.write(f"LOCAL_CENTER: {LOCAL_CENTER}\n")
            summary_file.write(f"LOCAL_WIDTH: {LOCAL_WIDTH}\n")
        summary_file.write(f"N_TRIALS: {N_TRIALS}\n")
        summary_file.write(f"EPOCHS: {EPOCHS}\n")
        summary_file.write(f"individual trial errors: {errors}\n")
        summary_file.write(f"mean relative L2 error: {mean_error}\n")

    plt.show()


if __name__ == "__main__":
    main()
