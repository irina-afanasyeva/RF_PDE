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
    N_TRIALS,
    NT,
    NV,
    NX,
    SIGMA_T,
    SIGMA_X,
    WEIGHT_DECAY,
    get_device,
)
from burgers_rf.data import g, make_reference_values, make_test_grid, sample_training_points  # noqa: E402
from burgers_rf.model import BurgersRF  # noqa: E402
from burgers_rf.physics import train  # noqa: E402
from test import plot_solution_slices, relative_l2  # noqa: E402


def main():
    torch.set_default_dtype(DTYPE)
    device = get_device()
    print(device)

    x_train, t_train = sample_training_points(M_TRAIN, device)
    x_test, t_test = make_test_grid(M_TEST, N_TEST_TIMES, device)
    y_true = make_reference_values(t_test, x_test, NV)

    errors = []
    y_pred = None

    for _ in range(N_TRIALS):
        model = BurgersRF(NX, NT, SIGMA_X, SIGMA_T, device=device).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        train(model, optimizer, x_train, t_train, M_BD, M_INT, NV, g, device, epochs=EPOCHS)

        y_pred = model(x_test, t_test)
        y_pred = y_pred.cpu().detach().numpy().reshape(-1)
        e = relative_l2(y_true, y_pred)
        errors.append(e)

    print(errors)
    print(np.mean(errors))

    plot_solution_slices(x_test, y_true, y_pred)
    plt.show()


if __name__ == "__main__":
    main()
