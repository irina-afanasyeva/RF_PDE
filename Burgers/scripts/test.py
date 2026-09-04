import matplotlib.pyplot as plt
import numpy as np


def relative_l2(y_true, y_pred):
    return np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true)


def plot_solution_slices(x_test, y_true, y_pred):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].plot(x_test.cpu().detach().numpy()[:100], y_pred[:100], label="predict")
    ax[0].plot(x_test.cpu().detach().numpy()[:100], y_true[:100], label="True")
    ax[0].set_title("t=0.0")
    ax[0].legend()

    ax[1].plot(x_test.cpu().detach().numpy()[300:400], y_pred[300:400], label="predict")
    ax[1].plot(x_test.cpu().detach().numpy()[300:400], y_true[300:400], label="True")
    ax[1].set_title("t=3/9")
    ax[1].legend()

    ax[2].plot(x_test.cpu().detach().numpy()[900:], y_pred[900:], label="predict")
    ax[2].plot(x_test.cpu().detach().numpy()[900:], y_true[900:], label="True")
    ax[2].set_title("t=1")
    ax[2].legend()

    return fig, ax
