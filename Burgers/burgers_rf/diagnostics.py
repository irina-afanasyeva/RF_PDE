"""
Reusable, diagnostic-only helper for visualizing an already-trained
BurgersRF model's u_RF / u_local / u_total decomposition. Does not train
or retrain anything; only calls model.forward_components on grids the
caller already built for its error metrics.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt


def _l2_norm(values):
    return float(np.linalg.norm(values.reshape(-1)))


def _grid_dims(t):
    t_np = t.detach().cpu().numpy().reshape(-1)
    n_times = len(np.unique(t_np))
    n_x = len(t_np) // n_times
    return n_times, n_x


def _components_as_numpy(model, x, t):
    u_total, u_rf, u_local = model.forward_components(x, t)
    return (
        u_total.detach().cpu().numpy().reshape(-1),
        u_rf.detach().cpu().numpy().reshape(-1),
        u_local.detach().cpu().numpy().reshape(-1),
    )


def _plot_component_panels(x, t, y_true, u_total, u_rf, u_local, title, save_path):
    n_times, n_x = _grid_dims(t)
    x2 = x.detach().cpu().numpy().reshape(n_times, n_x)
    t2 = t.detach().cpu().numpy().reshape(n_times, n_x)
    y_true2 = y_true.reshape(n_times, n_x)
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

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def save_component_diagnostics(
    model,
    case_name,
    x_full,
    t_full,
    y_true_full,
    x_shock,
    t_shock,
    y_true_shock,
    output_dir,
):
    """Given an already-trained model, save full-domain and shock-region
    u_RF / u_local / u_total component plots plus a numeric summary.
    Does not train or evaluate error metrics itself."""
    output_dir.mkdir(parents=True, exist_ok=True)

    u_total_full, u_rf_full, u_local_full = _components_as_numpy(model, x_full, t_full)
    u_total_shock, u_rf_shock, u_local_shock = _components_as_numpy(model, x_shock, t_shock)

    _plot_component_panels(
        x_full, t_full, y_true_full, u_total_full, u_rf_full, u_local_full,
        title=f"{case_name}: full-domain components",
        save_path=output_dir / f"components_{case_name}_full.png",
    )
    _plot_component_panels(
        x_shock, t_shock, y_true_shock, u_total_shock, u_rf_shock, u_local_shock,
        title=f"{case_name}: shock-region components",
        save_path=output_dir / f"components_{case_name}_shock.png",
    )

    if model.use_local:
        coef_norm = float(torch.linalg.norm(model.local_coefficients).item())
    else:
        coef_norm = 0.0

    ratio_full = _l2_norm(u_local_full) / _l2_norm(u_rf_full)
    ratio_shock = _l2_norm(u_local_shock) / _l2_norm(u_rf_shock)

    summary_path = output_dir / f"components_{case_name}_summary.txt"
    with summary_path.open("w") as f:
        f.write(f"case: {case_name}\n")
        f.write(f"use_local: {model.use_local}\n")
        f.write(f"norm of learned local coefficients: {coef_norm}\n")
        f.write(f"||u_local||_2 / ||u_RF||_2 (full grid): {ratio_full}\n")
        f.write(f"||u_local||_2 / ||u_RF||_2 (shock region |x|<=0.02): {ratio_shock}\n")

    return {
        "coef_norm": coef_norm,
        "ratio_full": ratio_full,
        "ratio_shock": ratio_shock,
    }
