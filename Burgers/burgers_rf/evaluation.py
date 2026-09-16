"""
Reusable relative-L2 evaluation helper for global / smooth-region /
shock-region error metrics.

Takes already-computed grids and predictions; does not build grids,
train, or touch a model. This module is the package-level canonical
home for SHOCK_X_HALF_WIDTH going forward; the existing
scripts/sweep_local_width.py keeps its own copy and is left unmodified.
"""
import numpy as np

SHOCK_X_HALF_WIDTH = 0.02


def _to_numpy(values):
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values).reshape(-1)


def _relative_l2(y_true, y_pred):
    return np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true)


def evaluate_error_metrics(
    x_global,
    y_true_global,
    y_pred_global,
    y_true_shock,
    y_pred_shock,
    shock_half_width=SHOCK_X_HALF_WIDTH,
):
    """Compute global, smooth-region (|x| > shock_half_width), and
    shock-region relative L2 errors from already-evaluated grids.

    x_global aligns index-for-index with y_true_global/y_pred_global; the
    shock-region grid/predictions are supplied separately (e.g. from a
    dedicated fine grid) and used as-is.
    """
    x_global_np = _to_numpy(x_global)
    y_true_global_np = _to_numpy(y_true_global)
    y_pred_global_np = _to_numpy(y_pred_global)
    y_true_shock_np = _to_numpy(y_true_shock)
    y_pred_shock_np = _to_numpy(y_pred_shock)

    global_relative_l2_error = _relative_l2(y_true_global_np, y_pred_global_np)

    smooth_mask = np.abs(x_global_np) > shock_half_width
    smooth_relative_l2_error = _relative_l2(
        y_true_global_np[smooth_mask], y_pred_global_np[smooth_mask]
    )

    shock_relative_l2_error = _relative_l2(y_true_shock_np, y_pred_shock_np)

    return {
        "global_relative_l2_error": global_relative_l2_error,
        "smooth_relative_l2_error": smooth_relative_l2_error,
        "shock_relative_l2_error": shock_relative_l2_error,
    }
