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
    IC_BC_WEIGHT,
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

# Controlled test of the professor's suggestion: "increase the weight of
# collocation points in the shock region" -- explicit PER-POINT PDE-loss
# reweighting on the SAME fixed original 3600 collocation points (no
# enrichment/redistribution/resampling), combined with the already-
# implemented IC-compatible local Gaussian feature (local_time_mode=
# "ic_compatible"). physics.burgers_residual/boundary_residual/
# initial_residual are reused unmodified; only the loss combination is
# duplicated locally (physics.py itself is not touched).
SIGMA_LOCAL = 0.0025
DELTA = 0.02
ALPHAS = [1, 2, 5, 10, 20]
EPOCHS = 1000
IC_GRID_N_X = 1000

OUTPUT_DIR = PROJECT_DIR / "outputs" / "shock_point_weighting"


def compute_weights(x, delta, alpha):
    """w_i = alpha if |x_i| <= delta else 1. x is a fixed (N,1) tensor;
    weights depend only on the (unchanging) collocation coordinates."""
    in_shock = x.detach().abs() <= delta
    return torch.where(in_shock, torch.full_like(x, float(alpha)), torch.ones_like(x))


def weighted_pde_loss(residual, weights):
    """Normalized weighted mean: sum(w_i r_i^2) / sum(w_i). Reduces exactly
    to torch.mean(residual**2) when weights are all 1 (alpha=1)."""
    return torch.sum(weights * residual**2) / torch.sum(weights)


def loss_fn_weighted(model, x, t, weights, m_bd, m_int, nv, g, device):
    residual = burgers_residual(model, x, t, nv)
    pde_loss = weighted_pde_loss(residual, weights)

    residual_bd = boundary_residual(model, m_bd, device)
    residual_int = initial_residual(model, m_int, g, device)
    bc_loss = torch.mean(residual_bd**2)
    ic_loss = torch.mean(residual_int**2)

    total = pde_loss + IC_BC_WEIGHT * (ic_loss + bc_loss)
    return total, pde_loss, ic_loss, bc_loss


def train_weighted(model, optimizer, x, t, weights, m_bd, m_int, nv, g, device, epochs):
    pde_loss_val = ic_loss_val = bc_loss_val = None
    for epoch in range(epochs):
        optimizer.zero_grad()
        total, pde_loss, ic_loss, bc_loss = loss_fn_weighted(model, x, t, weights, m_bd, m_int, nv, g, device)
        total.backward(retain_graph=True)
        optimizer.step()
        pde_loss_val, ic_loss_val, bc_loss_val = pde_loss.item(), ic_loss.item(), bc_loss.item()
        if epoch % 1000 == 0:
            print(f"  epoch {epoch}, total loss {total.item():.6e}")
    return pde_loss_val, ic_loss_val, bc_loss_val


def relative_l2(y_true, y_pred):
    return float(np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true))


def build_model(device):
    """seed -> burn RNG exactly as sample_training_points would -> fresh
    model/optimizer. The x_train/t_train actually used for training are
    passed in separately and are NOT re-derived here (see run_case)."""
    set_seed(SEED)
    sample_training_points(M_TRAIN, device)  # burn RNG to the standard position; output discarded
    model = BurgersRF(
        NX,
        NT,
        SIGMA_X,
        SIGMA_T,
        use_local=True,
        local_type="gaussian",
        local_center=LOCAL_CENTER,
        local_width=SIGMA_LOCAL,
        local_time_mode="ic_compatible",
        device=device,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    return model, optimizer


def run_case(alpha, x_train, t_train, x_test, t_test, y_true, x_shock, t_shock, y_true_shock, x_ic, t_ic, y_ic_true, device):
    print(f"=== alpha={alpha} ===")
    weights = compute_weights(x_train, DELTA, alpha)
    n_total = x_train.shape[0]
    n_shock = int((x_train.detach().abs() <= DELTA).sum().item())
    effective_shock_weight_fraction = float(weights[x_train.detach().abs() <= DELTA].sum() / weights.sum())
    print(f"n_total={n_total}, n_shock={n_shock}, effective_shock_weight_fraction={effective_shock_weight_fraction:.6f}")

    model, optimizer = build_model(device)
    print(f"model.local_time_mode = {model.local_time_mode}")

    pde_loss_final, ic_loss_final, bc_loss_final = train_weighted(
        model, optimizer, x_train, t_train, weights, M_BD, M_INT, NV, g, device, EPOCHS
    )

    # Post-hoc: UNWEIGHTED PDE loss on the same 3600 points, for cross-alpha comparability.
    residual_final = burgers_residual(model, x_train, t_train, NV)
    unweighted_pde_loss_final = torch.mean(residual_final**2).item()

    y_pred = model(x_test, t_test).cpu().detach().numpy().reshape(-1)
    y_pred_shock = model(x_shock, t_shock).cpu().detach().numpy().reshape(-1)
    metrics = evaluate_error_metrics(x_test, y_true, y_pred, y_true_shock, y_pred_shock)

    y_ic_pred = model(x_ic, t_ic).cpu().detach().numpy().reshape(-1)
    ic_relative_l2 = relative_l2(y_ic_true, y_ic_pred)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    case_name = f"alpha{alpha}"
    diag_stats = save_component_diagnostics(
        model, case_name, x_test, t_test, y_true, x_shock, t_shock, y_true_shock, OUTPUT_DIR
    )

    d_values = model.local_coefficients.detach().cpu().numpy().flatten().tolist()

    row = {
        "alpha": alpha,
        "n_total": n_total,
        "n_shock": n_shock,
        "effective_shock_weight_fraction": effective_shock_weight_fraction,
        "global_relative_l2_error": metrics["global_relative_l2_error"],
        "smooth_relative_l2_error": metrics["smooth_relative_l2_error"],
        "shock_relative_l2_error": metrics["shock_relative_l2_error"],
        "ic_relative_l2_error": ic_relative_l2,
        "local_coef_norm": diag_stats["coef_norm"],
        "ratio_full": diag_stats["ratio_full"],
        "ratio_shock": diag_stats["ratio_shock"],
        "final_weighted_pde_loss": pde_loss_final,
        "final_unweighted_pde_loss": unweighted_pde_loss_final,
        "final_unweighted_ic_loss": ic_loss_final,
        "final_unweighted_bc_loss": bc_loss_final,
        "d_values": d_values,
    }
    print(f"result: {row}")
    return row


SUMMARY_HEADER = [
    "alpha", "n_total", "n_shock", "effective_shock_weight_fraction",
    "global_relative_l2_error", "smooth_relative_l2_error", "shock_relative_l2_error", "ic_relative_l2_error",
    "local_coef_norm", "ratio_full", "ratio_shock",
    "final_weighted_pde_loss", "final_unweighted_pde_loss", "final_unweighted_ic_loss", "final_unweighted_bc_loss",
]


def write_summary(rows):
    csv_path = OUTPUT_DIR / "shock_point_weighting_summary.csv"
    with csv_path.open("w") as f:
        f.write(",".join(SUMMARY_HEADER) + "\n")
        for row in rows:
            f.write(",".join(str(row[key]) for key in SUMMARY_HEADER) + "\n")

    txt_path = OUTPUT_DIR / "shock_point_weighting_summary.txt"
    with txt_path.open("w") as f:
        for row in rows:
            f.write(f"alpha = {row['alpha']}\n")
            for key in SUMMARY_HEADER[1:]:
                f.write(f"  {key} = {row[key]}\n")
            f.write(f"  d_values = {row['d_values']}\n")
            f.write("\n")


def plot_vs_alpha(rows):
    alphas = [r["alpha"] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(alphas, [r["global_relative_l2_error"] for r in rows], marker="o")
    ax.set_xlabel("alpha")
    ax.set_ylabel("global relative L2 error")
    ax.set_title("A. Global error vs alpha")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "A_global_error_vs_alpha.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(alphas, [r["shock_relative_l2_error"] for r in rows], marker="o", color="tab:red")
    ax.set_xlabel("alpha")
    ax.set_ylabel("shock relative L2 error")
    ax.set_title("B. Shock error vs alpha")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "B_shock_error_vs_alpha.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(alphas, [r["ratio_shock"] for r in rows], marker="o", color="tab:green")
    ax.set_yscale("symlog", linthresh=1e-6)
    ax.set_xlabel("alpha")
    ax.set_ylabel("local/RF shock ratio (symlog)")
    ax.set_title("C. Local/RF shock ratio vs alpha")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "C_local_RF_shock_ratio_vs_alpha.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(alphas, [r["local_coef_norm"] for r in rows], marker="o", color="tab:purple")
    ax.set_xlabel("alpha")
    ax.set_ylabel("||d||_2")
    ax.set_title("D. ||d||_2 vs alpha")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "D_d_norm_vs_alpha.png")
    plt.close(fig)


def cheap_checks(device="cpu"):
    """All checks required before the full sweep. No training performed."""
    torch.set_default_dtype(DTYPE)

    print("=== cheap check: original collocation set + shock-point count ===")
    set_seed(SEED)
    x_train, t_train = sample_training_points(M_TRAIN, device)
    n_shock = int((x_train.detach().abs() <= DELTA).sum().item())
    print(f"n_total={M_TRAIN}, n_shock (|x|<={DELTA}) = {n_shock}")

    print("\n=== cheap check: effective shock-weight fraction per alpha ===")
    for alpha in ALPHAS:
        weights = compute_weights(x_train, DELTA, alpha)
        frac = float(weights[x_train.detach().abs() <= DELTA].sum() / weights.sum())
        print(f"alpha={alpha:>3}: effective_shock_weight_fraction = {frac:.6f}")

    print("\n=== cheap check: alpha=1 backward-compatibility (weighted == unweighted) ===")
    model, _ = build_model(device)
    print(f"model.local_time_mode = {model.local_time_mode}")
    residual = burgers_residual(model, x_train, t_train, NV)
    standard_loss = torch.mean(residual**2)
    weights_alpha1 = compute_weights(x_train, DELTA, 1)
    weighted_loss_alpha1 = weighted_pde_loss(residual, weights_alpha1)
    abs_diff = float((standard_loss - weighted_loss_alpha1).abs().item())
    print(f"standard PDE loss           = {standard_loss.item():.15e}")
    print(f"weighted PDE loss (alpha=1) = {weighted_loss_alpha1.item():.15e}")
    print(f"absolute difference         = {abs_diff:.3e}")
    assert abs_diff < 1e-12, "FAILED: alpha=1 weighted loss does not match unweighted mean"
    print("PASSED (alpha=1 reduces exactly to unweighted mean)")

    print("\n=== cheap check: local_coefficients receive gradients ===")
    model2, optimizer2 = build_model(device)
    weights_alpha10 = compute_weights(x_train, DELTA, 10)
    optimizer2.zero_grad()
    total, pde_loss, ic_loss, bc_loss = loss_fn_weighted(
        model2, x_train, t_train, weights_alpha10, M_BD, M_INT, NV, g, device
    )
    total.backward()
    grad = model2.local_coefficients.grad
    print(f"local_coefficients.grad is not None: {grad is not None}")
    print(f"||grad(local_coefficients)|| = {grad.norm().item():.6e}")
    assert grad is not None and grad.norm().item() > 0, "FAILED: local_coefficients did not receive gradients"
    print("PASSED (local_coefficients receive nonzero gradients)")

    print("\n=== cheap check: same x_train/t_train tensor reused across alphas (by construction) ===")
    print(
        "x_train/t_train are sampled ONCE in main() before the alpha loop, and the SAME "
        "tensor objects are passed into every run_case() call -- never re-derived per alpha. "
        "build_model() only re-burns the RNG (for RF-init reproducibility) and discards that "
        "sample_training_points() output; it never overwrites x_train/t_train."
    )
    print(f"x_train first 3 values: {x_train.detach().flatten()[:3].tolist()}")
    print(f"t_train first 3 values: {t_train.detach().flatten()[:3].tolist()}")

    print("\nAll cheap checks passed.")


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

    # Sampled ONCE; the SAME x_train/t_train tensors are reused, unmodified,
    # for every alpha case below -- no enrichment/redistribution/resampling.
    set_seed(SEED)
    x_train, t_train = sample_training_points(M_TRAIN, device)
    n_shock = int((x_train.detach().abs() <= DELTA).sum().item())
    print(f"Original collocation set: n_total={M_TRAIN}, n_shock (|x|<={DELTA}) = {n_shock}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for alpha in ALPHAS:
        row = run_case(alpha, x_train, t_train, x_test, t_test, y_true, x_shock, t_shock, y_true_shock, x_ic, t_ic, y_ic_true, device)
        rows.append(row)
        write_summary(rows)

    plot_vs_alpha(rows)
    print("[shock_point_weighting] done.")


if __name__ == "__main__":
    if "--check" in sys.argv:
        cheap_checks()
    else:
        main()
