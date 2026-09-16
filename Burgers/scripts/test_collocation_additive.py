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
from burgers_rf.evaluation import evaluate_error_metrics  # noqa: E402
from burgers_rf.diagnostics import save_component_diagnostics  # noqa: E402
from sweep_local_width import SHOCK_X_HALF_WIDTH, build_shock_grid  # noqa: E402

# CASE C: additive/refined shock oversampling. Preserves the exact original
# SEED=42 uniform 3600-point collocation set untouched, and ADDS 643 extra
# shock-region points (drawn from a separate torch.Generator so the global
# RNG stream feeding RF initialization and per-epoch BC/IC sampling is left
# completely unperturbed) so the final shock-region count matches CASE B's
# 720 without removing any smooth-region coverage.
CASE_NAME = "additive_shock_refined"
LOCAL_WIDTH = 0.02
N_ADDED_SHOCK = 643

OUTPUT_DIR = PROJECT_DIR / "outputs" / "collocation_study"


def sample_additive_shock_points(x_orig, t_orig, n_added, shock_half_width, device):
    """
    Draws n_added shock-region points from a SEPARATE torch.Generator, so
    these draws do not advance/perturb the global default RNG stream that
    RF initialization and per-epoch BC/IC sampling rely on. Returns the
    concatenation of the original points followed by the added points.
    """
    pre_state = torch.get_rng_state()

    refine_generator = torch.Generator()
    refine_generator.manual_seed(SEED)

    x_added = -shock_half_width + torch.rand(n_added, 1, generator=refine_generator) * (2 * shock_half_width)
    t_added = torch.rand(n_added, 1, generator=refine_generator)
    x_added = x_added.to(device)
    t_added = t_added.to(device)

    post_state = torch.get_rng_state()
    assert torch.equal(pre_state, post_state), (
        "Refinement sampling must not perturb the global RNG state, but it did"
    )

    x_train = torch.cat([x_orig, x_added], dim=0)
    t_train = torch.cat([t_orig, t_added], dim=0)
    return x_train, t_train


def save_collocation_csv(x, t):
    csv_path = OUTPUT_DIR / f"collocation_{CASE_NAME}.csv"
    with csv_path.open("w") as f:
        f.write("x,t\n")
        for xv, tv in zip(x, t):
            f.write(f"{xv},{tv}\n")


def save_collocation_plot(x, t):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, t, s=4, alpha=0.5)
    ax.axvline(-SHOCK_X_HALF_WIDTH, color="r", linestyle="--", linewidth=1, label=f"x=-{SHOCK_X_HALF_WIDTH}")
    ax.axvline(SHOCK_X_HALF_WIDTH, color="r", linestyle="--", linewidth=1, label=f"x=+{SHOCK_X_HALF_WIDTH}")
    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"Interior collocation points ({CASE_NAME})")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"collocation_{CASE_NAME}.png")
    plt.close(fig)


def write_summary(metrics, diag_stats, total_points, shock_points):
    path = OUTPUT_DIR / f"{CASE_NAME}_summary.txt"
    with path.open("w") as f:
        f.write(f"case: {CASE_NAME}\n")
        f.write(f"SEED: {SEED}\n")
        f.write(f"local_type: gaussian\n")
        f.write(f"LOCAL_CENTER: {LOCAL_CENTER}\n")
        f.write(f"LOCAL_WIDTH (sigma): {LOCAL_WIDTH}\n")
        f.write(f"EPOCHS: {EPOCHS}\n")
        f.write(f"N_TRIALS: {N_TRIALS}\n")
        f.write(f"original_uniform_points: {M_TRAIN}\n")
        f.write(f"added_shock_points: {N_ADDED_SHOCK}\n")
        f.write(f"total_points: {total_points}\n")
        f.write(f"shock_points: {shock_points}\n")
        f.write(f"shock_fraction: {shock_points / total_points}\n")
        f.write(f"global_relative_l2_error: {metrics['global_relative_l2_error']}\n")
        f.write(f"smooth_relative_l2_error: {metrics['smooth_relative_l2_error']}\n")
        f.write(f"shock_relative_l2_error: {metrics['shock_relative_l2_error']}\n")
        f.write(f"local coefficient norm: {diag_stats['coef_norm']}\n")
        f.write(f"||u_local||_2 / ||u_RF||_2 (full grid): {diag_stats['ratio_full']}\n")
        f.write(f"||u_local||_2 / ||u_RF||_2 (shock region): {diag_stats['ratio_shock']}\n")


def main():
    torch.set_default_dtype(DTYPE)
    device = get_device()
    print(device)

    x_test, t_test = make_test_grid(M_TEST, N_TEST_TIMES, device)
    y_true = make_reference_values(t_test, x_test, NV)

    x_shock, t_shock = build_shock_grid(t_test, device)
    y_true_shock = make_reference_values(t_shock, x_shock, NV)

    # Reset the seed, then reproduce the EXACT original uniform collocation
    # sample via the unmodified sample_training_points, before drawing any
    # refinement points.
    set_seed(SEED)
    x_orig, t_orig = sample_training_points(M_TRAIN, device)

    x_train, t_train = sample_additive_shock_points(
        x_orig, t_orig, N_ADDED_SHOCK, SHOCK_X_HALF_WIDTH, device
    )

    # Cheap checks before training.
    assert torch.equal(x_train[:M_TRAIN], x_orig), "first M_TRAIN x-points must match the original sample"
    assert torch.equal(t_train[:M_TRAIN], t_orig), "first M_TRAIN t-points must match the original sample"
    total_points = x_train.shape[0]
    assert total_points == M_TRAIN + N_ADDED_SHOCK

    model = BurgersRF(
        NX,
        NT,
        SIGMA_X,
        SIGMA_T,
        use_local=True,
        local_type="gaussian",
        local_center=LOCAL_CENTER,
        local_width=LOCAL_WIDTH,
        device=device,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    train(model, optimizer, x_train, t_train, M_BD, M_INT, NV, g, device, epochs=EPOCHS)

    y_pred = model(x_test, t_test).cpu().detach().numpy().reshape(-1)
    y_pred_shock = model(x_shock, t_shock).cpu().detach().numpy().reshape(-1)

    metrics = evaluate_error_metrics(x_test, y_true, y_pred, y_true_shock, y_pred_shock)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    diag_stats = save_component_diagnostics(
        model, CASE_NAME, x_test, t_test, y_true, x_shock, t_shock, y_true_shock, OUTPUT_DIR
    )

    x_np = x_train.detach().cpu().numpy().reshape(-1)
    t_np = t_train.detach().cpu().numpy().reshape(-1)
    save_collocation_csv(x_np, t_np)
    save_collocation_plot(x_np, t_np)

    shock_points = int((np.abs(x_np) <= SHOCK_X_HALF_WIDTH).sum())
    write_summary(metrics, diag_stats, total_points, shock_points)

    print(f"total_points={total_points}, shock_points={shock_points} ({100*shock_points/total_points:.4f}%)")
    print(metrics)
    print(diag_stats)


if __name__ == "__main__":
    main()
