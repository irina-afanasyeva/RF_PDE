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
from burgers_rf.physics import train  # noqa: E402
from burgers_rf.evaluation import evaluate_error_metrics  # noqa: E402
from burgers_rf.diagnostics import save_component_diagnostics  # noqa: E402
from sweep_local_width import SHOCK_N_X, build_shock_grid  # noqa: E402
from sweep_local_width_small_sigma import (  # noqa: E402
    FINE_SHOCK_N_X,
    build_fine_shock_grid,
    compute_components_numpy,
)
from sweep_local_width_small_sigma_resolved import (  # noqa: E402
    N_TARGET_LOCAL,
    sample_resolution_refined_points,
)

# "Best-case / fair chance" experiment for a narrow Gaussian local feature:
# combines (1) sigma-scaled collocation enrichment (reusing the exact,
# already-verified mechanism from sweep_local_width_small_sigma_resolved.py)
# with (2) one continuous 10000-epoch training trajectory per sigma (reusing
# the checkpointing strategy from test_epoch_schedule_sigma0.02.py). This
# intentionally conflates local resolution with residual weighting (the loss
# stays mean(residual**2), unmodified) -- it does not isolate the two.
SIGMAS = [1e-4, 5e-4, 2.5e-3]
CHECKPOINT_EPOCHS = [1000, 2500, 5000, 7500, 10000]
DELTAS = [1000, 1500, 2500, 2500, 2500]
assert [sum(DELTAS[: i + 1]) for i in range(len(DELTAS))] == CHECKPOINT_EPOCHS

# Already-verified expected values (from sweep_local_width_small_sigma_resolved.py).
EXPECTED_COLLOCATION = {
    1e-4: {
        "original_in_sigma": 0,
        "original_in_3sigma": 1,
        "added": 149,
        "final_in_sigma": 54,
        "final_in_3sigma": 150,
        "total": 3749,
    },
    5e-4: {
        "original_in_sigma": 2,
        "original_in_3sigma": 6,
        "added": 144,
        "final_in_sigma": 51,
        "final_in_3sigma": 150,
        "total": 3744,
    },
    2.5e-3: {
        "original_in_sigma": 11,
        "original_in_3sigma": 35,
        "added": 115,
        "final_in_sigma": 52,
        "final_in_3sigma": 150,
        "total": 3715,
    },
}

OUTPUT_DIR = PROJECT_DIR / "outputs" / "small_sigma_resolved_long_training"

PRIOR_OUTPUT_DIRS = [
    PROJECT_DIR / "outputs" / "baseline",
    PROJECT_DIR / "outputs" / "gaussian",
    PROJECT_DIR / "outputs" / "gaussian_sum",
    PROJECT_DIR / "outputs" / "diagnostics",
    PROJECT_DIR / "outputs" / "collocation_study",
    PROJECT_DIR / "outputs" / "epoch_study",
    PROJECT_DIR / "outputs" / "small_sigma_sweep",
    PROJECT_DIR / "outputs" / "small_sigma_sweep_resolved",
]


def verify_setup(x_shock, x_fine):
    assert torch.get_default_dtype() == DTYPE, "dtype must match config.DTYPE (float64)"

    shock_x_unique = torch.unique(x_shock.detach().cpu()).numel()
    assert shock_x_unique == SHOCK_N_X, (
        f"canonical shock grid must remain unchanged: expected {SHOCK_N_X} x-points, got {shock_x_unique}"
    )

    fine_x_unique = torch.unique(x_fine.detach().cpu())
    assert fine_x_unique.numel() == FINE_SHOCK_N_X, (
        f"fine grid must have {FINE_SHOCK_N_X} x-points, got {fine_x_unique.numel()}"
    )
    assert float(fine_x_unique.abs().min()) < 1e-12, "fine grid must include x=0"

    assert OUTPUT_DIR not in PRIOR_OUTPUT_DIRS, "output directory must not collide with any previous experiment"
    assert OUTPUT_DIR.resolve() not in [p.resolve() for p in PRIOR_OUTPUT_DIRS], (
        "output directory must be distinct from every previous experiment directory"
    )

    print(
        f"Setup verification passed: dtype={torch.get_default_dtype()}, "
        f"canonical shock grid unchanged ({SHOCK_N_X} points), "
        f"fine grid has {FINE_SHOCK_N_X} points including x=0, "
        f"output_dir={OUTPUT_DIR}"
    )


def verify_collocation_counts(sigma, counts, final_in_sigma, final_in_3sigma, final_total):
    expected = EXPECTED_COLLOCATION[sigma]
    assert counts["original_count_in_sigma"] == expected["original_in_sigma"], (
        f"sigma={sigma}: original_count_in_sigma mismatch: "
        f"got {counts['original_count_in_sigma']}, expected {expected['original_in_sigma']}"
    )
    assert counts["original_count_in_3sigma"] == expected["original_in_3sigma"], (
        f"sigma={sigma}: original_count_in_3sigma mismatch: "
        f"got {counts['original_count_in_3sigma']}, expected {expected['original_in_3sigma']}"
    )
    assert counts["n_added"] == expected["added"], (
        f"sigma={sigma}: n_added mismatch: got {counts['n_added']}, expected {expected['added']}"
    )
    assert final_in_sigma == expected["final_in_sigma"], (
        f"sigma={sigma}: final_count_in_sigma mismatch: got {final_in_sigma}, expected {expected['final_in_sigma']}"
    )
    assert final_in_3sigma == expected["final_in_3sigma"] == N_TARGET_LOCAL, (
        f"sigma={sigma}: final_count_in_3sigma mismatch: got {final_in_3sigma}, expected {expected['final_in_3sigma']}"
    )
    assert final_total == expected["total"], (
        f"sigma={sigma}: final_total_count mismatch: got {final_total}, expected {expected['total']}"
    )
    print(f"Collocation verification passed for sigma={sigma}: matches already-verified expected values.")


def save_fine_shock_plot(case_name, x_fine, t_fine, y_true_fine, u_total, u_rf, u_local):
    """Re-implemented locally (not imported) so its output goes to THIS
    experiment's OUTPUT_DIR."""
    t_np = t_fine.detach().cpu().numpy().reshape(-1)
    n_times = len(np.unique(t_np))
    n_x = len(t_np) // n_times

    x2 = x_fine.detach().cpu().numpy().reshape(n_times, n_x)
    t2 = t_np.reshape(n_times, n_x)
    y_true2 = y_true_fine.reshape(n_times, n_x)
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

    fig.suptitle(f"{case_name}: fine shock-region components (N_x={n_x})")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"components_{case_name}_fine_shock.png")
    plt.close(fig)


def save_checkpoint(model, optimizer, sigma, cumulative_epoch):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "sigma": sigma,
        "cumulative_epoch": cumulative_epoch,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    path = OUTPUT_DIR / f"checkpoint_sigma{sigma}_epoch_{cumulative_epoch}.pt"
    torch.save(checkpoint, path)
    print(f"[small_sigma_resolved_long_training] saved checkpoint: {path}")


HEADER = [
    "sigma",
    "cumulative_epoch",
    "global_relative_l2_error",
    "smooth_relative_l2_error",
    "shock_relative_l2_error",
    "local_coef_norm",
    "ratio_full",
    "ratio_shock",
    "ratio_shock_fine",
    "original_count_in_sigma",
    "original_count_in_3sigma",
    "n_added",
    "final_count_in_sigma",
    "final_count_in_3sigma",
    "final_total_count",
]


def write_master_summary(rows):
    csv_path = OUTPUT_DIR / "small_sigma_resolved_long_training_summary.csv"
    with csv_path.open("w") as f:
        f.write(",".join(HEADER) + "\n")
        for row in rows:
            f.write(",".join(str(row[key]) for key in HEADER) + "\n")

    txt_path = OUTPUT_DIR / "small_sigma_resolved_long_training_summary.txt"
    with txt_path.open("w") as f:
        for row in rows:
            f.write(f"sigma={row['sigma']}, cumulative_epoch={row['cumulative_epoch']}\n")
            for key in HEADER[2:]:
                f.write(f"  {key} = {row[key]}\n")
            f.write("\n")


def plot_sigma_trajectory(sigma, sigma_rows):
    epochs = [r["cumulative_epoch"] for r in sigma_rows]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, [r["global_relative_l2_error"] for r in sigma_rows], marker="o", label="global")
    ax.plot(epochs, [r["smooth_relative_l2_error"] for r in sigma_rows], marker="o", label="smooth")
    ax.plot(epochs, [r["shock_relative_l2_error"] for r in sigma_rows], marker="o", label="shock")
    ax.set_xlabel("cumulative epoch")
    ax.set_ylabel("relative L2 error")
    ax.set_title(f"sigma={sigma}: error vs. cumulative epoch")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"errors_vs_epoch_sigma{sigma}.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, [r["local_coef_norm"] for r in sigma_rows], marker="o", color="tab:purple")
    ax.set_xlabel("cumulative epoch")
    ax.set_ylabel("||local_coefficients||_2")
    ax.set_title(f"sigma={sigma}: local coefficient norm vs. cumulative epoch")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"coef_norm_vs_epoch_sigma{sigma}.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, [r["ratio_shock"] for r in sigma_rows], marker="o", label="ratio_shock (canonical grid)")
    ax.plot(epochs, [r["ratio_shock_fine"] for r in sigma_rows], marker="o", label="ratio_shock_fine (2001-pt grid)")
    ax.set_yscale("log")
    ax.set_xlabel("cumulative epoch")
    ax.set_ylabel("||u_local||_2 / ||u_RF||_2 (log scale)")
    ax.set_title(f"sigma={sigma}: local contribution ratio vs. cumulative epoch")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"ratio_vs_epoch_sigma{sigma}.png")
    plt.close(fig)


def run_sigma(sigma, x_test, t_test, y_true, x_shock, t_shock, y_true_shock, x_fine, t_fine, y_true_fine, device, all_rows):
    # A. Reproduce the exact resolved collocation construction.
    set_seed(SEED)
    x_orig, t_orig = sample_training_points(M_TRAIN, device)
    x_train, t_train, counts = sample_resolution_refined_points(x_orig, t_orig, sigma, N_TARGET_LOCAL, device)

    assert torch.equal(x_train[:M_TRAIN], x_orig), "first M_TRAIN x-points must match the original sample"
    assert torch.equal(t_train[:M_TRAIN], t_orig), "first M_TRAIN t-points must match the original sample"

    x_train_np = x_train.detach().cpu().numpy().reshape(-1)
    final_in_sigma = int((np.abs(x_train_np) <= sigma).sum())
    final_in_3sigma = int((np.abs(x_train_np) <= 3 * sigma).sum())
    final_total = x_train.shape[0]

    verify_collocation_counts(sigma, counts, final_in_sigma, final_in_3sigma, final_total)

    print(f"=== sigma={sigma} ===")
    print(f"dtype: {torch.get_default_dtype()}")
    print(
        f"original_in_sigma={counts['original_count_in_sigma']}, "
        f"original_in_3sigma={counts['original_count_in_3sigma']}, n_added={counts['n_added']}"
    )
    print(f"final_in_sigma={final_in_sigma}, final_in_3sigma={final_in_3sigma}, final_total={final_total}")

    # B. Construct one fresh model + optimizer for this sigma.
    model = BurgersRF(
        NX,
        NT,
        SIGMA_X,
        SIGMA_T,
        use_local=True,
        local_type="gaussian",
        local_center=LOCAL_CENTER,
        local_width=sigma,
        device=device,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sigma_rows = []
    prev_epoch = 0

    # C. One continuous trajectory to 10000 epochs, checkpointed.
    for target_epoch, delta in zip(CHECKPOINT_EPOCHS, DELTAS):
        print(
            f"[small_sigma_resolved_long_training] sigma={sigma}: "
            f"segment {prev_epoch} -> {target_epoch} (delta={delta})"
        )
        train(model, optimizer, x_train, t_train, M_BD, M_INT, NV, g, device, epochs=delta)
        print(f"[small_sigma_resolved_long_training] sigma={sigma}: completed cumulative epoch {target_epoch}")

        # D. Evaluate at this checkpoint.
        y_pred = model(x_test, t_test).cpu().detach().numpy().reshape(-1)
        y_pred_shock = model(x_shock, t_shock).cpu().detach().numpy().reshape(-1)
        metrics = evaluate_error_metrics(x_test, y_true, y_pred, y_true_shock, y_pred_shock)

        u_total_fine, u_rf_fine, u_local_fine = compute_components_numpy(model, x_fine, t_fine)
        ratio_shock_fine = float(np.linalg.norm(u_local_fine) / np.linalg.norm(u_rf_fine))

        coef_norm = float(torch.linalg.norm(model.local_coefficients).item())
        u_total_full, u_rf_full, u_local_full = compute_components_numpy(model, x_test, t_test)
        u_total_shock, u_rf_shock, u_local_shock = compute_components_numpy(model, x_shock, t_shock)
        ratio_full = float(np.linalg.norm(u_local_full) / np.linalg.norm(u_rf_full))
        ratio_shock = float(np.linalg.norm(u_local_shock) / np.linalg.norm(u_rf_shock))

        row = {
            "sigma": sigma,
            "cumulative_epoch": target_epoch,
            "global_relative_l2_error": metrics["global_relative_l2_error"],
            "smooth_relative_l2_error": metrics["smooth_relative_l2_error"],
            "shock_relative_l2_error": metrics["shock_relative_l2_error"],
            "local_coef_norm": coef_norm,
            "ratio_full": ratio_full,
            "ratio_shock": ratio_shock,
            "ratio_shock_fine": ratio_shock_fine,
            "original_count_in_sigma": counts["original_count_in_sigma"],
            "original_count_in_3sigma": counts["original_count_in_3sigma"],
            "n_added": counts["n_added"],
            "final_count_in_sigma": final_in_sigma,
            "final_count_in_3sigma": final_in_3sigma,
            "final_total_count": final_total,
        }
        sigma_rows.append(row)
        all_rows.append(row)
        write_master_summary(all_rows)

        # E. Safety checkpoint at every milestone.
        save_checkpoint(model, optimizer, sigma, target_epoch)

        # F. Component diagnostics only at the final checkpoint.
        if target_epoch == CHECKPOINT_EPOCHS[-1]:
            case_name = f"sigma{sigma}_resolved_epoch{target_epoch}"
            save_component_diagnostics(
                model, case_name, x_test, t_test, y_true, x_shock, t_shock, y_true_shock, OUTPUT_DIR
            )
            save_fine_shock_plot(case_name, x_fine, t_fine, y_true_fine, u_total_fine, u_rf_fine, u_local_fine)

        print(f"[small_sigma_resolved_long_training] sigma={sigma} epoch={target_epoch} metrics: {row}")
        prev_epoch = target_epoch

    plot_sigma_trajectory(sigma, sigma_rows)


def main():
    torch.set_default_dtype(DTYPE)
    device = get_device()
    print(device)

    x_test, t_test = make_test_grid(M_TEST, N_TEST_TIMES, device)
    y_true = make_reference_values(t_test, x_test, NV)

    x_shock, t_shock = build_shock_grid(t_test, device)
    y_true_shock = make_reference_values(t_shock, x_shock, NV)

    x_fine, t_fine = build_fine_shock_grid(t_test, device)
    y_true_fine = make_reference_values(t_fine, x_fine, NV)

    verify_setup(x_shock, x_fine)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for sigma in SIGMAS:
        run_sigma(
            sigma, x_test, t_test, y_true, x_shock, t_shock, y_true_shock, x_fine, t_fine, y_true_fine, device,
            all_rows,
        )

    print("[small_sigma_resolved_long_training] done.")


if __name__ == "__main__":
    main()
