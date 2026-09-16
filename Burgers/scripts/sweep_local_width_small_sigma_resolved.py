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

# Resolution-controlled follow-up to the unrefined small-sigma sweep. The
# EXACT original 3600 uniform collocation points (SEED=42) are preserved
# untouched; just enough additional points are drawn (via a separate,
# global-RNG-isolated torch.Generator) inside |x| <= 3*sigma so the final
# count there equals N_TARGET_LOCAL. This changes BOTH local resolution AND
# the implicit weighting of that region in mean(residual**2) -- no
# quadrature/sample weighting is introduced here; that is left to a
# separate follow-up if needed.
SIGMAS = [1e-4, 5e-4, 2.5e-3, 1.25e-2]
N_TARGET_LOCAL = 150

OUTPUT_DIR = PROJECT_DIR / "outputs" / "small_sigma_sweep_resolved"


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

    assert OUTPUT_DIR != PROJECT_DIR / "outputs" / "small_sigma_sweep", (
        "output directory must not collide with the unrefined small-sigma sweep"
    )

    print(
        f"Setup verification passed: dtype={torch.get_default_dtype()}, "
        f"canonical shock grid unchanged ({SHOCK_N_X} points), "
        f"fine grid has {FINE_SHOCK_N_X} points including x=0, "
        f"output_dir={OUTPUT_DIR}"
    )


def sample_resolution_refined_points(x_orig, t_orig, sigma, n_target_local, device):
    """
    Preserve the exact original collocation points; add just enough
    uniformly-sampled points inside |x| <= 3*sigma (t ~ Uniform(0,1)) so the
    final count inside that band equals n_target_local. Uses a SEPARATE
    torch.Generator so the added draws do not advance/perturb the global
    default RNG stream that RF initialization and per-epoch BC/IC sampling
    rely on.
    """
    half_width = 3 * sigma
    x_orig_np = x_orig.detach().cpu().numpy().reshape(-1)
    original_count_in_sigma = int((np.abs(x_orig_np) <= sigma).sum())
    original_count_in_3sigma = int((np.abs(x_orig_np) <= half_width).sum())
    n_add = n_target_local - original_count_in_3sigma
    assert n_add >= 0, (
        f"sigma={sigma}: already {original_count_in_3sigma} points in 3*sigma, "
        f">= target {n_target_local}; additive design assumes n_add >= 0"
    )

    pre_state = torch.get_rng_state()

    refine_generator = torch.Generator()
    refine_generator.manual_seed(SEED)

    x_added = -half_width + torch.rand(n_add, 1, generator=refine_generator) * (2 * half_width)
    t_added = torch.rand(n_add, 1, generator=refine_generator)
    x_added = x_added.to(device)
    t_added = t_added.to(device)

    post_state = torch.get_rng_state()
    assert torch.equal(pre_state, post_state), (
        "Refinement sampling must not perturb the global RNG state, but it did"
    )

    x_train = torch.cat([x_orig, x_added], dim=0)
    t_train = torch.cat([t_orig, t_added], dim=0)

    counts = {
        "original_count_in_sigma": original_count_in_sigma,
        "original_count_in_3sigma": original_count_in_3sigma,
        "n_added": n_add,
    }
    return x_train, t_train, counts


def save_collocation_csv(case_name, x, t):
    csv_path = OUTPUT_DIR / f"collocation_{case_name}.csv"
    with csv_path.open("w") as f:
        f.write("x,t\n")
        for xv, tv in zip(x, t):
            f.write(f"{xv},{tv}\n")


def save_collocation_plot(case_name, x, t, half_width):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, t, s=4, alpha=0.5)
    ax.axvline(-half_width, color="r", linestyle="--", linewidth=1, label=f"x=-3*sigma={-half_width:.4g}")
    ax.axvline(half_width, color="r", linestyle="--", linewidth=1, label=f"x=+3*sigma={half_width:.4g}")
    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"Interior collocation points ({case_name})")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"collocation_{case_name}.png")
    plt.close(fig)


def save_fine_shock_plot(case_name, x_fine, t_fine, y_true_fine, u_total, u_rf, u_local):
    """Re-implemented locally (not imported) so its output goes to THIS
    experiment's OUTPUT_DIR, not sweep_local_width_small_sigma's."""
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


def write_summary(rows):
    header = [
        "sigma",
        "original_collocation_count",
        "original_count_in_sigma",
        "original_count_in_3sigma",
        "n_added",
        "final_total_count",
        "final_count_in_sigma",
        "final_count_in_3sigma",
        "global_relative_l2_error",
        "smooth_relative_l2_error",
        "shock_relative_l2_error",
        "local_coef_norm",
        "ratio_full",
        "ratio_shock",
        "ratio_shock_fine",
    ]
    csv_path = OUTPUT_DIR / "small_sigma_sweep_resolved_summary.csv"
    with csv_path.open("w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(row[key]) for key in header) + "\n")

    txt_path = OUTPUT_DIR / "small_sigma_sweep_resolved_summary.txt"
    with txt_path.open("w") as f:
        for row in rows:
            f.write(f"sigma = {row['sigma']}\n")
            for key in header[1:]:
                f.write(f"  {key} = {row[key]}\n")
            f.write("\n")


def run_case(sigma, x_test, t_test, y_true, x_shock, t_shock, y_true_shock, x_fine, t_fine, y_true_fine, device):
    # Independent controlled run: reseed, resample the exact original set.
    set_seed(SEED)
    x_orig, t_orig = sample_training_points(M_TRAIN, device)

    x_train, t_train, counts = sample_resolution_refined_points(x_orig, t_orig, sigma, N_TARGET_LOCAL, device)

    # Cheap checks before training.
    assert torch.equal(x_train[:M_TRAIN], x_orig), "first M_TRAIN x-points must match the original sample"
    assert torch.equal(t_train[:M_TRAIN], t_orig), "first M_TRAIN t-points must match the original sample"
    total_points = x_train.shape[0]
    assert total_points == M_TRAIN + counts["n_added"]

    half_width = 3 * sigma
    x_train_np = x_train.detach().cpu().numpy().reshape(-1)
    t_train_np = t_train.detach().cpu().numpy().reshape(-1)
    final_count_in_sigma = int((np.abs(x_train_np) <= sigma).sum())
    final_count_in_3sigma = int((np.abs(x_train_np) <= half_width).sum())
    assert final_count_in_3sigma == N_TARGET_LOCAL, (
        f"sigma={sigma}: expected final count in 3*sigma == {N_TARGET_LOCAL}, got {final_count_in_3sigma}"
    )

    case_name = f"sigma{sigma}_resolved"
    print(f"=== {case_name} ===")
    print(f"dtype: {torch.get_default_dtype()}")
    print(
        f"original_count_in_sigma={counts['original_count_in_sigma']}, "
        f"original_count_in_3sigma={counts['original_count_in_3sigma']}, n_added={counts['n_added']}"
    )
    print(
        f"final_count_in_sigma={final_count_in_sigma}, final_count_in_3sigma={final_count_in_3sigma}, "
        f"final_total_count={total_points}"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_collocation_csv(case_name, x_train_np, t_train_np)
    save_collocation_plot(case_name, x_train_np, t_train_np, half_width)

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

    train(model, optimizer, x_train, t_train, M_BD, M_INT, NV, g, device, epochs=EPOCHS)

    y_pred = model(x_test, t_test).cpu().detach().numpy().reshape(-1)
    y_pred_shock = model(x_shock, t_shock).cpu().detach().numpy().reshape(-1)
    metrics = evaluate_error_metrics(x_test, y_true, y_pred, y_true_shock, y_pred_shock)

    diag_stats = save_component_diagnostics(
        model, case_name, x_test, t_test, y_true, x_shock, t_shock, y_true_shock, OUTPUT_DIR
    )

    u_total_fine, u_rf_fine, u_local_fine = compute_components_numpy(model, x_fine, t_fine)
    ratio_shock_fine = float(np.linalg.norm(u_local_fine) / np.linalg.norm(u_rf_fine))
    save_fine_shock_plot(case_name, x_fine, t_fine, y_true_fine, u_total_fine, u_rf_fine, u_local_fine)

    return {
        "sigma": sigma,
        "original_collocation_count": M_TRAIN,
        "original_count_in_sigma": counts["original_count_in_sigma"],
        "original_count_in_3sigma": counts["original_count_in_3sigma"],
        "n_added": counts["n_added"],
        "final_total_count": total_points,
        "final_count_in_sigma": final_count_in_sigma,
        "final_count_in_3sigma": final_count_in_3sigma,
        "global_relative_l2_error": metrics["global_relative_l2_error"],
        "smooth_relative_l2_error": metrics["smooth_relative_l2_error"],
        "shock_relative_l2_error": metrics["shock_relative_l2_error"],
        "local_coef_norm": diag_stats["coef_norm"],
        "ratio_full": diag_stats["ratio_full"],
        "ratio_shock": diag_stats["ratio_shock"],
        "ratio_shock_fine": ratio_shock_fine,
    }


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

    rows = []
    for sigma in SIGMAS:
        row = run_case(
            sigma, x_test, t_test, y_true, x_shock, t_shock, y_true_shock, x_fine, t_fine, y_true_fine, device
        )
        rows.append(row)
        write_summary(rows)
        print(f"sigma={sigma} done: {row}")

    print("[small_sigma_sweep_resolved] done.")


if __name__ == "__main__":
    main()
