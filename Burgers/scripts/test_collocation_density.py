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
from sweep_local_width import SHOCK_N_X, SHOCK_X_HALF_WIDTH, build_shock_grid  # noqa: E402

# Controlled comparison: does collocation density near the shock explain the
# poor sigma=0.02 result? Both cases keep sigma=0.02, EPOCHS, M_TRAIN, SEED,
# RF architecture/sampling, optimizer, loss, and evaluation grids identical;
# only the spatial distribution of the M_TRAIN interior collocation points
# differs.
LOCAL_WIDTH = 0.02
SHOCK_FRACTION = 0.20
COUNT_MULTIPLES = (1, 2, 3)

OUTPUT_DIR = PROJECT_DIR / "outputs" / "collocation_study"


def sample_shock_enriched_training_points(
    m_train, device, shock_half_width=SHOCK_X_HALF_WIDTH, shock_fraction=SHOCK_FRACTION
):
    """
    Same total collocation count, and the same RNG call shapes/order as
    data.sample_training_points (one (m_train,1) draw for x, then one
    (m_train,1) draw for t): this guarantees the RNG state entering model
    construction is identical to the uniform case. Only the *values* of
    the x-draw are remapped afterward (a pure post-hoc transform, no extra
    RNG cost) so that shock_fraction of points fall in
    |x| <= shock_half_width and the rest fall in the complementary region
    [-1,-shock_half_width) U (shock_half_width,1], uniformly in each part.
    """
    n_shock = round(shock_fraction * m_train)

    u = torch.rand(m_train, 1)
    u_shock = u[:n_shock]
    u_smooth = u[n_shock:]

    x_shock = -shock_half_width + u_shock * (2 * shock_half_width)

    half_len = 1.0 - shock_half_width
    combined_len = 2.0 * half_len
    v = u_smooth * combined_len
    x_smooth = torch.where(v < half_len, -1.0 + v, v - 1.0 + 2.0 * shock_half_width)

    x_train = torch.cat([x_shock, x_smooth], dim=0)
    x_train = x_train.requires_grad_(True).to(device)

    t_train = torch.rand(m_train, 1, requires_grad=True)
    t_train = t_train.to(device)

    return x_train, t_train


def save_collocation_csv(case_name, x, t):
    csv_path = OUTPUT_DIR / f"collocation_{case_name}.csv"
    with csv_path.open("w") as f:
        f.write("x,t\n")
        for xv, tv in zip(x, t):
            f.write(f"{xv},{tv}\n")


def save_collocation_plot(case_name, x, t, mark_shock_boundary):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, t, s=4, alpha=0.5)
    if mark_shock_boundary:
        ax.axvline(-SHOCK_X_HALF_WIDTH, color="r", linestyle="--", linewidth=1, label=f"x=-{SHOCK_X_HALF_WIDTH}")
        ax.axvline(SHOCK_X_HALF_WIDTH, color="r", linestyle="--", linewidth=1, label=f"x=+{SHOCK_X_HALF_WIDTH}")
        ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"Interior collocation points ({case_name})")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"collocation_{case_name}.png")
    plt.close(fig)


def report_collocation_counts(x):
    counts = {}
    for mult in COUNT_MULTIPLES:
        r = mult * LOCAL_WIDTH
        n = int((np.abs(x) <= r).sum())
        counts[mult] = {"half_width": r, "count": n, "percentage": 100 * n / M_TRAIN}
    return counts


def write_case_summary(case_name, metrics, diag_stats, collocation_counts):
    path = OUTPUT_DIR / f"{case_name}_summary.txt"
    with path.open("w") as f:
        f.write(f"case: {case_name}\n")
        f.write(f"SEED: {SEED}\n")
        f.write(f"local_type: gaussian\n")
        f.write(f"LOCAL_CENTER: {LOCAL_CENTER}\n")
        f.write(f"LOCAL_WIDTH (sigma): {LOCAL_WIDTH}\n")
        f.write(f"M_TRAIN: {M_TRAIN}\n")
        f.write(f"EPOCHS: {EPOCHS}\n")
        f.write(f"N_TRIALS: {N_TRIALS}\n")
        if case_name == "shock_enriched":
            f.write(f"SHOCK_FRACTION: {SHOCK_FRACTION}\n")
        f.write(f"global_relative_l2_error: {metrics['global_relative_l2_error']}\n")
        f.write(f"smooth_relative_l2_error: {metrics['smooth_relative_l2_error']}\n")
        f.write(f"shock_relative_l2_error: {metrics['shock_relative_l2_error']}\n")
        f.write(f"local coefficient norm: {diag_stats['coef_norm']}\n")
        f.write(f"||u_local||_2 / ||u_RF||_2 (full grid): {diag_stats['ratio_full']}\n")
        f.write(f"||u_local||_2 / ||u_RF||_2 (shock region): {diag_stats['ratio_shock']}\n")
        f.write("collocation counts (|x| <= multiple * sigma):\n")
        for mult in COUNT_MULTIPLES:
            info = collocation_counts[mult]
            f.write(
                f"  {mult}*sigma (|x| <= {info['half_width']}): "
                f"count={info['count']}, percentage={info['percentage']:.4f}%\n"
            )


def run_case(case_name, sampler_fn, mark_shock_boundary, x_test, t_test, y_true, x_shock, t_shock, y_true_shock, device):
    # Reset the seed right before sampling/model construction so both cases
    # draw the same RF initialization and, downstream, the same per-epoch
    # boundary/initial-condition sampling sequence.
    set_seed(SEED)
    x_train, t_train = sampler_fn(M_TRAIN, device)

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
        model, case_name, x_test, t_test, y_true, x_shock, t_shock, y_true_shock, OUTPUT_DIR
    )

    x_np = x_train.detach().cpu().numpy().reshape(-1)
    t_np = t_train.detach().cpu().numpy().reshape(-1)
    save_collocation_csv(case_name, x_np, t_np)
    save_collocation_plot(case_name, x_np, t_np, mark_shock_boundary)
    collocation_counts = report_collocation_counts(x_np)

    write_case_summary(case_name, metrics, diag_stats, collocation_counts)

    return metrics


def main():
    torch.set_default_dtype(DTYPE)
    device = get_device()
    print(device)

    x_test, t_test = make_test_grid(M_TEST, N_TEST_TIMES, device)
    y_true = make_reference_values(t_test, x_test, NV)

    x_shock, t_shock = build_shock_grid(t_test, device)
    y_true_shock = make_reference_values(t_shock, x_shock, NV)

    cases = [
        ("uniform", sample_training_points, False),
        ("shock_enriched", sample_shock_enriched_training_points, True),
    ]

    results = []
    for case_name, sampler_fn, mark_shock_boundary in cases:
        metrics = run_case(
            case_name, sampler_fn, mark_shock_boundary,
            x_test, t_test, y_true, x_shock, t_shock, y_true_shock, device,
        )
        results.append((case_name, metrics))
        print(f"{case_name}: {metrics}")

    print(
        "\ncase                 global_relative_l2_error   "
        "smooth_relative_l2_error   shock_relative_l2_error"
    )
    for case_name, metrics in results:
        print(
            f"{case_name:<20} {metrics['global_relative_l2_error']:<26} "
            f"{metrics['smooth_relative_l2_error']:<26} {metrics['shock_relative_l2_error']}"
        )


if __name__ == "__main__":
    main()
