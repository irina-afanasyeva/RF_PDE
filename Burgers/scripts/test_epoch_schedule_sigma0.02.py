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
from sweep_local_width import SHOCK_X_HALF_WIDTH, build_shock_grid  # noqa: E402

# One continuous 10000-epoch training trajectory for sigma=0.02, using the
# ORIGINAL uniform collocation setup only (no enrichment/refinement).
# set_seed(SEED) is called exactly once; x_train/t_train, model, and
# optimizer are each constructed exactly once; train() (unmodified) is then
# called repeatedly on the SAME model/optimizer objects to advance the same
# trajectory in segments, evaluating at each cumulative checkpoint.
LOCAL_WIDTH = 0.02
CHECKPOINTS = [1000, 2500, 5000, 7500, 10000]
DELTAS = [1000, 1500, 2500, 2500, 2500]
assert [sum(DELTAS[: i + 1]) for i in range(len(DELTAS))] == CHECKPOINTS

OUTPUT_DIR = PROJECT_DIR / "outputs" / "epoch_study"


def print_and_verify_config(device, x_train):
    assert torch.get_default_dtype() == DTYPE, "default dtype must match config.DTYPE"
    assert M_TRAIN == 3600, "this experiment assumes the original M_TRAIN=3600 setup"

    x_np = x_train.detach().cpu().numpy().reshape(-1)
    shock_count = int((np.abs(x_np) <= SHOCK_X_HALF_WIDTH).sum())

    print("=== Configuration ===")
    print(f"dtype: {torch.get_default_dtype()}")
    print(f"device: {device}")
    print(f"SEED: {SEED}")
    print(f"M_TRAIN: {M_TRAIN}")
    print(f"LOCAL_TYPE: gaussian")
    print(f"LOCAL_CENTER: {LOCAL_CENTER}")
    print(f"LOCAL_WIDTH (sigma): {LOCAL_WIDTH}")
    print(f"checkpoints: {CHECKPOINTS}")
    print(f"segment deltas: {DELTAS}")
    print(f"actual collocation count with |x| <= {SHOCK_X_HALF_WIDTH}: {shock_count}")
    print("=====================")


def write_metrics_csv(rows):
    path = OUTPUT_DIR / "epoch_metrics.csv"
    header = [
        "epoch",
        "global_relative_l2_error",
        "smooth_relative_l2_error",
        "shock_relative_l2_error",
        "local_coef_norm",
        "ratio_full",
        "ratio_shock",
    ]
    with path.open("w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(row[key]) for key in header) + "\n")


def write_loss_history(all_losses):
    path = OUTPUT_DIR / "loss_history.csv"
    with path.open("w") as f:
        f.write("epoch,loss\n")
        for i, loss_val in enumerate(all_losses, start=1):
            f.write(f"{i},{loss_val}\n")


def plot_loss_history(all_losses):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(range(1, len(all_losses) + 1), all_losses)
    ax.set_yscale("log")
    ax.set_xlabel("cumulative epoch")
    ax.set_ylabel("loss (log scale)")
    ax.set_title("Loss history, sigma=0.02, continuous 10000-epoch trajectory")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "loss_history.png")
    plt.close(fig)


def save_checkpoint(model, optimizer, cumulative_epoch):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "cumulative_epoch": cumulative_epoch,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    path = OUTPUT_DIR / f"checkpoint_epoch_{cumulative_epoch}.pt"
    torch.save(checkpoint, path)
    print(f"[epoch_schedule] saved checkpoint: {path}")


def main():
    torch.set_default_dtype(DTYPE)
    device = get_device()
    print(device)

    x_test, t_test = make_test_grid(M_TEST, N_TEST_TIMES, device)
    y_true = make_reference_values(t_test, x_test, NV)

    x_shock, t_shock = build_shock_grid(t_test, device)
    y_true_shock = make_reference_values(t_shock, x_shock, NV)

    # set_seed is called exactly once, before any sampling/model construction.
    set_seed(SEED)
    x_train, t_train = sample_training_points(M_TRAIN, device)

    print_and_verify_config(device, x_train)

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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_losses = []
    metric_rows = []
    prev_epoch = 0

    for target_epoch, delta in zip(CHECKPOINTS, DELTAS):
        print(f"[epoch_schedule] starting segment: cumulative epoch {prev_epoch} -> {target_epoch} (delta={delta})")
        losses = train(model, optimizer, x_train, t_train, M_BD, M_INT, NV, g, device, epochs=delta)
        all_losses.extend(losses)
        print(f"[epoch_schedule] completed cumulative epoch {target_epoch}")

        y_pred = model(x_test, t_test).cpu().detach().numpy().reshape(-1)
        y_pred_shock = model(x_shock, t_shock).cpu().detach().numpy().reshape(-1)
        metrics = evaluate_error_metrics(x_test, y_true, y_pred, y_true_shock, y_pred_shock)

        case_name = f"sigma0.02_epoch{target_epoch}"
        diag_stats = save_component_diagnostics(
            model, case_name, x_test, t_test, y_true, x_shock, t_shock, y_true_shock, OUTPUT_DIR
        )

        metric_rows.append(
            {
                "epoch": target_epoch,
                "global_relative_l2_error": metrics["global_relative_l2_error"],
                "smooth_relative_l2_error": metrics["smooth_relative_l2_error"],
                "shock_relative_l2_error": metrics["shock_relative_l2_error"],
                "local_coef_norm": diag_stats["coef_norm"],
                "ratio_full": diag_stats["ratio_full"],
                "ratio_shock": diag_stats["ratio_shock"],
            }
        )
        write_metrics_csv(metric_rows)
        write_loss_history(all_losses)
        plot_loss_history(all_losses)
        save_checkpoint(model, optimizer, target_epoch)

        print(f"[epoch_schedule] epoch {target_epoch} metrics: {metric_rows[-1]}")

        prev_epoch = target_epoch

    print("[epoch_schedule] done.")


if __name__ == "__main__":
    main()
