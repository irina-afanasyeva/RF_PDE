import copy
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(PROJECT_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from burgers_rf.config import (  # noqa: E402
    DTYPE, LR, M_BD, M_INT, M_TEST, M_TRAIN, N_TEST_TIMES, NT, NV, NX,
    SEED, SIGMA_T, SIGMA_X, WEIGHT_DECAY, get_device, set_seed,
)
from burgers_rf.data import g, make_reference_values, make_test_grid, sample_training_points  # noqa: E402
from burgers_rf.model import BurgersRF  # noqa: E402
from burgers_rf.physics import burgers_residual, train  # noqa: E402
from burgers_rf.evaluation import evaluate_error_metrics  # noqa: E402
from sweep_local_width import build_shock_grid  # noqa: E402
import sweep_rf_basis_size as sweep  # noqa: E402 -- reused unchanged: build_complete_checkpoint
import test_pde_trained_gaussian_correction as ptgc  # noqa: E402 -- reused unchanged: build_search_baseline
import test_fixed_gaussian_correction as tfgc  # noqa: E402 -- reused unchanged: solution_ylim
from test_rf_nx2000_long_training import boundary_residual_fixed, initial_residual_fixed  # noqa: E402

# EXPERIMENT A -- restore original training duration (2026-10-21 advisor
# meeting prep). Question: does the canonical Nx=500,Nt=10,seed=42 RF-only
# model improve substantially when training is continued from the existing
# 1000-epoch checkpoint out to 2000 and 5000 epochs (the ORIGINAL notebook's
# training duration, see RESEARCH_LOG.md)? BurgersRF construction and
# physics.train()/loss_fn are reused completely unmodified; no architecture
# change, no multiscale, no local Gaussian; collocation points are NOT
# changed (regenerated bit-identically via set_seed(42) ->
# sample_training_points, the SAME convention established in
# sweep_rf_random_seeds.py).
#
# CONTINUATION, NOT RESTART: loads the EXISTING canonical checkpoint
# (outputs/pde_trained_rf_residual/checkpoint_complete.pt) and continues
# training via repeated train() calls on the SAME model/optimizer objects
# -- same proven pattern as test_rf_nx2000_long_training.py.
#
# OPTIMIZER-STATE CAVEAT (explicit, do not hide): this project's established
# checkpoint format (build_complete_checkpoint / the original
# checkpoint_complete.pt) stores ONLY model_state_dict + metadata -- NO
# optimizer state was ever saved anywhere in this codebase. This
# continuation therefore uses a FRESH Adam optimizer (zero moment buffers),
# not a byte-identical resumption of the original optimizer trajectory.
# Results reflect "warm-started-weights + fresh-Adam" continuation, not a
# literal continuation of the unrecorded original run. Likewise, the
# per-epoch BC/IC random draws for epochs 1001-5000 are freshly drawn from
# the RNG state immediately after the set_seed(42)->sample_training_points
# call, NOT a replay of whatever the original (unrecorded) epochs 1-1000
# actually consumed -- the original run never saved intermediate RNG state,
# so this is unavoidable and does not change the scientific validity of
# "does more training help starting from these exact trained weights".
#
# DEVICE SAFETY: model is reconstructed with device="cpu" first (so the
# known nn.Parameter(...).to(device) registration bug, RESEARCH_LOG.md
# Section 16, cannot trigger during construction), then load_state_dict is
# guaranteed correct, then moved to the training device with an explicit
# defensive per-tensor re-pin of Wx/Wt/bx/bt (same established pattern as
# test_rf_nx2000_long_training.py / test_rf_multiscale_two_population.py /
# sweep_rf_random_seeds.py) so this is safe regardless of PyTorch version
# behavior for nn.Module.to() on already-registered parameters.

CANONICAL_GLOBAL = 0.27149430095802507
CANONICAL_SMOOTH = 0.2443827084159696
CANONICAL_SHOCK = 0.8390487998889169
VERIFY_RTOL = 1e-4

SOURCE_CHECKPOINT_PATH = PROJECT_DIR / "outputs" / "pde_trained_rf_residual" / "checkpoint_complete.pt"
EXPECTED_STATE_DICT_KEYS = {"Wx", "Wt", "bx", "bt", "model.0.weight"}

OUTPUT_DIR = PROJECT_DIR / "outputs" / "rf_nx500_long_training"
IC_GRID_N_X = 1000

CHECKPOINT_EPOCHS = [1000, 2000, 5000]
SEGMENTS = [(2000, 1000), (5000, 3000)]  # (target_epoch, additional_epochs)

CSV_HEADER = [
    "epoch",
    "global_relative_l2_error", "smooth_relative_l2_error", "shock_relative_l2_error",
    "ic_relative_l2_error", "final_pde_loss", "final_ic_loss", "final_bc_loss",
    "t1_global_relative_l2_error", "t1_smooth_relative_l2_error", "t1_shock_relative_l2_error",
]


def relative_l2(y_true, y_pred):
    return float(np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true))


def load_source_checkpoint():
    if not SOURCE_CHECKPOINT_PATH.exists():
        raise RuntimeError(
            f"Expected canonical seed-42 checkpoint not found at "
            f"{SOURCE_CHECKPOINT_PATH}. This script does NOT fall back to "
            "retraining from scratch -- transfer/produce it there first."
        )
    checkpoint = torch.load(SOURCE_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    state_dict_keys = set(checkpoint["model_state_dict"].keys())
    if state_dict_keys != EXPECTED_STATE_DICT_KEYS:
        raise RuntimeError(
            f"{SOURCE_CHECKPOINT_PATH} model_state_dict keys {sorted(state_dict_keys)} "
            f"do not match the expected complete set {sorted(EXPECTED_STATE_DICT_KEYS)}."
        )
    if checkpoint["seed"] != SEED or checkpoint["NX"] != NX or checkpoint["NT"] != NT:
        raise RuntimeError(
            f"{SOURCE_CHECKPOINT_PATH} metadata (seed={checkpoint['seed']}, "
            f"NX={checkpoint['NX']}, NT={checkpoint['NT']}) does not match the expected "
            f"canonical (seed={SEED}, NX={NX}, NT={NT})."
        )
    return checkpoint


def build_model_from_checkpoint(checkpoint, device):
    """CPU-only construction (guaranteed-correct Parameter registration,
    see module docstring) + load_state_dict, then move to the training
    device with a defensive explicit re-pin of Wx/Wt/bx/bt."""
    model = BurgersRF(
        checkpoint["NX"], checkpoint["NT"], checkpoint["SIGMA_X"], checkpoint["SIGMA_T"],
        use_local=checkpoint["use_local"], device="cpu",
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    for attr_name in ("Wx", "Wt", "bx", "bt"):
        value = getattr(model, attr_name).detach().to(device).clone()
        delattr(model, attr_name)
        setattr(model, attr_name, value)
    return model


def verify_gate(row):
    def rel_diff(a, b):
        return abs(a - b) / abs(b)

    checks = {
        "global": rel_diff(row["global_relative_l2_error"], CANONICAL_GLOBAL),
        "smooth": rel_diff(row["smooth_relative_l2_error"], CANONICAL_SMOOTH),
        "shock": rel_diff(row["shock_relative_l2_error"], CANONICAL_SHOCK),
    }
    print("Loaded-checkpoint (epoch=1000) reproduction gate (vs. canonical RF-only baseline):")
    for k, v in checks.items():
        print(f"  {k}: relative difference = {v:.3e} (tolerance {VERIFY_RTOL:.1e})")
    failed = {k: v for k, v in checks.items() if v > VERIFY_RTOL}
    if failed:
        raise RuntimeError(
            f"EPOCH-1000 GATE FAILED: {failed}. The loaded checkpoint does not reproduce "
            "the canonical reference -- refusing to continue training from it."
        )
    print("Reproduction gate PASSED.")


def evaluate_at_epoch(
    model, epoch, x_train, t_train, t_bd_fixed, x_int_fixed,
    x_test, t_test, y_true, x_shock, t_shock, y_true_shock,
    x_ic, t_ic, y_ic_true, device,
):
    """RNG-free diagnostics (fixed grids for BC/IC loss) -- safe to call
    mid-trajectory without perturbing subsequent training segments."""
    residual = burgers_residual(model, x_train, t_train, NV)
    final_pde_loss = torch.mean(residual**2).item()
    residual_bd = boundary_residual_fixed(model, t_bd_fixed, device)
    final_bc_loss = torch.mean(residual_bd**2).item()
    residual_int = initial_residual_fixed(model, x_int_fixed, g, device)
    final_ic_loss = torch.mean(residual_int**2).item()

    with torch.no_grad():
        y_pred = model(x_test, t_test).cpu().numpy().reshape(-1)
        y_pred_shock = model(x_shock, t_shock).cpu().numpy().reshape(-1)
    canonical_metrics = evaluate_error_metrics(x_test, y_true, y_pred, y_true_shock, y_pred_shock)

    with torch.no_grad():
        y_ic_pred = model(x_ic, t_ic).cpu().numpy().reshape(-1)
    ic_relative_l2 = relative_l2(y_ic_true, y_ic_pred)

    # t=1-only metrics via an independent CPU copy (established device-safe
    # pattern) -- the live `model` is never modified by this.
    cpu_model = copy.deepcopy(model).to("cpu")
    for attr_name in ("Wx", "Wt", "bx", "bt"):
        cpu_value = getattr(cpu_model, attr_name).detach().cpu().clone()
        delattr(cpu_model, attr_name)
        setattr(cpu_model, attr_name, cpu_value)
    baseline = ptgc.build_search_baseline(cpu_model, "cpu")
    t1_metrics = evaluate_error_metrics(
        baseline["x_eval_full"], baseline["y_eval_full"], baseline["u_RF_full"],
        baseline["y_eval_shock"], baseline["u_RF_shock"],
    )

    metadata = {
        "Nt": NT, "SIGMA_X": SIGMA_X, "SIGMA_T": SIGMA_T, "NV": NV, "DTYPE": str(DTYPE),
        "seed": SEED, "M_TRAIN": M_TRAIN, "M_BD": M_BD, "M_INT": M_INT,
        "EPOCHS": epoch, "LR": LR, "WEIGHT_DECAY": WEIGHT_DECAY,
        "global_relative_l2_error": canonical_metrics["global_relative_l2_error"],
        "smooth_relative_l2_error": canonical_metrics["smooth_relative_l2_error"],
        "shock_relative_l2_error": canonical_metrics["shock_relative_l2_error"],
        "ic_relative_l2_error": ic_relative_l2,
        "final_pde_loss": final_pde_loss, "final_ic_loss": final_ic_loss, "final_bc_loss": final_bc_loss,
    }
    checkpoint = sweep.build_complete_checkpoint(model, NX, metadata)
    checkpoint["epoch"] = epoch

    row = {
        "epoch": epoch,
        "global_relative_l2_error": canonical_metrics["global_relative_l2_error"],
        "smooth_relative_l2_error": canonical_metrics["smooth_relative_l2_error"],
        "shock_relative_l2_error": canonical_metrics["shock_relative_l2_error"],
        "ic_relative_l2_error": ic_relative_l2,
        "final_pde_loss": final_pde_loss, "final_ic_loss": final_ic_loss, "final_bc_loss": final_bc_loss,
        "t1_global_relative_l2_error": t1_metrics["global_relative_l2_error"],
        "t1_smooth_relative_l2_error": t1_metrics["smooth_relative_l2_error"],
        "t1_shock_relative_l2_error": t1_metrics["shock_relative_l2_error"],
    }
    return row, checkpoint, baseline


def write_csv(rows):
    path = OUTPUT_DIR / "summary.csv"
    with path.open("w") as f:
        f.write(",".join(CSV_HEADER) + "\n")
        for row in rows:
            f.write(",".join(str(row[k]) for k in CSV_HEADER) + "\n")


def write_summary(rows):
    lines = [
        "Experiment A -- restore original training duration (Nx=500, Nt=10, seed=42)",
        "Continuation from the existing canonical 1000-epoch checkpoint to 2000 and 5000 "
        "epochs. FRESH Adam optimizer (no original optimizer state was ever saved); "
        "collocation points regenerated bit-identically via set_seed(42) -> "
        "sample_training_points (unchanged from the original training set).",
        "",
    ]
    for row in rows:
        lines.append(f"--- epoch = {row['epoch']} ---")
        for k in CSV_HEADER[1:]:
            lines.append(f"  {k} = {row[k]}")
    (OUTPUT_DIR / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def plot_error_vs_epochs(rows):
    epochs = [row["epoch"] for row in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [r["global_relative_l2_error"] for r in rows], "o-", label="global relative L2")
    ax.plot(epochs, [r["shock_relative_l2_error"] for r in rows], "s-", label="shock relative L2")
    ax.set_xlabel("epoch")
    ax.set_ylabel("relative L2 error")
    ax.set_title("RF-only error vs epoch (Nx=500, Nt=10, seed=42, continuation)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "error_vs_epochs.png")
    plt.close(fig)


def plot_solution(epoch, baseline, suffix, key_x, key_y, key_u, metric_value, metric_name):
    x = baseline[key_x]
    y = baseline[key_y]
    u = baseline[key_u]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(x, y, "k-", linewidth=2, label="u_true")
    ax.plot(x, u, "--", label="u_RF")
    ax.set_ylim(tfgc.solution_ylim(y, u))
    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t=1)")
    ax.set_title(f"RF solution, Nx=500, epoch={epoch}, t=1\n{metric_name} = {metric_value:.6f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"solution_{suffix}_epoch{epoch}.png")
    plt.close(fig)


def print_table(rows):
    header = ["epoch", "global", "smooth", "shock", "IC", "pde_loss", "ic_loss", "bc_loss", "t1_global", "t1_smooth", "t1_shock"]
    print(" | ".join(f"{h:>11}" for h in header))
    for row in rows:
        vals = [
            str(row["epoch"]),
            f"{row['global_relative_l2_error']:.6f}", f"{row['smooth_relative_l2_error']:.6f}",
            f"{row['shock_relative_l2_error']:.6f}", f"{row['ic_relative_l2_error']:.6f}",
            f"{row['final_pde_loss']:.6e}", f"{row['final_ic_loss']:.6e}", f"{row['final_bc_loss']:.6e}",
            f"{row['t1_global_relative_l2_error']:.6f}", f"{row['t1_smooth_relative_l2_error']:.6f}",
            f"{row['t1_shock_relative_l2_error']:.6f}",
        ]
        print(" | ".join(f"{v:>11}" for v in vals))


def main():
    torch.set_default_dtype(DTYPE)
    device = get_device()
    print(device)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = load_source_checkpoint()
    model = build_model_from_checkpoint(checkpoint, device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Bit-identical reproduction of the original training points (NOT the
    # mid-training RNG state -- see module docstring's optimizer-state
    # caveat). Collocation points are NOT changed from the canonical set.
    set_seed(SEED)
    x_train, t_train = sample_training_points(M_TRAIN, device)

    t_bd_fixed = torch.linspace(0, 1, M_BD).reshape(-1, 1).to(device)
    x_int_fixed = torch.linspace(-1, 1, M_INT).reshape(-1, 1).to(device)
    x_test, t_test = make_test_grid(M_TEST, N_TEST_TIMES, device)
    y_true = make_reference_values(t_test, x_test, NV)
    x_shock, t_shock = build_shock_grid(t_test, device)
    y_true_shock = make_reference_values(t_shock, x_shock, NV)
    x_ic = torch.linspace(-1, 1, IC_GRID_N_X).reshape(-1, 1).to(device)
    t_ic = torch.zeros_like(x_ic).to(device)
    y_ic_true = g(x_ic).detach().cpu().numpy().reshape(-1)

    eval_args = (x_train, t_train, t_bd_fixed, x_int_fixed, x_test, t_test, y_true,
                 x_shock, t_shock, y_true_shock, x_ic, t_ic, y_ic_true, device)

    rows = []
    row1000, ckpt1000, baseline1000 = evaluate_at_epoch(model, 1000, *eval_args)
    verify_gate(row1000)
    rows.append(row1000)
    torch.save(ckpt1000, OUTPUT_DIR / "checkpoint_epoch1000.pt")
    plot_solution(1000, baseline1000, "full", "x_eval_full", "y_eval_full", "u_RF_full",
                  row1000["t1_global_relative_l2_error"], "t=1 global relative L2")
    plot_solution(1000, baseline1000, "shock", "x_eval_shock", "y_eval_shock", "u_RF_shock",
                  row1000["t1_shock_relative_l2_error"], "t=1 shock relative L2")

    cumulative_epoch = 1000
    for target_epoch, segment_len in SEGMENTS:
        print(f"=== training segment: epoch {cumulative_epoch} -> {target_epoch} ({segment_len} epochs) ===")
        train(model, optimizer, x_train, t_train, M_BD, M_INT, NV, g, device, epochs=segment_len)
        cumulative_epoch += segment_len
        assert cumulative_epoch == target_epoch

        row, ckpt, baseline = evaluate_at_epoch(model, target_epoch, *eval_args)
        rows.append(row)
        torch.save(ckpt, OUTPUT_DIR / f"checkpoint_epoch{target_epoch}.pt")
        plot_solution(target_epoch, baseline, "full", "x_eval_full", "y_eval_full", "u_RF_full",
                      row["t1_global_relative_l2_error"], "t=1 global relative L2")
        plot_solution(target_epoch, baseline, "shock", "x_eval_shock", "y_eval_shock", "u_RF_shock",
                      row["t1_shock_relative_l2_error"], "t=1 shock relative L2")

    write_csv(rows)
    write_summary(rows)
    plot_error_vs_epochs(rows)
    print_table(rows)

    print("[test_rf_nx500_long_training] done.")


if __name__ == "__main__":
    main()
