import argparse
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
    DTYPE,
    LR,
    M_BD,
    M_INT,
    M_TEST,
    M_TRAIN,
    N_TEST_TIMES,
    NT,
    NV,
    SEED,
    SIGMA_T,
    SIGMA_X,
    WEIGHT_DECAY,
    get_device,
    set_seed,
)
from burgers_rf.data import g, make_reference_values, make_test_grid, sample_training_points  # noqa: E402
from burgers_rf.model import BurgersRF  # noqa: E402
from burgers_rf.physics import burgers_residual, train  # noqa: E402
from burgers_rf.evaluation import evaluate_error_metrics  # noqa: E402
from sweep_local_width import build_shock_grid  # noqa: E402
import sweep_rf_basis_size as sweep  # noqa: E402 -- reused unchanged: build_complete_checkpoint
import test_pde_trained_gaussian_correction as ptgc  # noqa: E402 -- reused unchanged: build_search_baseline
import test_fixed_gaussian_correction as tfgc  # noqa: E402 -- reused unchanged: solution_ylim

# LONG-TRAINING DIAGNOSTIC for Nx=2000 (TODO follow-up after the RF
# basis-size sweep showed degradation at Nx=2000). Tests whether the poor
# Nx=2000 result is explained by insufficient training duration, by running
# ONE continuous training trajectory from initialization through 10,000
# epochs and checkpointing at 1000/2000/5000/10000.
#
# Nx, Nt, seed, collocation counts, optimizer, and the physics/loss
# formulation are all IDENTICAL to the Nx=2000 case in
# sweep_rf_basis_size.py; physics.train()/loss_fn are reused completely
# unmodified; no new RNG scheme is introduced anywhere in this script.
#
# CONTINUOUS TRAJECTORY: physics.train() has no resume/callback mechanism
# (confirmed by inspection -- it is a plain `for epoch in range(epochs)`
# loop over the SAME model/optimizer objects passed in, with no other
# state). Rather than reimplementing the training loop, this script calls
# train() FOUR TIMES on the SAME model and SAME optimizer object, with
# segment lengths that sum to each checkpoint target (1000, then +1000,
# then +3000, then +5000). Adam's internal per-parameter step/moment state
# is a property of the optimizer object and is correctly preserved across
# separate .step()-call sequences regardless of which Python-level train()
# invocation they occur in -- this is mathematically IDENTICAL to one
# uninterrupted 10,000-epoch call, PROVIDED nothing between the four calls
# consumes any random numbers (see next paragraph). This was verified safe
# without needing to redesign or reimplement physics.train() itself.
#
# RNG SAFETY BETWEEN SEGMENTS: physics.boundary_residual/initial_residual
# are the ONLY functions in the training loop that consume random numbers
# (fresh torch.rand draws every call). All intermediate-checkpoint
# diagnostics in this script are deliberately RNG-FREE:
#   - PDE loss: burgers_residual(model, x_train, t_train, NV) operates on
#     the fixed training interior points -- no RNG.
#   - BC/IC loss: computed via boundary_residual_fixed/initial_residual_fixed
#     (defined below), which mirror physics.boundary_residual/
#     initial_residual's exact formulas but take FIXED, pre-built
#     (torch.linspace-based, non-random) grids instead of freshly sampling
#     m_bd/m_int random points every call -- consuming ZERO random numbers.
#   - canonical global/smooth/shock metrics and IC relative L2: all built
#     from deterministic torch.linspace/cartesian_prod grids (same
#     convention as every other script in this project) -- no RNG.
#   - checkpoint saving and solution plotting: pure I/O / deterministic
#     forward passes -- no RNG.
# Therefore the global RNG stream entering each subsequent train() segment
# is EXACTLY where it would be in an uninterrupted single run, and the
# resulting trained weights at every checkpoint are bit-identical to what a
# single continuous call would have produced.

# NX, OUTPUT_DIR, REPRO_GLOBAL/SMOOTH/SHOCK are set from --nx in main() (see
# parse_args()/REPRO_REFERENCE below); the values here are only the
# Nx=2000 defaults used if the script is imported without main() running.
NX = 2000
CHECKPOINT_EPOCHS = [1000, 2000, 5000, 10000]
SEGMENT_LENGTHS = [CHECKPOINT_EPOCHS[0]] + [
    CHECKPOINT_EPOCHS[i] - CHECKPOINT_EPOCHS[i - 1] for i in range(1, len(CHECKPOINT_EPOCHS))
]

REPRO_EPOCH = 1000
# 1000-epoch reproduction-gate reference, per Nx (from the existing
# sweep_rf_basis_size.py sweep results).
REPRO_REFERENCE = {
    1000: {"global": 0.2839157305655459, "smooth": 0.25354926454918303, "shock": 0.9181808292859333},
    2000: {"global": 0.3160344409079877, "smooth": 0.27944891957843243, "shock": 1.0849605011988503},
}
REPRO_GLOBAL = REPRO_REFERENCE[NX]["global"]
REPRO_SMOOTH = REPRO_REFERENCE[NX]["smooth"]
REPRO_SHOCK = REPRO_REFERENCE[NX]["shock"]
VERIFY_RTOL = 1e-4

IC_GRID_N_X = 1000

OUTPUT_DIR = PROJECT_DIR / "outputs" / f"rf_nx{NX}_long_training"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Nx long-training diagnostic (continuous trajectory, checkpoints at "
        "1000/2000/5000/10000 epochs)."
    )
    parser.add_argument("--nx", type=int, choices=[1000, 2000], default=2000)
    return parser.parse_args()


CSV_HEADER = [
    "epoch",
    "global_relative_l2_error", "smooth_relative_l2_error", "shock_relative_l2_error",
    "ic_relative_l2_error", "final_pde_loss", "final_ic_loss", "final_bc_loss",
]


def relative_l2(y_true, y_pred):
    return float(np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true))


def boundary_residual_fixed(model, t_bd_fixed, device):
    """Deterministic analog of physics.boundary_residual: mirrors its exact
    formula, but takes a FIXED (pre-built, non-random) set of boundary
    times instead of freshly sampling m_bd random points every call.
    Consumes ZERO random numbers -- safe to call between training segments
    without perturbing the RNG stream used by subsequent training."""
    x_left = -torch.ones_like(t_bd_fixed).to(device)
    x_right = torch.ones_like(t_bd_fixed).to(device)
    u_bd1 = model(x_left, t_bd_fixed)
    u_bd2 = model(x_right, t_bd_fixed)
    return torch.cat((u_bd1, u_bd2))


def initial_residual_fixed(model, x_int_fixed, g_fn, device):
    """Deterministic analog of physics.initial_residual: mirrors its exact
    formula, but takes a FIXED (pre-built, non-random) set of interior
    x-points instead of freshly sampling m_int random points every call.
    Consumes ZERO random numbers."""
    t_int_fixed = torch.zeros_like(x_int_fixed).to(device)
    u_int = model(x_int_fixed, t_int_fixed)
    return u_int - g_fn(x_int_fixed).reshape(u_int.shape)


def evaluate_at_checkpoint(
    model, epoch, x_train, t_train, t_bd_fixed, x_int_fixed,
    x_test, t_test, y_true, x_shock, t_shock, y_true_shock,
    x_ic, t_ic, y_ic_true, device,
):
    """All operations here are deterministic (see module docstring) -- this
    function consumes NO random numbers and is safe to call mid-trajectory
    without perturbing subsequent training."""
    residual = burgers_residual(model, x_train, t_train, NV)
    final_pde_loss = torch.mean(residual**2).item()
    residual_bd = boundary_residual_fixed(model, t_bd_fixed, device)
    final_bc_loss = torch.mean(residual_bd**2).item()
    residual_int = initial_residual_fixed(model, x_int_fixed, g, device)
    final_ic_loss = torch.mean(residual_int**2).item()

    with torch.no_grad():
        y_pred = model(x_test, t_test).cpu().numpy().reshape(-1)
        y_pred_shock = model(x_shock, t_shock).cpu().numpy().reshape(-1)
    metrics = evaluate_error_metrics(x_test, y_true, y_pred, y_true_shock, y_pred_shock)

    with torch.no_grad():
        y_ic_pred = model(x_ic, t_ic).cpu().numpy().reshape(-1)
    ic_relative_l2 = relative_l2(y_ic_true, y_ic_pred)

    metadata = {
        "Nt": NT, "SIGMA_X": SIGMA_X, "SIGMA_T": SIGMA_T, "NV": NV, "DTYPE": str(DTYPE),
        "seed": SEED, "M_TRAIN": M_TRAIN, "M_BD": M_BD, "M_INT": M_INT,
        "EPOCHS": epoch, "LR": LR, "WEIGHT_DECAY": WEIGHT_DECAY,
        "global_relative_l2_error": metrics["global_relative_l2_error"],
        "smooth_relative_l2_error": metrics["smooth_relative_l2_error"],
        "shock_relative_l2_error": metrics["shock_relative_l2_error"],
        "ic_relative_l2_error": ic_relative_l2,
        "final_pde_loss": final_pde_loss, "final_ic_loss": final_ic_loss, "final_bc_loss": final_bc_loss,
    }
    checkpoint = sweep.build_complete_checkpoint(model, NX, metadata)
    checkpoint["epoch"] = epoch

    row = {
        "epoch": epoch,
        "global_relative_l2_error": metrics["global_relative_l2_error"],
        "smooth_relative_l2_error": metrics["smooth_relative_l2_error"],
        "shock_relative_l2_error": metrics["shock_relative_l2_error"],
        "ic_relative_l2_error": ic_relative_l2,
        "final_pde_loss": final_pde_loss,
        "final_ic_loss": final_ic_loss,
        "final_bc_loss": final_bc_loss,
    }
    return row, checkpoint


def verify_reproduction_gate(row):
    def rel_diff(a, b):
        return abs(a - b) / abs(b)

    checks = {
        "global": rel_diff(row["global_relative_l2_error"], REPRO_GLOBAL),
        "smooth": rel_diff(row["smooth_relative_l2_error"], REPRO_SMOOTH),
        "shock": rel_diff(row["shock_relative_l2_error"], REPRO_SHOCK),
    }
    print(f"epoch={REPRO_EPOCH} reproduction gate (vs. existing Nx={NX} sweep reference):")
    for k, v in checks.items():
        print(f"  {k}: relative difference = {v:.3e} (tolerance {VERIFY_RTOL:.1e})")
    failed = {k: v for k, v in checks.items() if v > VERIFY_RTOL}
    if failed:
        raise RuntimeError(
            f"epoch={REPRO_EPOCH} REPRODUCTION GATE FAILED: {failed}. Refusing to continue "
            "training past 1000 epochs."
        )
    print("Reproduction gate PASSED.")


def write_csv(rows):
    path = OUTPUT_DIR / "summary.csv"
    with path.open("w") as f:
        f.write(",".join(CSV_HEADER) + "\n")
        for row in rows:
            f.write(",".join(str(row[k]) for k in CSV_HEADER) + "\n")


def write_summary(rows):
    lines = []
    lines.append(f"Nx={NX} long-training diagnostic -- checkpoints at {CHECKPOINT_EPOCHS}")
    lines.append(
        "One continuous training trajectory: the SAME model and optimizer objects "
        "are reused across all four train() segments (no restart). Intermediate "
        "diagnostics (PDE/IC/BC loss, canonical metrics, IC relative L2) are all "
        "computed via deterministic, RNG-free operations -- fixed torch.linspace-"
        "based grids replace physics.boundary_residual/initial_residual's random "
        "sampling for the loss diagnostics specifically, so no random numbers are "
        "consumed between segments and the trained weights at each checkpoint are "
        "bit-identical to an uninterrupted single run."
    )
    lines.append("")
    lines.append(f"epoch={REPRO_EPOCH} reproduction gate: PASSED (script would have stopped otherwise).")
    lines.append("")
    lines.append("Per-checkpoint results:")
    for row in rows:
        lines.append(f"--- epoch = {row['epoch']} ---")
        for k in CSV_HEADER[1:]:
            lines.append(f"  {k} = {row[k]}")
    (OUTPUT_DIR / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def plot_error_vs_epochs(rows):
    epochs = [row["epoch"] for row in rows]
    global_errors = [row["global_relative_l2_error"] for row in rows]
    shock_errors = [row["shock_relative_l2_error"] for row in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, global_errors, "o-", label="global relative L2")
    ax.plot(epochs, shock_errors, "s-", label="shock relative L2")
    ax.set_xlabel("epoch")
    ax.set_ylabel("relative L2 error")
    ax.set_title(f"RF-only error vs epoch (Nx={NX}, Nt={NT})")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "error_vs_epochs.png")
    plt.close(fig)


def plot_solution_full(epoch, baseline, t1_global_error):
    x_full = baseline["x_eval_full"]
    y_full = baseline["y_eval_full"]
    u_rf_full = baseline["u_RF_full"]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(x_full, y_full, "k-", linewidth=2, label="u_true")
    ax.plot(x_full, u_rf_full, "--", label="u_RF")
    ax.set_ylim(tfgc.solution_ylim(y_full, u_rf_full))
    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t=1)")
    ax.set_title(f"RF solution, Nx={NX}, epoch={epoch}, t=1\nGlobal relative L2 = {t1_global_error:.6f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"solution_full_epoch{epoch}.png")
    plt.close(fig)


def plot_solution_shock(epoch, baseline, t1_shock_error):
    x_shock = baseline["x_eval_shock"]
    y_shock = baseline["y_eval_shock"]
    u_rf_shock = baseline["u_RF_shock"]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(x_shock, y_shock, "k-", linewidth=2, label="u_true")
    ax.plot(x_shock, u_rf_shock, "--", label="u_RF")
    ax.set_ylim(tfgc.solution_ylim(y_shock, u_rf_shock))
    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t=1)")
    ax.set_title(f"RF shock region, Nx={NX}, epoch={epoch}, t=1\nShock relative L2 = {t1_shock_error:.6f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"solution_shock_epoch{epoch}.png")
    plt.close(fig)


def print_table(rows):
    header = ["epoch", "global", "smooth", "shock", "IC", "pde_loss", "ic_loss", "bc_loss"]
    print(" | ".join(f"{h:>12}" for h in header))
    for row in rows:
        vals = [
            str(row["epoch"]),
            f"{row['global_relative_l2_error']:.6f}", f"{row['smooth_relative_l2_error']:.6f}",
            f"{row['shock_relative_l2_error']:.6f}", f"{row['ic_relative_l2_error']:.6f}",
            f"{row['final_pde_loss']:.6e}", f"{row['final_ic_loss']:.6e}", f"{row['final_bc_loss']:.6e}",
        ]
        print(" | ".join(f"{v:>12}" for v in vals))


def main():
    args = parse_args()

    # Rebind the module-level NX/OUTPUT_DIR/REPRO_* globals from --nx, before
    # anything else runs. All other functions in this script read these as
    # module globals (looked up at call time), so this is sufficient to
    # retarget the whole script to the requested Nx without threading an
    # `nx` parameter through every function signature.
    global NX, OUTPUT_DIR, REPRO_GLOBAL, REPRO_SMOOTH, REPRO_SHOCK
    NX = args.nx
    OUTPUT_DIR = PROJECT_DIR / "outputs" / f"rf_nx{NX}_long_training"
    REPRO_GLOBAL = REPRO_REFERENCE[NX]["global"]
    REPRO_SMOOTH = REPRO_REFERENCE[NX]["smooth"]
    REPRO_SHOCK = REPRO_REFERENCE[NX]["shock"]

    # Must be set before any grid is built or model is constructed -- see
    # RESEARCH_LOG.md's dtype lesson from plot_rf_basis_size_solutions.py.
    torch.set_default_dtype(DTYPE)
    device = get_device()
    print(device)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Identical initialization/RNG procedure to the Nx=2000 case in
    # sweep_rf_basis_size.py.
    set_seed(SEED)
    x_train, t_train = sample_training_points(M_TRAIN, device)

    model = BurgersRF(NX, NT, SIGMA_X, SIGMA_T, use_local=False, device=device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Fixed (non-random) BC/IC evaluation grids, built ONCE, reused at every
    # checkpoint -- consume zero RNG, so they cannot perturb the training
    # RNG stream regardless of when they are constructed or evaluated.
    t_bd_fixed = torch.linspace(0, 1, M_BD).reshape(-1, 1).to(device)
    x_int_fixed = torch.linspace(-1, 1, M_INT).reshape(-1, 1).to(device)

    # Canonical multi-time evaluation grids (deterministic, built once).
    x_test, t_test = make_test_grid(M_TEST, N_TEST_TIMES, device)
    y_true = make_reference_values(t_test, x_test, NV)
    x_shock, t_shock = build_shock_grid(t_test, device)
    y_true_shock = make_reference_values(t_shock, x_shock, NV)

    x_ic = torch.linspace(-1, 1, IC_GRID_N_X).reshape(-1, 1).to(device)
    t_ic = torch.zeros_like(x_ic).to(device)
    y_ic_true = g(x_ic).detach().cpu().numpy().reshape(-1)

    rows = []
    cumulative_epoch = 0
    for target_epoch, segment_len in zip(CHECKPOINT_EPOCHS, SEGMENT_LENGTHS):
        print(f"=== training segment: epoch {cumulative_epoch} -> {target_epoch} "
              f"({segment_len} epochs) ===")
        # physics.train/loss_fn reused completely unmodified. Calling train()
        # again on the SAME model/optimizer objects continues the exact same
        # trajectory -- Adam's internal state and the model's parameters
        # persist across calls; this is not a restart.
        train(model, optimizer, x_train, t_train, M_BD, M_INT, NV, g, device, epochs=segment_len)
        cumulative_epoch += segment_len
        assert cumulative_epoch == target_epoch

        row, checkpoint = evaluate_at_checkpoint(
            model, target_epoch, x_train, t_train, t_bd_fixed, x_int_fixed,
            x_test, t_test, y_true, x_shock, t_shock, y_true_shock,
            x_ic, t_ic, y_ic_true, device,
        )

        if target_epoch == REPRO_EPOCH:
            verify_reproduction_gate(row)

        torch.save(checkpoint, OUTPUT_DIR / f"checkpoint_epoch{target_epoch}.pt")
        rows.append(row)
        write_csv(rows)

        baseline = ptgc.build_search_baseline(model, device)
        t1_metrics = evaluate_error_metrics(
            baseline["x_eval_full"], baseline["y_eval_full"], baseline["u_RF_full"],
            baseline["y_eval_shock"], baseline["u_RF_shock"],
        )
        plot_solution_full(target_epoch, baseline, t1_metrics["global_relative_l2_error"])
        plot_solution_shock(target_epoch, baseline, t1_metrics["shock_relative_l2_error"])

    write_summary(rows)
    plot_error_vs_epochs(rows)
    print_table(rows)

    print("[test_rf_nx2000_long_training] done.")


if __name__ == "__main__":
    main()
