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
    EPOCHS,
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
from burgers_rf.physics import boundary_residual, burgers_residual, initial_residual, train  # noqa: E402
from burgers_rf.evaluation import evaluate_error_metrics  # noqa: E402
from sweep_local_width import build_shock_grid  # noqa: E402

# RF BASIS-SIZE SWEEP (TODO.md Task A, 2026-10-07 advisor meeting). Trains
# the canonical RF-only PDE baseline (use_local=False) at Nx in
# {500, 1000, 2000}; Nt, SIGMA_X, SIGMA_T, seed, collocation counts,
# optimizer, epochs, and IC_BC_WEIGHT are all held fixed at their config.py
# values -- isolating feature COUNT only, not RF frequency bandwidth.
# physics.train/loss_fn are reused completely unmodified; this script
# introduces NO new RNG scheme and does NOT modify physics.train() or any
# other package file.
#
# RNG NOTE (also written into summary.txt): set_seed(seed) ->
# sample_training_points(M_TRAIN, device) is called BEFORE BurgersRF(...)
# construction, so the interior collocation points x_train,t_train are
# BIT-IDENTICAL across all three Nx cases. However, BurgersRF.__init__
# draws an Nx-dependent number of random values for Wx/bx, so the global
# RNG stream's position after construction differs across Nx -- the
# per-epoch boundary_residual/initial_residual draws inside physics.train()
# therefore necessarily differ across Nx cases. This is an unavoidable
# consequence of Nx itself changing the model's RNG footprint, not a flaw
# in the procedure: every Nx case independently follows the exact canonical
# set_seed -> sample -> construct -> train sequence.
#
# CHECKPOINT NOTE: known bug (RESEARCH_LOG.md Section 16) -- on CUDA,
# nn.Parameter(...).to(device) inside BurgersRF.__init__ silently downgrades
# Wx/Wt/bx/bt to plain Tensors, so they can be ABSENT from
# model.state_dict(). This script does NOT rely on model.state_dict() for
# reconstruction at all: Wx, Wt, bx, bt, and the trained RF linear weight
# are read directly off the live model object's attributes (unaffected by
# the state_dict-registration bug) and saved as explicit, individually
# detached CPU tensors -- the single source of truth for reconstruction,
# requiring no RNG replay.

NX_VALUES = [500, 1000, 2000]

CANONICAL_GLOBAL = 0.27149430095802507
CANONICAL_SMOOTH = 0.2443827084159696
CANONICAL_SHOCK = 0.8390487998889169
VERIFY_RTOL = 1e-4

IC_GRID_N_X = 1000

OUTPUT_DIR = PROJECT_DIR / "outputs" / "rf_basis_size_sweep"

CSV_HEADER = [
    "Nx", "n_trainable_rf_coefficients",
    "global_relative_l2_error", "smooth_relative_l2_error", "shock_relative_l2_error",
    "ic_relative_l2_error", "final_pde_loss", "final_ic_loss", "final_bc_loss",
]


def relative_l2(y_true, y_pred):
    return float(np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true))


def build_complete_checkpoint(model, nx, metadata):
    """Explicit, detached-CPU-tensor checkpoint. Does NOT rely on
    model.state_dict() (which can silently omit Wx/Wt/bx/bt on CUDA --
    RESEARCH_LOG.md Section 16). Wx/Wt/bx/bt and the trained RF linear
    weight are read directly off the live model object and stored as the
    single source of truth for reconstruction; no RNG replay is needed."""
    return {
        "Nx": nx,
        "Nt": metadata["Nt"],
        "Wx": model.Wx.detach().cpu().clone(),
        "Wt": model.Wt.detach().cpu().clone(),
        "bx": model.bx.detach().cpu().clone(),
        "bt": model.bt.detach().cpu().clone(),
        "rf_linear_weight": model.model[0].weight.detach().cpu().clone(),
        "use_local": False,
        "SIGMA_X": metadata["SIGMA_X"],
        "SIGMA_T": metadata["SIGMA_T"],
        "NV": metadata["NV"],
        "DTYPE": metadata["DTYPE"],
        "seed": metadata["seed"],
        "M_TRAIN": metadata["M_TRAIN"],
        "M_BD": metadata["M_BD"],
        "M_INT": metadata["M_INT"],
        "EPOCHS": metadata["EPOCHS"],
        "LR": metadata["LR"],
        "WEIGHT_DECAY": metadata["WEIGHT_DECAY"],
        "global_relative_l2_error": metadata["global_relative_l2_error"],
        "smooth_relative_l2_error": metadata["smooth_relative_l2_error"],
        "shock_relative_l2_error": metadata["shock_relative_l2_error"],
        "ic_relative_l2_error": metadata["ic_relative_l2_error"],
        "final_pde_loss": metadata["final_pde_loss"],
        "final_ic_loss": metadata["final_ic_loss"],
        "final_bc_loss": metadata["final_bc_loss"],
    }


def reconstruct_model_from_checkpoint(checkpoint, device="cpu"):
    """Reconstruct the exact trained model directly from the checkpoint's
    explicit tensors -- NO RNG replay, NO reliance on model.state_dict().
    A throwaway BurgersRF is constructed only to obtain the right module
    structure/shapes; its own randomly-initialized Wx/Wt/bx/bt/linear
    weight are immediately overwritten with the saved tensors. `.to(device)`
    is applied to each plain saved Tensor BEFORE wrapping as nn.Parameter --
    the safe order that avoids the known Parameter-registration bug."""
    nx = checkpoint["Nx"]
    nt = checkpoint["Nt"]
    model = BurgersRF(
        nx, nt, checkpoint["SIGMA_X"], checkpoint["SIGMA_T"], use_local=False, device=device
    ).to(device)
    with torch.no_grad():
        model.Wx = torch.nn.Parameter(checkpoint["Wx"].to(device), requires_grad=False)
        model.Wt = torch.nn.Parameter(checkpoint["Wt"].to(device), requires_grad=False)
        model.bx = torch.nn.Parameter(checkpoint["bx"].to(device), requires_grad=False)
        model.bt = torch.nn.Parameter(checkpoint["bt"].to(device), requires_grad=False)
        model.model[0].weight.copy_(checkpoint["rf_linear_weight"].to(device))
    model.eval()
    return model


def run_one_nx(nx, device):
    set_seed(SEED)
    x_train, t_train = sample_training_points(M_TRAIN, device)

    print(f"=== Nx={nx} ===")
    model = BurgersRF(nx, NT, SIGMA_X, SIGMA_T, use_local=False, device=device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # physics.train / loss_fn reused completely unmodified.
    train(model, optimizer, x_train, t_train, M_BD, M_INT, NV, g, device, epochs=EPOCHS)

    # Post-hoc unweighted component losses, same established pattern as
    # test_ic_compatible_local_feature.py etc.
    residual = burgers_residual(model, x_train, t_train, NV)
    final_pde_loss = torch.mean(residual**2).item()
    residual_bd = boundary_residual(model, M_BD, device)
    final_bc_loss = torch.mean(residual_bd**2).item()
    residual_int = initial_residual(model, M_INT, g, device)
    final_ic_loss = torch.mean(residual_int**2).item()

    x_test, t_test = make_test_grid(M_TEST, N_TEST_TIMES, device)
    y_true = make_reference_values(t_test, x_test, NV)
    x_shock, t_shock = build_shock_grid(t_test, device)
    y_true_shock = make_reference_values(t_shock, x_shock, NV)
    y_pred = model(x_test, t_test).cpu().detach().numpy().reshape(-1)
    y_pred_shock = model(x_shock, t_shock).cpu().detach().numpy().reshape(-1)
    metrics = evaluate_error_metrics(x_test, y_true, y_pred, y_true_shock, y_pred_shock)

    x_ic = torch.linspace(-1, 1, IC_GRID_N_X).reshape(-1, 1).to(device)
    t_ic = torch.zeros_like(x_ic).to(device)
    y_ic_true = g(x_ic).detach().cpu().numpy().reshape(-1)
    y_ic_pred = model(x_ic, t_ic).cpu().detach().numpy().reshape(-1)
    ic_relative_l2 = relative_l2(y_ic_true, y_ic_pred)

    metadata = {
        "Nt": NT, "SIGMA_X": SIGMA_X, "SIGMA_T": SIGMA_T, "NV": NV, "DTYPE": str(DTYPE),
        "seed": SEED, "M_TRAIN": M_TRAIN, "M_BD": M_BD, "M_INT": M_INT,
        "EPOCHS": EPOCHS, "LR": LR, "WEIGHT_DECAY": WEIGHT_DECAY,
        "global_relative_l2_error": metrics["global_relative_l2_error"],
        "smooth_relative_l2_error": metrics["smooth_relative_l2_error"],
        "shock_relative_l2_error": metrics["shock_relative_l2_error"],
        "ic_relative_l2_error": ic_relative_l2,
        "final_pde_loss": final_pde_loss, "final_ic_loss": final_ic_loss, "final_bc_loss": final_bc_loss,
    }
    checkpoint = build_complete_checkpoint(model, nx, metadata)

    row = {
        "Nx": nx,
        "n_trainable_rf_coefficients": nx * NT,
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
        "global": rel_diff(row["global_relative_l2_error"], CANONICAL_GLOBAL),
        "smooth": rel_diff(row["smooth_relative_l2_error"], CANONICAL_SMOOTH),
        "shock": rel_diff(row["shock_relative_l2_error"], CANONICAL_SHOCK),
    }
    print("Nx=500 reproduction gate (vs. canonical RF-only baseline):")
    for k, v in checks.items():
        print(f"  {k}: relative difference = {v:.3e} (tolerance {VERIFY_RTOL:.1e})")
    failed = {k: v for k, v in checks.items() if v > VERIFY_RTOL}
    if failed:
        raise RuntimeError(
            f"Nx=500 REPRODUCTION GATE FAILED: {failed}. Refusing to run/interpret "
            "the Nx=1000/2000 sweep cases."
        )
    print("Reproduction gate PASSED.")
    return checks


def write_csv(rows):
    path = OUTPUT_DIR / "summary.csv"
    with path.open("w") as f:
        f.write(",".join(CSV_HEADER) + "\n")
        for row in rows:
            f.write(",".join(str(row[k]) for k in CSV_HEADER) + "\n")


def write_summary(rows, repro_checks):
    lines = []
    lines.append("RF basis-size sweep -- Nx in {500, 1000, 2000}")
    lines.append(
        f"Fixed: Nt={NT}, SIGMA_X={SIGMA_X}, SIGMA_T={SIGMA_T}, seed={SEED}, "
        f"M_TRAIN={M_TRAIN}, M_BD={M_BD}, M_INT={M_INT}, EPOCHS={EPOCHS}, "
        f"LR={LR}, WEIGHT_DECAY={WEIGHT_DECAY}"
    )
    lines.append("")
    lines.append(
        "RNG/reproducibility note: set_seed(seed) -> sample_training_points(...) is "
        "called BEFORE BurgersRF(...) construction, so the interior collocation points "
        "x_train,t_train are BIT-IDENTICAL across all three Nx cases. However, "
        "BurgersRF.__init__ draws an Nx-dependent number of random values for Wx/bx, "
        "so the global RNG stream's position after construction differs across Nx -- "
        "the per-epoch boundary_residual/initial_residual draws inside physics.train() "
        "therefore necessarily differ across Nx cases. This is an unavoidable "
        "consequence of Nx changing the model's own RNG footprint, not a flaw in the "
        "procedure; every Nx case independently follows the exact canonical "
        "set_seed -> sample -> construct -> train sequence. physics.train() itself was "
        "not modified and no new RNG scheme was introduced."
    )
    lines.append("")
    lines.append(
        "Checkpoint note: each saved checkpoint stores Wx, Wt, bx, bt, and the trained "
        "RF linear weight as EXPLICIT, individually detached CPU tensors read directly "
        "off the live model object -- NOT via model.state_dict(), which is known to "
        "silently omit Wx/Wt/bx/bt on CUDA (RESEARCH_LOG.md Section 16). Reconstruction "
        "(see reconstruct_model_from_checkpoint) requires NO RNG replay."
    )
    lines.append("")
    lines.append("Nx=500 reproduction gate:")
    for k, v in repro_checks.items():
        lines.append(f"  {k}: relative difference = {v:.3e} (tolerance {VERIFY_RTOL:.1e})")
    lines.append("Gate PASSED (script would have stopped otherwise).")
    lines.append("")
    lines.append("Per-Nx results:")
    for row in rows:
        lines.append(f"--- Nx = {row['Nx']} ---")
        for k in CSV_HEADER[1:]:
            lines.append(f"  {k} = {row[k]}")

    (OUTPUT_DIR / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def plot_error_vs_nx(rows):
    nx_values = [row["Nx"] for row in rows]
    global_errors = [row["global_relative_l2_error"] for row in rows]
    shock_errors = [row["shock_relative_l2_error"] for row in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(nx_values, global_errors, "o-", label="global relative L2")
    ax.plot(nx_values, shock_errors, "s-", label="shock relative L2")
    ax.set_xlabel("Nx")
    ax.set_ylabel("relative L2 error")
    ax.set_title("RF-only error vs Nx (Nt=10, all else fixed)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "error_vs_nx.png")
    plt.close(fig)


def print_table(rows):
    header = ["Nx", "n_coef", "global", "smooth", "shock", "IC", "pde_loss", "ic_loss", "bc_loss"]
    print(" | ".join(f"{h:>12}" for h in header))
    for row in rows:
        vals = [
            str(row["Nx"]), str(row["n_trainable_rf_coefficients"]),
            f"{row['global_relative_l2_error']:.6f}", f"{row['smooth_relative_l2_error']:.6f}",
            f"{row['shock_relative_l2_error']:.6f}", f"{row['ic_relative_l2_error']:.6f}",
            f"{row['final_pde_loss']:.6e}", f"{row['final_ic_loss']:.6e}", f"{row['final_bc_loss']:.6e}",
        ]
        print(" | ".join(f"{v:>12}" for v in vals))


def main():
    torch.set_default_dtype(DTYPE)
    device = get_device()
    print(device)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    repro_checks = None
    for i, nx in enumerate(NX_VALUES):
        row, checkpoint = run_one_nx(nx, device)

        if i == 0:
            repro_checks = verify_reproduction_gate(row)

        torch.save(checkpoint, OUTPUT_DIR / f"checkpoint_nx{nx}.pt")
        rows.append(row)
        write_csv(rows)

    write_summary(rows, repro_checks)
    plot_error_vs_nx(rows)
    print_table(rows)

    print("[sweep_rf_basis_size] done.")


if __name__ == "__main__":
    main()
