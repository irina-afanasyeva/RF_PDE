import copy
import statistics
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
    NX,
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
import sweep_rf_basis_size as sweep  # noqa: E402 -- reused unchanged: build_complete_checkpoint
import test_pde_trained_gaussian_correction as ptgc  # noqa: E402 -- reused unchanged: build_search_baseline
import test_fixed_gaussian_correction as tfgc  # noqa: E402 -- reused unchanged: solution_ylim

# CANONICAL RF RANDOM-SEED SENSITIVITY (TODO.md Phase A item B, 2026-10-14
# advisor meeting / RESEARCH_LOG.md Section 35). The basis-size sweep
# (sweep_rf_basis_size.py) used seed=42 only; this script asks whether the
# canonical Nx=500,Nt=10 RF-only result is a property of the method or a
# particular random draw, by training 5 additional independent seeds under
# IDENTICAL canonical settings (Nx, Nt, sigma_x, sigma_t, M_TRAIN, M_BD,
# M_INT, epochs, LR, weight_decay, IC_BC_WEIGHT) -- the SEED is the only
# variable. BurgersRF.__init__ and physics.train()/loss_fn are reused
# completely unmodified; no multiscale/local-Gaussian features are
# introduced anywhere in this script.
#
# SEED 42 IS REUSED, NOT RETRAINED: outputs/pde_trained_rf_residual/
# checkpoint_complete.pt already contains a COMPLETE model_state_dict
# (Wx, Wt, bx, bt, model.0.weight all present -- it was produced
# specifically to fix the missing-Wx/Wt/bx/bt bug, RESEARCH_LOG.md
# Section 16), so the canonical seed-42 model can be reconstructed by
# building a throwaway BurgersRF and calling load_state_dict(...) directly
# -- the SAME established pattern test_pde_trained_gaussian_correction.py
# already uses (see its build_baseline). This requires NO RNG replay: the
# checkpoint's own saved tensors are the single source of truth. The
# matching x_train/t_train (needed only for the post-hoc PDE-loss
# diagnostic) are separately regenerated via set_seed(42) ->
# sample_training_points(M_TRAIN, "cpu") -- this reproduces the ORIGINAL
# training points bit-identically, since sample_training_points is the
# very first RNG-consuming call after set_seed and is independent of
# anything that happens during model construction/reconstruction.
#
# SEED 42 IS A STATISTICAL PEER, NOT A BASELINE: per explicit instruction,
# seed 42 is just another realization of the same canonical random
# experiment -- the fact that it was trained earlier does not exclude it
# from the sensitivity statistics. Six-seed statistics (20,21,22,23,24,42)
# are the primary report; five-new-seed-only statistics are reported
# alongside for reference.
#
# DEVICE SAFETY: build_search_baseline builds its internal evaluation
# tensors on CPU unconditionally (no .to(device) call), so it only works
# correctly when the model passed to it is fully on CPU. For the 5 newly
# trained seeds (trained via get_device(), CUDA if available), the t=1
# diagnostic uses an INDEPENDENT deep copy moved to CPU with Wx/Wt/bx/bt
# explicitly del+reassigned -- the same safe pattern established in
# test_rf_nx2000_long_training.py and test_rf_multiscale_two_population.py,
# required because nn.Parameter(...).to(device) inside BurgersRF.__init__
# can silently downgrade Wx/Wt/bx/bt to plain Tensors on a genuine device
# conversion (RESEARCH_LOG.md Section 16), which nn.Module.to() cannot see.
# The live trained model/optimizer are never touched by this. The
# reconstructed seed-42 model is already CPU-only throughout, so no such
# copy is needed for it, but the SAME evaluation function is used for both
# to avoid duplicating diagnostic logic.

NEW_SEEDS = [20, 21, 22, 23, 24]
HISTORICAL_SEED = 42
ALL_SEEDS_ORDERED = NEW_SEEDS + [HISTORICAL_SEED]

CANONICAL_GLOBAL = 0.27149430095802507
CANONICAL_SMOOTH = 0.2443827084159696
CANONICAL_SHOCK = 0.8390487998889169
VERIFY_RTOL = 1e-4

IC_GRID_N_X = 1000

OUTPUT_DIR = PROJECT_DIR / "outputs" / "rf_random_seed_sweep"

HISTORICAL_CHECKPOINT_PATH = (
    PROJECT_DIR / "outputs" / "pde_trained_rf_residual" / "checkpoint_complete.pt"
)
EXPECTED_STATE_DICT_KEYS = {"Wx", "Wt", "bx", "bt", "model.0.weight"}

CSV_HEADER = [
    "seed", "is_historical_reference",
    "global_error", "smooth_error", "shock_error", "ic_error",
    "pde_loss", "ic_loss", "bc_loss",
    "t1_global_error", "t1_smooth_error", "t1_shock_error",
]


def relative_l2(y_true, y_pred):
    return float(np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true))


def train_one_seed(seed, device):
    """Canonical construction/training trajectory, IDENTICAL to
    sweep_rf_basis_size.py's run_one_nx except that `seed` (not Nx) is the
    varied quantity. BurgersRF construction and physics.train()/loss_fn are
    reused completely unmodified."""
    set_seed(seed)
    x_train, t_train = sample_training_points(M_TRAIN, device)

    print(f"=== seed={seed} ===")
    model = BurgersRF(NX, NT, SIGMA_X, SIGMA_T, use_local=False, device=device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    train(model, optimizer, x_train, t_train, M_BD, M_INT, NV, g, device, epochs=EPOCHS)

    return model, x_train, t_train


def load_historical_seed42_model():
    """Reconstruct the canonical seed-42 model directly from the existing
    complete checkpoint's own state_dict -- NO RNG replay, NO retraining.
    Fails loudly (does not fall back to retraining) if the checkpoint is
    missing or incomplete."""
    if not HISTORICAL_CHECKPOINT_PATH.exists():
        raise RuntimeError(
            f"Expected canonical seed-42 checkpoint not found at "
            f"{HISTORICAL_CHECKPOINT_PATH}. This script does NOT fall back to "
            "retraining seed 42 -- run outputs/pde_trained_rf_residual/ must "
            "already exist (e.g. transfer it from wherever the canonical "
            "baseline was trained, or run reconstruct_pde_trained_rf_checkpoint.py "
            "there first)."
        )
    checkpoint = torch.load(HISTORICAL_CHECKPOINT_PATH, map_location="cpu", weights_only=False)

    state_dict_keys = set(checkpoint["model_state_dict"].keys())
    if state_dict_keys != EXPECTED_STATE_DICT_KEYS:
        raise RuntimeError(
            f"{HISTORICAL_CHECKPOINT_PATH} model_state_dict keys "
            f"{sorted(state_dict_keys)} do not match the expected complete set "
            f"{sorted(EXPECTED_STATE_DICT_KEYS)}. Refusing to reuse this "
            "checkpoint as the seed-42 reference."
        )

    if checkpoint["seed"] != HISTORICAL_SEED:
        raise RuntimeError(
            f"{HISTORICAL_CHECKPOINT_PATH} records seed={checkpoint['seed']}, "
            f"not the expected {HISTORICAL_SEED}. Refusing to use it as the "
            "seed-42 reference."
        )

    model = BurgersRF(
        checkpoint["NX"], checkpoint["NT"], checkpoint["SIGMA_X"], checkpoint["SIGMA_T"],
        use_local=checkpoint["use_local"], device="cpu",
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Matching training points, needed only for the post-hoc PDE-loss
    # diagnostic. sample_training_points is the very first RNG-consuming
    # call after set_seed(seed) and is independent of model construction,
    # so this reproduces the ORIGINAL seed-42 training points bit-
    # identically regardless of what happened above.
    set_seed(checkpoint["seed"])
    x_train, t_train = sample_training_points(checkpoint["M_TRAIN"], "cpu")

    return model, x_train, t_train


def verify_seed42_reconstruction_gate(canonical_metrics):
    def rel_diff(a, b):
        return abs(a - b) / abs(b)

    checks = {
        "global": rel_diff(canonical_metrics["global_relative_l2_error"], CANONICAL_GLOBAL),
        "smooth": rel_diff(canonical_metrics["smooth_relative_l2_error"], CANONICAL_SMOOTH),
        "shock": rel_diff(canonical_metrics["shock_relative_l2_error"], CANONICAL_SHOCK),
    }
    print("Seed-42 reconstruction gate (vs. canonical RF-only baseline):")
    for k, v in checks.items():
        print(f"  {k}: relative difference = {v:.3e} (tolerance {VERIFY_RTOL:.1e})")
    failed = {k: v for k, v in checks.items() if v > VERIFY_RTOL}
    if failed:
        raise RuntimeError(
            f"SEED-42 RECONSTRUCTION GATE FAILED: {failed}. The reconstructed "
            "historical model does not reproduce the canonical reference -- "
            "refusing to include it in the seed-sensitivity comparison."
        )
    print("Reproduction gate PASSED.")


def evaluate_model_diagnostics(model, x_train, t_train, device, seed, is_historical):
    """All metrics required by the experiment, computed via established,
    reused evaluation code (no new error-region definitions): canonical
    multi-time global/smooth/shock + IC relative L2 + final PDE/IC/BC loss
    (same post-hoc pattern as sweep_rf_basis_size.py's run_one_nx), plus
    t=1-only global/smooth/shock via build_search_baseline (same pattern as
    plot_rf_basis_size_solutions.py). Returns (row, baseline) -- baseline is
    kept for the t=1 solution-comparison plots."""
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
    canonical_metrics = evaluate_error_metrics(x_test, y_true, y_pred, y_true_shock, y_pred_shock)

    x_ic = torch.linspace(-1, 1, IC_GRID_N_X).reshape(-1, 1).to(device)
    t_ic = torch.zeros_like(x_ic).to(device)
    y_ic_true = g(x_ic).detach().cpu().numpy().reshape(-1)
    y_ic_pred = model(x_ic, t_ic).cpu().detach().numpy().reshape(-1)
    ic_relative_l2 = relative_l2(y_ic_true, y_ic_pred)

    # t=1-only metrics. Use an independent CPU copy with Wx/Wt/bx/bt
    # explicitly del+reassigned (established device-safe pattern) -- the
    # live `model` is never modified by this.
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

    row = {
        "seed": seed,
        "is_historical_reference": is_historical,
        "global_error": canonical_metrics["global_relative_l2_error"],
        "smooth_error": canonical_metrics["smooth_relative_l2_error"],
        "shock_error": canonical_metrics["shock_relative_l2_error"],
        "ic_error": ic_relative_l2,
        "pde_loss": final_pde_loss,
        "ic_loss": final_ic_loss,
        "bc_loss": final_bc_loss,
        "t1_global_error": t1_metrics["global_relative_l2_error"],
        "t1_smooth_error": t1_metrics["smooth_relative_l2_error"],
        "t1_shock_error": t1_metrics["shock_relative_l2_error"],
    }
    return row, baseline, canonical_metrics


def write_csv(rows):
    path = OUTPUT_DIR / "summary.csv"
    rows_by_seed = {row["seed"]: row for row in rows}
    with path.open("w") as f:
        f.write(",".join(CSV_HEADER) + "\n")
        for seed in ALL_SEEDS_ORDERED:
            row = rows_by_seed[seed]
            f.write(",".join(str(row[k]) for k in CSV_HEADER) + "\n")


def compute_statistics(rows, seeds_subset):
    rows_by_seed = {row["seed"]: row for row in rows}
    selected = [rows_by_seed[s] for s in seeds_subset]
    stats = {}
    for metric in ("global_error", "smooth_error", "shock_error"):
        values = [row[metric] for row in selected]
        stats[metric] = {
            "mean": statistics.mean(values),
            "sample_stdev": statistics.stdev(values) if len(values) > 1 else float("nan"),
            "min": min(values),
            "max": max(values),
        }
    return stats


def identify_best_seeds(rows, seeds_subset):
    rows_by_seed = {row["seed"]: row for row in rows}
    selected = [rows_by_seed[s] for s in seeds_subset]
    best = {}
    for metric in ("global_error", "smooth_error", "shock_error"):
        best_row = min(selected, key=lambda row: row[metric])
        best[metric] = best_row["seed"]
    return best


def write_summary(rows, six_seed_stats, five_seed_stats, best_seeds):
    lines = []
    lines.append("Canonical RF random-seed sensitivity -- Nx=500, Nt=10")
    lines.append(
        f"New seeds trained independently: {NEW_SEEDS}. Historical reference "
        f"(reused, NOT retrained): seed={HISTORICAL_SEED}."
    )
    lines.append(
        f"Fixed: Nt={NT}, SIGMA_X={SIGMA_X}, SIGMA_T={SIGMA_T}, M_TRAIN={M_TRAIN}, "
        f"M_BD={M_BD}, M_INT={M_INT}, EPOCHS={EPOCHS}, LR={LR}, WEIGHT_DECAY={WEIGHT_DECAY}. "
        "Only the seed varies -- BurgersRF construction and physics.train()/loss_fn "
        "are reused completely unmodified; no multiscale or local-Gaussian features."
    )
    lines.append("")
    lines.append(
        "Seed-42 reconstruction: loaded directly from the existing complete "
        "model_state_dict (outputs/pde_trained_rf_residual/checkpoint_complete.pt) -- "
        "NO RNG replay, NO retraining. Matching training points regenerated via "
        "set_seed(42) -> sample_training_points(...) for the post-hoc PDE-loss "
        "diagnostic only. Gated against the canonical reference at rtol=1e-4 "
        "before being included below."
    )
    lines.append("")
    lines.append(
        "Per-seed results (seed=42 is a statistical peer of the canonical random "
        "experiment, not a baseline -- it is included in the six-seed statistics):"
    )
    rows_by_seed = {row["seed"]: row for row in rows}
    for seed in ALL_SEEDS_ORDERED:
        row = rows_by_seed[seed]
        tag = " (historical reference)" if row["is_historical_reference"] else ""
        lines.append(f"--- seed = {seed}{tag} ---")
        for k in CSV_HEADER[2:]:
            lines.append(f"  {k} = {row[k]}")
    lines.append("")
    lines.append(f"Six-seed statistics ({ALL_SEEDS_ORDERED}):")
    for metric, s in six_seed_stats.items():
        lines.append(
            f"  {metric}: mean={s['mean']:.6f}, sample_stdev={s['sample_stdev']:.6f}, "
            f"min={s['min']:.6f}, max={s['max']:.6f}"
        )
    lines.append("")
    lines.append(f"Five-new-seed-only statistics ({NEW_SEEDS}, reference only):")
    for metric, s in five_seed_stats.items():
        lines.append(
            f"  {metric}: mean={s['mean']:.6f}, sample_stdev={s['sample_stdev']:.6f}, "
            f"min={s['min']:.6f}, max={s['max']:.6f}"
        )
    lines.append("")
    lines.append("Best seed by metric (six-seed set) -- NOT a recommended fixed seed for")
    lines.append("future science; this experiment measures sensitivity to random realization:")
    for metric, seed in best_seeds.items():
        lines.append(f"  best by {metric}: seed={seed}")

    (OUTPUT_DIR / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def plot_t1_solution_comparison(baselines_by_seed):
    truth_x = baselines_by_seed[ALL_SEEDS_ORDERED[0]]["x_eval_full"]
    truth_y = baselines_by_seed[ALL_SEEDS_ORDERED[0]]["y_eval_full"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(truth_x, truth_y, "k-", linewidth=2, label="u_true")
    for seed in NEW_SEEDS:
        baseline = baselines_by_seed[seed]
        ax.plot(baseline["x_eval_full"], baseline["u_RF_full"], "--", label=f"seed={seed}")
    historical = baselines_by_seed[HISTORICAL_SEED]
    ax.plot(
        historical["x_eval_full"], historical["u_RF_full"], "-", color="black",
        linewidth=1.2, alpha=0.6, label=f"seed={HISTORICAL_SEED} (historical)",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t=1)")
    ax.set_title("RF-only solutions across seeds, t=1 (Nx=500, Nt=10)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "t1_solution_comparison.png")
    plt.close(fig)


def plot_t1_shock_comparison(baselines_by_seed):
    truth_x = baselines_by_seed[ALL_SEEDS_ORDERED[0]]["x_eval_shock"]
    truth_y = baselines_by_seed[ALL_SEEDS_ORDERED[0]]["y_eval_shock"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(truth_x, truth_y, "k-", linewidth=2, label="u_true")
    for seed in NEW_SEEDS:
        baseline = baselines_by_seed[seed]
        ax.plot(baseline["x_eval_shock"], baseline["u_RF_shock"], "--", label=f"seed={seed}")
    historical = baselines_by_seed[HISTORICAL_SEED]
    ax.plot(
        historical["x_eval_shock"], historical["u_RF_shock"], "-", color="black",
        linewidth=1.2, alpha=0.6, label=f"seed={HISTORICAL_SEED} (historical)",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t=1)")
    ax.set_title("RF-only solutions across seeds, shock region, t=1")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "t1_shock_comparison.png")
    plt.close(fig)


def plot_error_vs_seed(rows):
    """Seed is a categorical experimental identifier, not a continuous
    physical variable -- markers only, no connecting/fitted line, so no
    continuous trend is implied."""
    rows_by_seed = {row["seed"]: row for row in rows}
    ordered_rows = [rows_by_seed[s] for s in ALL_SEEDS_ORDERED]
    x_positions = list(range(len(ALL_SEEDS_ORDERED)))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(x_positions, [r["global_error"] for r in ordered_rows], marker="o", label="global relative L2")
    ax.scatter(x_positions, [r["smooth_error"] for r in ordered_rows], marker="s", label="smooth relative L2")
    ax.scatter(x_positions, [r["shock_error"] for r in ordered_rows], marker="^", label="shock relative L2")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{s}\n(historical)" if s == HISTORICAL_SEED else str(s) for s in ALL_SEEDS_ORDERED])
    ax.set_xlabel("seed (categorical)")
    ax.set_ylabel("canonical multi-time relative L2 error")
    ax.set_title("RF-only error vs. random seed (Nx=500, Nt=10, all else fixed)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "error_vs_seed.png")
    plt.close(fig)


def print_table(rows):
    header = ["seed", "hist", "global", "smooth", "shock", "IC", "pde_loss", "ic_loss", "bc_loss", "t1_global", "t1_smooth", "t1_shock"]
    print(" | ".join(f"{h:>11}" for h in header))
    rows_by_seed = {row["seed"]: row for row in rows}
    for seed in ALL_SEEDS_ORDERED:
        row = rows_by_seed[seed]
        vals = [
            str(row["seed"]), ("Y" if row["is_historical_reference"] else ""),
            f"{row['global_error']:.6f}", f"{row['smooth_error']:.6f}", f"{row['shock_error']:.6f}",
            f"{row['ic_error']:.6f}", f"{row['pde_loss']:.6e}", f"{row['ic_loss']:.6e}", f"{row['bc_loss']:.6e}",
            f"{row['t1_global_error']:.6f}", f"{row['t1_smooth_error']:.6f}", f"{row['t1_shock_error']:.6f}",
        ]
        print(" | ".join(f"{v:>11}" for v in vals))


def main():
    torch.set_default_dtype(DTYPE)
    device = get_device()
    print(device)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    baselines_by_seed = {}

    # Seed 42 first: fail fast if the historical checkpoint is missing or
    # does not reproduce the canonical reference, before spending time
    # training the 5 new seeds.
    model42, x_train_42, t_train_42 = load_historical_seed42_model()
    row42, baseline42, canonical_metrics_42 = evaluate_model_diagnostics(
        model42, x_train_42, t_train_42, "cpu", HISTORICAL_SEED, is_historical=True
    )
    verify_seed42_reconstruction_gate(canonical_metrics_42)
    rows.append(row42)
    baselines_by_seed[HISTORICAL_SEED] = baseline42

    for seed in NEW_SEEDS:
        model, x_train, t_train = train_one_seed(seed, device)
        row, baseline, canonical_metrics = evaluate_model_diagnostics(
            model, x_train, t_train, device, seed, is_historical=False
        )

        metadata = {
            "Nt": NT, "SIGMA_X": SIGMA_X, "SIGMA_T": SIGMA_T, "NV": NV, "DTYPE": str(DTYPE),
            "seed": seed, "M_TRAIN": M_TRAIN, "M_BD": M_BD, "M_INT": M_INT,
            "EPOCHS": EPOCHS, "LR": LR, "WEIGHT_DECAY": WEIGHT_DECAY,
            "global_relative_l2_error": canonical_metrics["global_relative_l2_error"],
            "smooth_relative_l2_error": canonical_metrics["smooth_relative_l2_error"],
            "shock_relative_l2_error": canonical_metrics["shock_relative_l2_error"],
            "ic_relative_l2_error": row["ic_error"],
            "final_pde_loss": row["pde_loss"], "final_ic_loss": row["ic_loss"], "final_bc_loss": row["bc_loss"],
        }
        checkpoint = sweep.build_complete_checkpoint(model, NX, metadata)
        torch.save(checkpoint, OUTPUT_DIR / f"checkpoint_seed{seed}.pt")

        rows.append(row)
        baselines_by_seed[seed] = baseline

    write_csv(rows)

    six_seed_stats = compute_statistics(rows, ALL_SEEDS_ORDERED)
    five_seed_stats = compute_statistics(rows, NEW_SEEDS)
    best_seeds = identify_best_seeds(rows, ALL_SEEDS_ORDERED)

    write_summary(rows, six_seed_stats, five_seed_stats, best_seeds)
    plot_t1_solution_comparison(baselines_by_seed)
    plot_t1_shock_comparison(baselines_by_seed)
    plot_error_vs_seed(rows)
    print_table(rows)

    print("[sweep_rf_random_seeds] done.")


if __name__ == "__main__":
    main()
