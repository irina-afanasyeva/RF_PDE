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
import test_pde_trained_gaussian_correction as ptgc  # noqa: E402 -- reused unchanged: build_search_baseline
import test_fixed_gaussian_correction as tfgc  # noqa: E402 -- reused unchanged: solution_ylim
import sweep_rf_basis_size as sweep  # noqa: E402 -- reused unchanged: build_complete_checkpoint

# MULTISCALE / TWO-POPULATION RF experiment (TODO.md item E, RESEARCH_LOG.md
# Section 30). This is a first hypothesis-driven diagnostic, NOT a tuned
# optimum. Nx_total=500, Nt=10 unchanged from the canonical baseline; ONLY
# the frequency CONTENT of the last 250 Wx slots is changed -- everything
# else (Wt, bx, bt, model.0.weight initialization, training points,
# optimizer, LR, epochs, weight decay, IC_BC_WEIGHT, evaluation grids) is
# taken from the EXACT canonical seed-42 construction trajectory, untouched.
# model.py/physics.py/config.py are NOT modified.
#
# CRITICAL RNG DESIGN (do not deviate from this -- see the user's explicit
# correction to the original proposal): naively drawing a second
# torch.randn(1,250) for a "high" population would consume ADDITIONAL
# random numbers and shift the RNG trajectory for Wt/bx/bt and all
# subsequent per-epoch BC/IC sampling relative to the canonical baseline --
# this was verified empirically to break bit-identical reproduction in an
# earlier step. Instead:
#   1. Construct the model via the EXACT canonical trajectory:
#          set_seed(SEED) -> sample_training_points(M_TRAIN,device) ->
#          BurgersRF(NX,NT,SIGMA_X,SIGMA_T,use_local=False,device).to(device)
#      This draws Wx_canonical, Wt, bx, bt, and initializes model.0.weight
#      via the SAME RNG consumption as every other canonical experiment in
#      this project -- NO extra randn/rand calls happen anywhere.
#   2. Wx_canonical = Z / SIGMA_X, i.e. Z = Wx_canonical * SIGMA_X (the
#      underlying standard-normal samples).
#   3. Keep Wx[:, :N_CANONICAL] exactly as drawn (population 1, canonical
#      scale).
#   4. For slots N_CANONICAL:NX, RESCALE the EXISTING canonical values
#      (not a new draw): Wx[:, N_CANONICAL:] = Wx_canonical[:, N_CANONICAL:]
#      * (SIGMA_X / SIGMA_HIGH) = Z[:, N_CANONICAL:] / SIGMA_HIGH -- the
#      SAME underlying Z samples, now at the high-frequency scale.
#   5. Replace ONLY model.Wx with this multiscale tensor (del+reassign,
#      re-wrapped as a genuine nn.Parameter -- the same safe pattern
#      already established in test_rf_nx2000_long_training.py's device fix,
#      here used to ensure correct registration regardless of device).
# Wt, bx, bt, and model.0.weight are NEVER touched -- they remain whatever
# the canonical construction produced, by construction (not merely by
# intention). All validation/diagnostic code below operates via pure tensor
# arithmetic/comparison -- ZERO additional random draws -- and this is
# verified explicitly by snapshotting torch.get_rng_state() immediately
# after canonical construction and confirming it is UNCHANGED immediately
# before training begins.

N_CANONICAL = 250
N_HIGH = 250
SIGMA_CANONICAL = SIGMA_X  # population 1: unchanged canonical scale (0.04)
SIGMA_HIGH = 0.0032        # population 2: principled first diagnostic choice (see design report)

CANONICAL_GLOBAL = 0.27149430095802507
CANONICAL_SMOOTH = 0.2443827084159696
CANONICAL_SHOCK = 0.8390487998889169

IC_GRID_N_X = 1000
UNDERLYING_SAMPLE_RTOL = 1e-10
UNDERLYING_SAMPLE_ATOL = 1e-12

OUTPUT_DIR = PROJECT_DIR / "outputs" / "rf_multiscale_two_population"

CSV_HEADER = [
    "population", "count", "mean", "std", "min_abs_W", "max_abs_W",
]


def relative_l2(y_true, y_pred):
    return float(np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true))


def build_multiscale_model(device):
    """Construct the EXACT canonical seed-42 trajectory, snapshot every
    canonical tensor BEFORE any modification, build the multiscale Wx by
    RESCALING (not redrawing) the existing canonical upper half, replace
    ONLY model.Wx, and run every validation assertion the task requires.
    Returns (model, x_train, t_train, canonical_snapshot, population_stats).
    Raises immediately if any control assertion fails."""
    rng_state_before = torch.get_rng_state()

    set_seed(SEED)
    x_train, t_train = sample_training_points(M_TRAIN, device)
    model = BurgersRF(NX, NT, SIGMA_X, SIGMA_T, use_local=False, device=device).to(device)

    # Snapshot EVERY canonical tensor before any modification.
    Wx_canonical = model.Wx.detach().clone()
    Wt_canonical = model.Wt.detach().clone()
    bx_canonical = model.bx.detach().clone()
    bt_canonical = model.bt.detach().clone()
    weight_canonical = model.model[0].weight.detach().clone()

    rng_state_after_canonical_construction = torch.get_rng_state()

    # Build the multiscale Wx: population 1 unchanged, population 2 =
    # RESCALED existing canonical values (same underlying Z, new sigma).
    Wx_multiscale = Wx_canonical.clone()
    rescale_factor = SIGMA_CANONICAL / SIGMA_HIGH
    Wx_multiscale[:, N_CANONICAL:] = Wx_canonical[:, N_CANONICAL:] * rescale_factor

    # Replace ONLY model.Wx. del+reassign (not a bare attribute set) is
    # required because Wx may already be a registered nn.Parameter (the
    # CPU-only/no-genuine-conversion case) -- nn.Module.__setattr__ refuses
    # to overwrite an already-registered parameter name with anything that
    # isn't itself an nn.Parameter/None. Re-wrapping explicitly as
    # nn.Parameter here (rather than leaving it a plain Tensor) ensures
    # correct registration regardless of whether the known
    # nn.Parameter(...).to(device) registration bug (RESEARCH_LOG.md
    # Section 16) was triggered during the canonical construction above.
    del model.Wx
    model.Wx = torch.nn.Parameter(Wx_multiscale, requires_grad=False)

    # --- Validation assertions (ALL pure tensor comparisons -- ZERO RNG) ---
    checks = {}
    checks["first_250_Wx_unchanged"] = torch.equal(model.Wx[:, :N_CANONICAL], Wx_canonical[:, :N_CANONICAL])
    checks["underlying_samples_match"] = torch.allclose(
        model.Wx[:, N_CANONICAL:] * SIGMA_HIGH,
        Wx_canonical[:, N_CANONICAL:] * SIGMA_CANONICAL,
        rtol=UNDERLYING_SAMPLE_RTOL, atol=UNDERLYING_SAMPLE_ATOL,
    )
    checks["last_250_Wx_actually_changed"] = not torch.equal(model.Wx[:, N_CANONICAL:], Wx_canonical[:, N_CANONICAL:])
    checks["Wt_unchanged"] = torch.equal(model.Wt, Wt_canonical)
    checks["bx_unchanged"] = torch.equal(model.bx, bx_canonical)
    checks["bt_unchanged"] = torch.equal(model.bt, bt_canonical)
    checks["model_weight_init_unchanged"] = torch.equal(model.model[0].weight, weight_canonical)

    print("Control assertions (all must PASS before proceeding):")
    all_passed = True
    for name, passed in checks.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
        all_passed = all_passed and passed
    if not all_passed:
        raise RuntimeError(f"Control assertion(s) FAILED: {checks}. Refusing to proceed.")
    print("All control assertions PASSED.")

    # Strongest possible RNG-safety check: confirm the global RNG state is
    # UNCHANGED between immediately after canonical construction and
    # immediately before training -- i.e. none of the Wx-replacement or
    # assertion code above consumed any random numbers.
    rng_state_now = torch.get_rng_state()
    rng_preserved = torch.equal(rng_state_after_canonical_construction, rng_state_now)
    print(f"RNG state unchanged since canonical construction (zero RNG consumed by Wx replacement/assertions): {rng_preserved}")
    if not rng_preserved:
        raise RuntimeError(
            "RNG state changed between canonical construction and the start of training -- "
            "the Wx-replacement/validation code consumed random numbers, which would shift "
            "the per-epoch BC/IC sampling trajectory relative to the canonical baseline. "
            "Refusing to proceed."
        )

    canonical_snapshot = {
        "Wx_canonical": Wx_canonical, "Wt_canonical": Wt_canonical,
        "bx_canonical": bx_canonical, "bt_canonical": bt_canonical,
        "weight_canonical": weight_canonical,
        "rng_state_before": rng_state_before,
    }

    population_stats = compute_population_stats(model.Wx)

    return model, x_train, t_train, canonical_snapshot, population_stats


def compute_population_stats(Wx):
    """Empirical statistics per population. Pure tensor reductions -- no RNG."""
    pop1 = Wx[:, :N_CANONICAL]
    pop2 = Wx[:, N_CANONICAL:]
    stats = []
    for label, pop in [("population_1_canonical", pop1), ("population_2_high", pop2)]:
        stats.append({
            "population": label,
            "count": pop.numel(),
            "mean": float(pop.mean().item()),
            "std": float(pop.std().item()),
            "min_abs_W": float(pop.abs().min().item()),
            "max_abs_W": float(pop.abs().max().item()),
        })
    return stats


def write_population_stats_csv(stats):
    path = OUTPUT_DIR / "population_stats.csv"
    with path.open("w") as f:
        f.write(",".join(CSV_HEADER) + "\n")
        for row in stats:
            f.write(",".join(str(row[k]) for k in CSV_HEADER) + "\n")


def write_summary(
    population_stats, final_metrics_canonical, ic_relative_l2,
    final_pde_loss, final_ic_loss, final_bc_loss, metrics_t1,
):
    lines = []
    lines.append("RF multiscale two-population experiment (first hypothesis-driven diagnostic)")
    lines.append(f"Nx_total={NX}, Nt={NT}, N_canonical={N_CANONICAL}, N_high={N_HIGH}")
    lines.append(f"sigma_canonical={SIGMA_CANONICAL} (std={1/SIGMA_CANONICAL}), sigma_high={SIGMA_HIGH} (std={1/SIGMA_HIGH})")
    lines.append(
        "Population 2 uses the SAME underlying standard-normal samples as the canonical "
        "model's own Wx[:, 250:500], rescaled to the new sigma -- NOT a fresh draw. Wt, bx, "
        "bt, and the initial model.0.weight are bit-identical to the canonical seed-42 "
        "construction."
    )
    lines.append("")
    lines.append("Empirical per-population Wx statistics:")
    for row in population_stats:
        lines.append(f"  {row}")
    lines.append("")
    lines.append("Canonical Nx=500 single-population baseline reference metrics:")
    lines.append(f"  global={CANONICAL_GLOBAL}, smooth={CANONICAL_SMOOTH}, shock={CANONICAL_SHOCK}")
    lines.append("")
    lines.append("Multiscale model -- canonical multi-time metrics:")
    lines.append(f"  {final_metrics_canonical}")
    lines.append(f"  ic_relative_l2_error = {ic_relative_l2}")
    lines.append("")
    lines.append(
        "Final unweighted losses (PDE residual is reported as mean-squared residual, "
        "matching this project's established 'final_pde_loss' convention -- not a separately "
        "normalized 'relative' quantity, since no such normalization is defined elsewhere in "
        "this codebase):"
    )
    lines.append(f"  final_pde_loss = {final_pde_loss}")
    lines.append(f"  final_ic_loss = {final_ic_loss}")
    lines.append(f"  final_bc_loss = {final_bc_loss}")
    lines.append("")
    lines.append("Multiscale model -- t=1-only (fine grid) metrics:")
    lines.append(f"  {metrics_t1}")

    (OUTPUT_DIR / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def write_csv(final_metrics_canonical, ic_relative_l2, final_pde_loss, final_ic_loss, final_bc_loss, metrics_t1):
    header = [
        "global_relative_l2_error", "smooth_relative_l2_error", "shock_relative_l2_error",
        "ic_relative_l2_error", "final_pde_loss", "final_ic_loss", "final_bc_loss",
        "t1_global_relative_l2_error", "t1_smooth_relative_l2_error", "t1_shock_relative_l2_error",
    ]
    row = [
        final_metrics_canonical["global_relative_l2_error"], final_metrics_canonical["smooth_relative_l2_error"],
        final_metrics_canonical["shock_relative_l2_error"], ic_relative_l2, final_pde_loss, final_ic_loss, final_bc_loss,
        metrics_t1["global_relative_l2_error"], metrics_t1["smooth_relative_l2_error"], metrics_t1["shock_relative_l2_error"],
    ]
    path = OUTPUT_DIR / "summary.csv"
    with path.open("w") as f:
        f.write(",".join(header) + "\n")
        f.write(",".join(str(v) for v in row) + "\n")


def plot_t1_solution(suffix, x, y_true, u_rf, title_prefix):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(x, y_true, "k-", linewidth=2, label="u_true")
    ax.plot(x, u_rf, "--", label="u_RF (multiscale)")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t=1)")
    ax.set_ylim(tfgc.solution_ylim(y_true, u_rf))
    ax.set_title(f"{title_prefix}, t=1")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"solution_{suffix}.png")
    plt.close(fig)


def main():
    torch.set_default_dtype(DTYPE)
    device = get_device()
    print(device)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model, x_train, t_train, canonical_snapshot, population_stats = build_multiscale_model(device)
    print("Per-population Wx statistics:")
    for row in population_stats:
        print(f"  {row}")
    write_population_stats_csv(population_stats)

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # physics.train/loss_fn reused completely unmodified.
    train(model, optimizer, x_train, t_train, M_BD, M_INT, NV, g, device, epochs=EPOCHS)

    # Post-hoc unweighted component losses, same established pattern as
    # sweep_rf_basis_size.py / test_rf_nx2000_long_training.py.
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
    final_metrics_canonical = evaluate_error_metrics(x_test, y_true, y_pred, y_true_shock, y_pred_shock)

    x_ic = torch.linspace(-1, 1, IC_GRID_N_X).reshape(-1, 1).to(device)
    t_ic = torch.zeros_like(x_ic).to(device)
    y_ic_true = g(x_ic).detach().cpu().numpy().reshape(-1)
    y_ic_pred = model(x_ic, t_ic).cpu().detach().numpy().reshape(-1)
    ic_relative_l2 = relative_l2(y_ic_true, y_ic_pred)

    metadata = {
        "Nt": NT, "SIGMA_X": SIGMA_X, "SIGMA_T": SIGMA_T, "NV": NV, "DTYPE": str(DTYPE),
        "seed": SEED, "M_TRAIN": M_TRAIN, "M_BD": M_BD, "M_INT": M_INT,
        "EPOCHS": EPOCHS, "LR": LR, "WEIGHT_DECAY": WEIGHT_DECAY,
        "global_relative_l2_error": final_metrics_canonical["global_relative_l2_error"],
        "smooth_relative_l2_error": final_metrics_canonical["smooth_relative_l2_error"],
        "shock_relative_l2_error": final_metrics_canonical["shock_relative_l2_error"],
        "ic_relative_l2_error": ic_relative_l2,
        "final_pde_loss": final_pde_loss, "final_ic_loss": final_ic_loss, "final_bc_loss": final_bc_loss,
    }
    checkpoint = sweep.build_complete_checkpoint(model, NX, metadata)
    checkpoint["N_CANONICAL"] = N_CANONICAL
    checkpoint["N_HIGH"] = N_HIGH
    checkpoint["SIGMA_HIGH"] = SIGMA_HIGH
    torch.save(checkpoint, OUTPUT_DIR / "checkpoint.pt")

    # build_search_baseline builds its internal x/t evaluation tensors on
    # CPU unconditionally (no .to(device) call -- see RESEARCH_LOG.md /
    # test_rf_nx2000_long_training.py's device-handling fix), so it only
    # works correctly when `model` itself is fully on CPU. Training is
    # already complete at this point, but `model.Wt/bx/bt` (never touched
    # by this script -- they are the canonical construction's own tensors)
    # may still be plain Tensors rather than registered Parameters if the
    # canonical construction ran on a genuine device conversion (the known
    # nn.Parameter(...).to(device) registration bug, RESEARCH_LOG.md
    # Section 16) -- in that case a bare `.to("cpu")` on the whole module
    # would leave them stranded on the original device. Use an independent
    # deep copy with each RF tensor explicitly moved (del+reassign, same
    # safe pattern as test_rf_nx2000_long_training.py's fix) rather than
    # touching the live model.
    cpu_model_for_eval = copy.deepcopy(model).to("cpu")
    for attr_name in ("Wx", "Wt", "bx", "bt"):
        cpu_value = getattr(cpu_model_for_eval, attr_name).detach().cpu().clone()
        delattr(cpu_model_for_eval, attr_name)
        setattr(cpu_model_for_eval, attr_name, cpu_value)

    baseline = ptgc.build_search_baseline(cpu_model_for_eval, "cpu")
    metrics_t1 = evaluate_error_metrics(
        baseline["x_eval_full"], baseline["y_eval_full"], baseline["u_RF_full"],
        baseline["y_eval_shock"], baseline["u_RF_shock"],
    )
    plot_t1_solution("full", baseline["x_eval_full"], baseline["y_eval_full"], baseline["u_RF_full"], "Multiscale RF full-domain solution")
    plot_t1_solution("shock", baseline["x_eval_shock"], baseline["y_eval_shock"], baseline["u_RF_shock"], "Multiscale RF shock-region solution")

    write_csv(final_metrics_canonical, ic_relative_l2, final_pde_loss, final_ic_loss, final_bc_loss, metrics_t1)
    write_summary(population_stats, final_metrics_canonical, ic_relative_l2, final_pde_loss, final_ic_loss, final_bc_loss, metrics_t1)

    print("[test_rf_multiscale_two_population] done.")


if __name__ == "__main__":
    main()
