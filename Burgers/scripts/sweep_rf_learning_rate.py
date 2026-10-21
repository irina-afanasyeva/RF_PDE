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
    DTYPE, EPOCHS, LR as CANONICAL_LR, M_BD, M_INT, M_TEST, M_TRAIN, N_TEST_TIMES,
    NT, NV, NX, SEED, SIGMA_T, SIGMA_X, WEIGHT_DECAY, get_device, set_seed,
)
from burgers_rf.data import g, make_reference_values, make_test_grid, sample_training_points  # noqa: E402
from burgers_rf.model import BurgersRF  # noqa: E402
from burgers_rf.physics import burgers_residual, train  # noqa: E402
from burgers_rf.evaluation import evaluate_error_metrics  # noqa: E402
from sweep_local_width import build_shock_grid  # noqa: E402
import sweep_rf_basis_size as sweep  # noqa: E402 -- reused unchanged: build_complete_checkpoint
import test_pde_trained_gaussian_correction as ptgc  # noqa: E402 -- reused unchanged: build_search_baseline
from test_rf_nx2000_long_training import boundary_residual_fixed, initial_residual_fixed  # noqa: E402
import test_rf_nx500_long_training as nx500long  # noqa: E402 -- reused unchanged: load_source_checkpoint,
# build_model_from_checkpoint (established CPU-first + load_state_dict + defensive device-safe
# re-pin pattern -- avoids the known nn.Parameter(...).to(device) registration bug, Section 16)

# EXPERIMENT 3 -- learning-rate sensitivity (2026-10-21 advisor meeting
# follow-up). Question: is the canonical Adam LR=5e-4 too aggressive and a
# contributor to the RF-only solution error? Compares LR in
# {2e-4, 1e-4, 5e-5} (freshly trained) against the EXISTING canonical
# LR=5e-4 result (reused, NOT retrained). BurgersRF construction and
# physics.train()/loss_fn are reused completely unmodified; Nx, Nt, sigma_x,
# sigma_t, epochs, weight_decay, IC_BC_WEIGHT, collocation counts,
# architecture, and precision are all held at their canonical config.py
# values -- LR is the only intended experimental variable.
#
# RNG / FAIR-COMPARISON NOTE: for EACH of the three new LR runs,
# set_seed(SEED) -> sample_training_points(M_TRAIN, device) ->
# BurgersRF(NX, NT, SIGMA_X, SIGMA_T, ...) is called fresh, in that exact
# order, before the optimizer (with that run's own LR) is constructed.
# None of these RNG-consuming calls depend on the Adam learning-rate value
# in any way -- LR only affects the OPTIMIZER's parameter-update step
# (optimizer.step()), never a sampling call. Consequently: (1) the interior
# collocation points x_train,t_train are BIT-IDENTICAL across all three new
# LR runs (and would also match the original LR=5e-4 run's own points, had
# it used this identical procedure); (2) Wx,Wt,bx,bt and the initial
# trainable coefficients are BIT-IDENTICAL across all three new LR runs,
# since BurgersRF.__init__'s random draws depend only on (Nx,Nt,sigma_x,
# sigma_t), never on LR; (3) the per-epoch boundary_residual/
# initial_residual draws inside physics.train()'s loss_fn are therefore
# ALSO bit-identical across all three new LR runs epoch-for-epoch, since
# they depend only on the RNG stream's position, which follows an identical
# sequence of consuming operations (same model construction, same
# epochs-per-segment) regardless of LR. No separate RNG stream or new
# sampling methodology is introduced.
#
# SEGMENTED TRAINING: each new LR trains continuously from epoch 0 to 1000
# via three consecutive train() calls (0->250, 250->500, 500->1000) on the
# SAME model/optimizer objects, with intermediate diagnostics computed via
# the RNG-FREE fixed-grid helpers (boundary_residual_fixed/
# initial_residual_fixed, imported unchanged from
# test_rf_nx2000_long_training.py) -- this is the already-proven-safe
# pattern: segmenting train() this way is mathematically identical to one
# continuous 1000-epoch call, since nothing besides train() consumes RNG,
# and the fixed-grid diagnostics consume none. Evaluating at epoch 250/500
# therefore does NOT alter the RNG/training trajectory for the remaining
# segments in any way.
#
# CANONICAL LR=5e-4 REFERENCE: reconstructed directly from the existing
# outputs/pde_trained_rf_residual/checkpoint_complete.pt via
# test_rf_nx500_long_training.load_source_checkpoint/
# build_model_from_checkpoint (unchanged, zero retraining), then gated
# against the known canonical global/smooth/shock values at rtol=1e-4
# before being trusted for comparison. ONLY its real epoch=1000 result
# exists -- no 250/500-epoch values are fabricated for it.

LR_VALUES_NEW = [2e-4, 1e-4, 5e-5]
CHECKPOINT_EPOCHS = [250, 500, 1000]
SEGMENT_LENGTHS = [250, 250, 500]
assert sum(SEGMENT_LENGTHS) == EPOCHS == CHECKPOINT_EPOCHS[-1]

CANONICAL_GLOBAL = 0.27149430095802507
CANONICAL_SMOOTH = 0.2443827084159696
CANONICAL_SHOCK = 0.8390487998889169
VERIFY_RTOL = 1e-4

OUTPUT_DIR = PROJECT_DIR / "outputs" / "rf_learning_rate_sweep"
IC_GRID_N_X = 1000

CSV_HEADER = [
    "learning_rate", "epoch",
    "global_relative_l2_error", "smooth_relative_l2_error", "shock_relative_l2_error",
    "ic_relative_l2_error", "final_pde_loss", "final_ic_loss", "final_bc_loss",
    "t1_global_relative_l2_error", "t1_smooth_relative_l2_error", "t1_shock_relative_l2_error",
]


def lr_tag(learning_rate):
    """Safe, clear, sortable filename fragment, e.g. 5e-4 -> '5e-04'."""
    return f"{learning_rate:.0e}"


def relative_l2(y_true, y_pred):
    return float(np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true))


def evaluate_at_epoch(
    model, learning_rate, epoch, x_train, t_train, t_bd_fixed, x_int_fixed,
    x_test, t_test, y_true, x_shock, t_shock, y_true_shock, x_ic, t_ic, y_ic_true, device,
):
    """Same RNG-free diagnostic pattern as test_rf_nx500_long_training.py's
    evaluate_at_epoch, parameterized by `learning_rate` (that module hardcodes
    the canonical config.LR in its checkpoint metadata, which would mislabel
    a non-canonical-LR run -- this is the one deliberate, minimal deviation)."""
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
        "EPOCHS": epoch, "LR": learning_rate, "WEIGHT_DECAY": WEIGHT_DECAY,
        "global_relative_l2_error": canonical_metrics["global_relative_l2_error"],
        "smooth_relative_l2_error": canonical_metrics["smooth_relative_l2_error"],
        "shock_relative_l2_error": canonical_metrics["shock_relative_l2_error"],
        "ic_relative_l2_error": ic_relative_l2,
        "final_pde_loss": final_pde_loss, "final_ic_loss": final_ic_loss, "final_bc_loss": final_bc_loss,
    }
    checkpoint = sweep.build_complete_checkpoint(model, NX, metadata)
    checkpoint["epoch"] = epoch
    checkpoint["learning_rate"] = learning_rate

    row = {
        "learning_rate": learning_rate, "epoch": epoch,
        "global_relative_l2_error": canonical_metrics["global_relative_l2_error"],
        "smooth_relative_l2_error": canonical_metrics["smooth_relative_l2_error"],
        "shock_relative_l2_error": canonical_metrics["shock_relative_l2_error"],
        "ic_relative_l2_error": ic_relative_l2,
        "final_pde_loss": final_pde_loss, "final_ic_loss": final_ic_loss, "final_bc_loss": final_bc_loss,
        "t1_global_relative_l2_error": t1_metrics["global_relative_l2_error"],
        "t1_smooth_relative_l2_error": t1_metrics["smooth_relative_l2_error"],
        "t1_shock_relative_l2_error": t1_metrics["shock_relative_l2_error"],
    }
    return row, checkpoint


def verify_gate(row, label):
    def rel_diff(a, b):
        return abs(a - b) / abs(b)

    checks = {
        "global": rel_diff(row["global_relative_l2_error"], CANONICAL_GLOBAL),
        "smooth": rel_diff(row["smooth_relative_l2_error"], CANONICAL_SMOOTH),
        "shock": rel_diff(row["shock_relative_l2_error"], CANONICAL_SHOCK),
    }
    print(f"{label} reproduction gate:")
    for k, v in checks.items():
        print(f"  {k}: relative difference = {v:.3e} (tolerance {VERIFY_RTOL:.1e})")
    failed = {k: v for k, v in checks.items() if v > VERIFY_RTOL}
    if failed:
        raise RuntimeError(f"{label} GATE FAILED: {failed}. Refusing to trust it for comparison.")
    print(f"{label} gate PASSED.")


def write_csv(rows):
    path = OUTPUT_DIR / "summary.csv"
    with path.open("w") as f:
        f.write(",".join(CSV_HEADER) + "\n")
        for row in rows:
            f.write(",".join(str(row[k]) for k in CSV_HEADER) + "\n")


def write_loss_history_csv(learning_rate, loss_history):
    path = OUTPUT_DIR / f"training_loss_lr{lr_tag(learning_rate)}.csv"
    with path.open("w") as f:
        f.write("epoch,loss\n")
        for epoch_idx, loss_value in enumerate(loss_history, start=1):
            f.write(f"{epoch_idx},{loss_value}\n")


def write_summary(rows, flags=None):
    lines = [
        "Experiment 3 -- learning-rate sensitivity (Nx=500, Nt=10, seed=42, epochs=1000).",
        f"Canonical reference LR={CANONICAL_LR} (epoch=1000 only, reused/reconstructed, NOT retrained). "
        f"New LRs (freshly trained, 250/500/1000 recorded): {LR_VALUES_NEW}.",
        "RNG note: collocation points, Wx/Wt/bx/bt, and initial coefficients are bit-identical "
        "across all three new LR runs (LR affects only optimizer.step(), never a sampling call); "
        "per-epoch BC/IC draws are therefore also bit-identical across them epoch-for-epoch. "
        "See module docstring for the full argument.",
        "",
    ]
    for row in rows:
        lines.append(f"--- LR={row['learning_rate']}, epoch={row['epoch']} ---")
        for k in CSV_HEADER[2:]:
            lines.append(f"  {k} = {row[k]}")
    if flags:
        lines.append("")
        lines.append("Outcome flags (heuristic, based on global_relative_l2_error trend 250->500->1000):")
        for learning_rate, flag_text in flags.items():
            lines.append(f"  LR={learning_rate}: {flag_text}")
    (OUTPUT_DIR / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def plot_error_vs_lr(rows):
    final_rows = [r for r in rows if r["epoch"] == 1000]
    final_rows = sorted(final_rows, key=lambda r: r["learning_rate"])
    lrs = [r["learning_rate"] for r in final_rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lrs, [r["global_relative_l2_error"] for r in final_rows], "o-", label="global relative L2")
    ax.plot(lrs, [r["smooth_relative_l2_error"] for r in final_rows], "s-", label="smooth relative L2")
    ax.plot(lrs, [r["shock_relative_l2_error"] for r in final_rows], "^-", label="shock relative L2")
    ax.set_xscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("relative L2 error (epoch=1000)")
    ax.set_title("RF-only error vs. learning rate (Nx=500, Nt=10, seed=42, epoch=1000)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "error_vs_lr.png")
    plt.close(fig)


def plot_convergence_vs_epoch(rows):
    fig, ax = plt.subplots(figsize=(8, 5))
    for learning_rate in LR_VALUES_NEW:
        lr_rows = sorted([r for r in rows if r["learning_rate"] == learning_rate], key=lambda r: r["epoch"])
        ax.plot([r["epoch"] for r in lr_rows], [r["global_relative_l2_error"] for r in lr_rows],
                "o-", label=f"LR={learning_rate} (global)")
    canonical_rows = [r for r in rows if r["learning_rate"] == CANONICAL_LR]
    if canonical_rows:
        ax.axhline(canonical_rows[0]["global_relative_l2_error"], color="k", linestyle="--",
                   label=f"LR={CANONICAL_LR} canonical (epoch=1000 only)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("global relative L2 error")
    ax.set_title("Convergence: global relative L2 vs. epoch, per learning rate")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "convergence_vs_epoch.png")
    plt.close(fig)


def plot_training_loss_vs_epoch(loss_histories):
    fig, ax = plt.subplots(figsize=(8, 5))
    for learning_rate, loss_history in loss_histories.items():
        ax.plot(range(1, len(loss_history) + 1), loss_history, label=f"LR={learning_rate}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("total training loss")
    ax.set_yscale("log")
    ax.set_title("Per-epoch training loss, new learning rates")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "training_loss_vs_epoch.png")
    plt.close(fig)


def flag_lr_outcomes(rows):
    """Heuristic classification per new LR, based ONLY on solution-error
    trend (not training loss) -- see module docstring / CLI instructions.
    A: clearly better than canonical at epoch=1000.
    B: worse than canonical and converged (last-segment change small).
    C: still improving substantially at epoch=1000 -> longer training may help."""
    canonical_rows = [r for r in rows if r["learning_rate"] == CANONICAL_LR]
    canonical_global = canonical_rows[0]["global_relative_l2_error"] if canonical_rows else None

    flags = {}
    for learning_rate in LR_VALUES_NEW:
        lr_rows = {r["epoch"]: r for r in rows if r["learning_rate"] == learning_rate}
        e250, e500, e1000 = lr_rows[250]["global_relative_l2_error"], lr_rows[500]["global_relative_l2_error"], lr_rows[1000]["global_relative_l2_error"]
        improve_250_500 = e250 - e500
        improve_500_1000 = e500 - e1000
        # "Substantial" ongoing improvement requires BOTH a meaningful ratio
        # relative to the first-half improvement AND a non-negligible
        # absolute size relative to the epoch=500 error itself -- a 40%
        # ratio of a near-zero improvement is noise-level convergence, not
        # "still improving substantially" (caught by the smoke test below).
        still_improving = (
            improve_250_500 > 0
            and (improve_500_1000 / improve_250_500) > 0.2
            and improve_500_1000 > 0.02 * e500
        )
        better_than_canonical = canonical_global is not None and e1000 < canonical_global

        if still_improving:
            flag = (f"C: still improving substantially at epoch=1000 (500->1000 improvement is "
                    f"{100 * improve_500_1000 / improve_250_500 if improve_250_500 else float('nan'):.0f}% "
                    "of the 250->500 improvement) -- motivates a longer-training follow-up experiment, "
                    "NOT run here (capped at 1000 epochs as instructed).")
        elif better_than_canonical:
            flag = f"A: outperforms canonical LR={CANONICAL_LR} at epoch=1000 ({e1000:.6f} < {canonical_global:.6f})."
        else:
            flag = f"B: worse than canonical LR={CANONICAL_LR} at epoch=1000 and appears converged."
        flags[learning_rate] = flag
        print(f"LR={learning_rate}: {flag}")
    return flags


def print_table(rows):
    header = ["LR", "epoch", "global", "smooth", "shock", "IC", "pde_loss", "t1_global", "t1_smooth", "t1_shock"]
    print(" | ".join(f"{h:>11}" for h in header))
    for row in sorted(rows, key=lambda r: (r["learning_rate"], r["epoch"])):
        vals = [
            str(row["learning_rate"]), str(row["epoch"]),
            f"{row['global_relative_l2_error']:.6f}", f"{row['smooth_relative_l2_error']:.6f}",
            f"{row['shock_relative_l2_error']:.6f}", f"{row['ic_relative_l2_error']:.6f}",
            f"{row['final_pde_loss']:.6e}",
            f"{row['t1_global_relative_l2_error']:.6f}", f"{row['t1_smooth_relative_l2_error']:.6f}",
            f"{row['t1_shock_relative_l2_error']:.6f}",
        ]
        print(" | ".join(f"{v:>11}" for v in vals))


def main():
    torch.set_default_dtype(DTYPE)
    device = get_device()
    print(device)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Fixed, deterministic evaluation grids -- built once, RNG-independent,
    # reused for the canonical reference and every new LR/checkpoint.
    t_bd_fixed = torch.linspace(0, 1, M_BD).reshape(-1, 1).to(device)
    x_int_fixed = torch.linspace(-1, 1, M_INT).reshape(-1, 1).to(device)
    x_test, t_test = make_test_grid(M_TEST, N_TEST_TIMES, device)
    y_true = make_reference_values(t_test, x_test, NV)
    x_shock, t_shock = build_shock_grid(t_test, device)
    y_true_shock = make_reference_values(t_shock, x_shock, NV)
    x_ic = torch.linspace(-1, 1, IC_GRID_N_X).reshape(-1, 1).to(device)
    t_ic = torch.zeros_like(x_ic).to(device)
    y_ic_true = g(x_ic).detach().cpu().numpy().reshape(-1)
    fixed_grids = (t_bd_fixed, x_int_fixed, x_test, t_test, y_true, x_shock, t_shock, y_true_shock, x_ic, t_ic, y_ic_true)

    rows = []

    # Canonical LR reference: reconstructed, NOT retrained. Only its real
    # epoch=1000 result exists -- no 250/500 values are fabricated.
    checkpoint42 = nx500long.load_source_checkpoint()
    model42 = nx500long.build_model_from_checkpoint(checkpoint42, device)
    set_seed(checkpoint42["seed"])
    x_train_42, t_train_42 = sample_training_points(checkpoint42["M_TRAIN"], device)
    row42, _ = evaluate_at_epoch(model42, CANONICAL_LR, 1000, x_train_42, t_train_42, *fixed_grids, device)
    verify_gate(row42, f"Canonical LR={CANONICAL_LR} (epoch=1000) reference")
    rows.append(row42)
    write_csv(rows)
    write_summary(rows)

    loss_histories = {}
    for learning_rate in LR_VALUES_NEW:
        print(f"=== LR={learning_rate} ===")
        set_seed(SEED)
        x_train, t_train = sample_training_points(M_TRAIN, device)
        model = BurgersRF(NX, NT, SIGMA_X, SIGMA_T, use_local=False, device=device).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)

        loss_history = []
        cumulative_epoch = 0
        for target_epoch, segment_len in zip(CHECKPOINT_EPOCHS, SEGMENT_LENGTHS):
            segment_losses = train(model, optimizer, x_train, t_train, M_BD, M_INT, NV, g, device, epochs=segment_len)
            loss_history.extend(segment_losses)
            cumulative_epoch += segment_len
            assert cumulative_epoch == target_epoch

            row, checkpoint = evaluate_at_epoch(
                model, learning_rate, target_epoch, x_train, t_train, *fixed_grids, device,
            )
            rows.append(row)
            torch.save(checkpoint, OUTPUT_DIR / f"checkpoint_lr{lr_tag(learning_rate)}_epoch{target_epoch}.pt")
            write_csv(rows)       # rewritten after EVERY checkpoint/evaluation
            write_summary(rows)   # rewritten after EVERY checkpoint/evaluation
            write_loss_history_csv(learning_rate, loss_history)

        loss_histories[learning_rate] = loss_history
        plot_training_loss_vs_epoch(loss_histories)  # regenerated after every completed LR

    plot_error_vs_lr(rows)
    plot_convergence_vs_epoch(rows)
    flags = flag_lr_outcomes(rows)
    write_summary(rows, flags=flags)
    print_table(rows)

    print("[sweep_rf_learning_rate] done.")


if __name__ == "__main__":
    main()
