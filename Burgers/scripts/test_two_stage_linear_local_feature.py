import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(PROJECT_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from burgers_rf.config import (  # noqa: E402
    DTYPE,
    IC_BC_WEIGHT,
    LR,
    M_BD,
    M_INT,
    M_TEST,
    N_TEST_TIMES,
    NV,
    WEIGHT_DECAY,
    set_seed,
)
from burgers_rf.data import g, make_reference_values, make_test_grid, sample_training_points  # noqa: E402
from burgers_rf.model import BurgersRF  # noqa: E402
from burgers_rf.local_features import GaussianLocal  # noqa: E402
from burgers_rf.physics import boundary_residual, burgers_residual, initial_residual  # noqa: E402
from burgers_rf.evaluation import evaluate_error_metrics  # noqa: E402
from burgers_rf.diagnostics import save_component_diagnostics  # noqa: E402
from sweep_local_width import build_shock_grid  # noqa: E402

# TWO-STAGE TRAINING experiment. Stage 1 (RF-only PDE training) is ALREADY
# COMPLETE -- loaded from the verified checkpoint_complete.pt and frozen
# exactly. Stage 2 trains ONLY the scalar linear_t local coefficient d,
# initialized at exactly 0, against the SAME training objective/optimizer
# settings as normal Burgers training (LR, WEIGHT_DECAY, IC_BC_WEIGHT,
# EPOCHS=1000 all unchanged from config.py). No RF/local-basis code is
# modified; physics.py's burgers_residual/boundary_residual/initial_residual
# are reused UNMODIFIED, with the per-epoch loop reimplemented locally
# (mathematically identical to physics.train/loss_fn, call-for-call) only
# because physics.train()'s current API does not expose per-epoch d/loss
# components or the pre-first-step gradient needed for this diagnostic.
#
# device="cpu" throughout is a CORRECTNESS requirement, not just
# portability: model.py's known nn.Parameter(...).to(device) bug would
# silently reproduce on a fresh CUDA construction, making the strict
# checkpoint load fail (Wx/Wt/bx/bt would be absent from a CUDA-constructed
# model's own state_dict()). Loading on CPU takes the harmless .to("cpu")
# no-op path, matching every other checkpoint-loading script in this study.

CHECKPOINT_PATH = PROJECT_DIR / "outputs" / "pde_trained_rf_residual" / "checkpoint_complete.pt"
CANONICAL_GLOBAL = 0.27149430095802507
CANONICAL_SMOOTH = 0.2443827084159696
CANONICAL_SHOCK = 0.8390487998889169
VERIFY_RTOL = 1e-4

CENTER = 0.014
WIDTH = 0.007646112
EPOCHS = 1000
IC_GRID_N_X = 1000

# Reference results for the final comparison table only -- not used
# anywhere in the training/initialization logic.
D_JOINT = 0.02335906311371956
JOINT_GLOBAL = 0.2863551208182298
JOINT_SMOOTH = 0.25478499908981045
JOINT_SHOCK = 0.9391014811156511
D_FIXED_RF_WEIGHTED_LOSS_OPTIMUM = -0.010  # approximate, from the sweep diagnostic

OUTPUT_DIR = PROJECT_DIR / "outputs" / "two_stage_linear_local_feature"


def relative_l2(y_true, y_pred):
    return float(np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true))


def load_and_retrofit_model(device="cpu"):
    """Load the complete, verified Stage-1 RF-only checkpoint with a strict
    state_dict match (guarantees Wx/Wt/bx/bt/model.0.weight are exactly the
    canonical checkpoint's -- never a freshly regenerated basis), then
    retrofit the linear_t local feature by assigning the SAME attributes
    BurgersRF.__init__ would set for use_local=True, local_time_mode=
    "linear_t" -- not a different code path. Finally freeze model.0.weight
    (the only parameter that defaults to requires_grad=True)."""
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model = BurgersRF(
        checkpoint["NX"], checkpoint["NT"], checkpoint["SIGMA_X"], checkpoint["SIGMA_T"],
        use_local=False, device=device,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    model.use_local = True
    model.local_type = "gaussian"
    model.local_time_mode = "linear_t"
    model.local_feature = GaussianLocal(center=CENTER, width=WIDTH)
    model.local_coefficients = torch.nn.Parameter(torch.zeros(1))

    model.model[0].weight.requires_grad_(False)

    return model, checkpoint


def verify_pretraining_state(model, checkpoint, device):
    """Before Stage 2 training begins: verify d==0, verify EXACTLY one
    trainable parameter (local_coefficients), print all parameter names/
    shapes/requires_grad, and verify the d=0 forward pass reproduces the
    canonical RF-only metrics. Raises and stops before training on any
    failure."""
    print("Parameters (name, shape, requires_grad):")
    trainable = []
    for name, p in model.named_parameters():
        print(f"  {name}: shape={tuple(p.shape)}, requires_grad={p.requires_grad}")
        if p.requires_grad:
            trainable.append(name)

    if trainable != ["local_coefficients"]:
        raise RuntimeError(
            f"Expected exactly one trainable parameter ['local_coefficients'], got {trainable}. "
            "Refusing to proceed -- Stage 2 would not be a clean frozen-RF experiment."
        )
    print(f"Confirmed: only trainable parameter is {trainable}.")

    d_init = model.local_coefficients.item()
    if d_init != 0.0:
        raise RuntimeError(f"Expected d==0.0 at initialization, got {d_init}. Refusing to proceed.")
    print(f"Confirmed: d == {d_init} at initialization.")

    nv = checkpoint.get("NV", NV)
    x_test, t_test = make_test_grid(M_TEST, N_TEST_TIMES, device)
    y_true = make_reference_values(t_test, x_test, nv)
    x_shock, t_shock = build_shock_grid(t_test, device)
    y_true_shock = make_reference_values(t_shock, x_shock, nv)

    with torch.no_grad():
        y_pred = model(x_test, t_test).cpu().numpy().reshape(-1)
        y_pred_shock = model(x_shock, t_shock).cpu().numpy().reshape(-1)
    metrics = evaluate_error_metrics(x_test, y_true, y_pred, y_true_shock, y_pred_shock)

    def rel_diff(a, b):
        return abs(a - b) / abs(b)

    checks = {
        "global": rel_diff(metrics["global_relative_l2_error"], CANONICAL_GLOBAL),
        "smooth": rel_diff(metrics["smooth_relative_l2_error"], CANONICAL_SMOOTH),
        "shock": rel_diff(metrics["shock_relative_l2_error"], CANONICAL_SHOCK),
    }
    print("d=0 pre-training verification vs. canonical metrics:")
    for k, v in checks.items():
        print(f"  {k}: relative difference = {v:.3e} (tolerance {VERIFY_RTOL:.1e})")
    failed = {k: v for k, v in checks.items() if v > VERIFY_RTOL}
    if failed:
        raise RuntimeError(
            f"Pre-training verification FAILED: {failed}. Refusing to start Stage 2 training."
        )
    print("Pre-training verification PASSED.")
    return metrics


def reconstruct_interior_points(checkpoint, device="cpu"):
    """Reproduce the EXACT seed-42 interior collocation points, same
    approach as reconstruct_pde_trained_rf_checkpoint.py /
    diagnose_fixed_rf_linear_d_sweep.py."""
    set_seed(checkpoint["seed"])
    x_train, t_train = sample_training_points(checkpoint["M_TRAIN"], device)
    return x_train, t_train


def snapshot_frozen_tensors(model):
    return {
        "Wx": model.Wx.clone(),
        "Wt": model.Wt.clone(),
        "bx": model.bx.clone(),
        "bt": model.bt.clone(),
        "model.0.weight": model.model[0].weight.clone(),
    }


def check_frozen_invariance(model, before):
    diffs = {}
    after = {
        "Wx": model.Wx,
        "Wt": model.Wt,
        "bx": model.bx,
        "bt": model.bt,
        "model.0.weight": model.model[0].weight,
    }
    for key in before:
        diffs[key] = torch.max(torch.abs(after[key] - before[key])).item()
    return diffs


def train_stage2(model, x_train, t_train, device):
    """Stage-2 training loop. Mathematically identical to physics.train's
    loop body (same burgers_residual/boundary_residual/initial_residual
    calls, same IC_BC_WEIGHT-weighted combination as physics.loss_fn, same
    backward()/step() order) -- reimplemented locally, not as a modification
    to physics.py, solely to expose per-epoch d/loss components and the
    pre-first-step gradient of d, which physics.train()'s current API does
    not return."""
    optimizer = optim.Adam([model.local_coefficients], lr=LR, weight_decay=WEIGHT_DECAY)

    history = []
    grad_d_initial = None
    d_after_first_step = None

    for epoch in range(EPOCHS):
        optimizer.zero_grad()

        residual_pde = burgers_residual(model, x_train, t_train, NV)
        l_pde = torch.mean(residual_pde**2)
        residual_bd = boundary_residual(model, M_BD, device)
        l_bc = torch.mean(residual_bd**2)
        residual_int = initial_residual(model, M_INT, g, device)
        l_ic = torch.mean(residual_int**2)
        loss = l_pde + IC_BC_WEIGHT * (l_ic + l_bc)

        # retain_graph=True matches physics.train's own loss.backward() call
        # exactly (physics.py line 52) -- required because x_train/t_train
        # are the same reused leaf tensors across epochs.
        loss.backward(retain_graph=True)

        if epoch == 0:
            grad_d_initial = model.local_coefficients.grad.item()

        optimizer.step()

        d_now = model.local_coefficients.item()
        if epoch == 0:
            d_after_first_step = d_now

        history.append({
            "epoch": epoch,
            "d": d_now,
            "pde_loss": l_pde.item(),
            "ic_loss": l_ic.item(),
            "bc_loss": l_bc.item(),
            "total_loss": loss.item(),
        })

        if epoch % 100 == 0:
            print(f"epoch {epoch}: d={d_now:.10f}, pde_loss={l_pde.item():.6e}, total_loss={loss.item():.6e}")

    return history, grad_d_initial, d_after_first_step


def write_history_csv(history):
    header = ["epoch", "d", "pde_loss", "ic_loss", "bc_loss", "total_loss"]
    path = OUTPUT_DIR / "two_stage_training_history.csv"
    with path.open("w") as f:
        f.write(",".join(header) + "\n")
        for row in history:
            f.write(",".join(str(row[k]) for k in header) + "\n")


def plot_history(history):
    import matplotlib.pyplot as plt

    epochs = [r["epoch"] for r in history]
    d_values = [r["d"] for r in history]
    pde_losses = [r["pde_loss"] for r in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, d_values, "-")
    ax.axhline(0.0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel("epoch")
    ax.set_ylabel("d")
    ax.set_title("Stage-2 learned d vs epoch (frozen RF)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "d_vs_epoch.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, pde_losses, "-", color="tab:red")
    ax.set_xlabel("epoch")
    ax.set_ylabel("PDE loss")
    ax.set_title("Stage-2 PDE loss vs epoch (frozen RF)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "pde_loss_vs_epoch.png")
    plt.close(fig)


def write_summary(
    history, grad_d_initial, d_after_first_step, final_metrics, ic_relative_l2,
    diag_stats, frozen_diffs,
):
    d_final = history[-1]["d"]
    lines = []
    lines.append("=== Stage-2 (frozen RF, linear_t) result ===")
    lines.append(f"d at initialization = 0.0")
    lines.append(f"grad_d at initialization (epoch 0, before first step) = {grad_d_initial}")
    lines.append(f"d after first optimizer step (epoch 0) = {d_after_first_step}")
    lines.append(f"final d (epoch {EPOCHS - 1}) = {d_final}")
    lines.append(f"sign of final d: {'negative' if d_final < 0 else 'positive' if d_final > 0 else 'zero'}")
    lines.append("")
    lines.append(f"global_relative_l2_error = {final_metrics['global_relative_l2_error']}")
    lines.append(f"smooth_relative_l2_error = {final_metrics['smooth_relative_l2_error']}")
    lines.append(f"shock_relative_l2_error = {final_metrics['shock_relative_l2_error']}")
    lines.append(f"ic_relative_l2_error = {ic_relative_l2}")
    lines.append(f"ratio_full = {diag_stats['ratio_full']}")
    lines.append(f"ratio_shock = {diag_stats['ratio_shock']}")
    lines.append(f"local_coef_norm = {diag_stats['coef_norm']}")
    lines.append(f"final_pde_loss = {history[-1]['pde_loss']}")
    lines.append(f"final_unweighted_ic_loss = {history[-1]['ic_loss']}")
    lines.append(f"final_unweighted_bc_loss = {history[-1]['bc_loss']}")
    lines.append("")
    lines.append("Frozen-RF invariance check (max |after - before|, expected exactly 0.0):")
    for k, v in frozen_diffs.items():
        lines.append(f"  {k}: {v}")
    lines.append("")

    lines.append("=== Comparison ===")
    lines.append("A. RF-only baseline (Stage 1):")
    lines.append(f"   d = 0, global = {CANONICAL_GLOBAL}, smooth = {CANONICAL_SMOOTH}, shock = {CANONICAL_SHOCK}")
    lines.append("B. JOINT linear_t training:")
    lines.append(f"   d = {D_JOINT}, global = {JOINT_GLOBAL}, smooth = {JOINT_SMOOTH}, shock = {JOINT_SHOCK}")
    lines.append("C. TWO-STAGE / frozen-RF linear_t (this result):")
    lines.append(
        f"   d = {d_final}, global = {final_metrics['global_relative_l2_error']}, "
        f"smooth = {final_metrics['smooth_relative_l2_error']}, shock = {final_metrics['shock_relative_l2_error']}"
    )
    lines.append("")
    lines.append(
        f"Fixed-RF sweep weighted-training-loss optimum was approximately "
        f"d* ~ {D_FIXED_RF_WEIGHTED_LOSS_OPTIMUM} -- compare Stage-2 final d against THIS value, "
        "since both optimize the identical weighted PDE/IC/BC training objective with RF frozen. "
        "Do NOT compare Stage-2's final d magnitude directly against the t=1 post-hoc solution-"
        "error oracle (~-0.6578595) without noting that the oracle minimizes t=1 solution error, "
        "not the space-time weighted training objective -- these are different objectives."
    )
    lines.append("")
    lines.append(
        f"NOTE: WEIGHT_DECAY={WEIGHT_DECAY} is large and (for vanilla Adam, not decoupled AdamW) "
        "adds weight_decay*d directly to the gradient every step, pulling d toward 0 in proportion "
        "to its own current magnitude. This was deliberately preserved unchanged (per instructions) "
        "rather than treated as a confound to remove, but it should be considered when interpreting "
        "how far d was able to move from zero."
    )

    (OUTPUT_DIR / "two_stage_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    torch.set_default_dtype(DTYPE)
    device = "cpu"

    model, checkpoint = load_and_retrofit_model(device)
    verify_pretraining_state(model, checkpoint, device)

    x_train, t_train = reconstruct_interior_points(checkpoint, device)

    frozen_before = snapshot_frozen_tensors(model)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    history, grad_d_initial, d_after_first_step = train_stage2(model, x_train, t_train, device)

    frozen_diffs = check_frozen_invariance(model, frozen_before)

    write_history_csv(history)
    plot_history(history)

    nv = checkpoint.get("NV", NV)
    x_test, t_test = make_test_grid(M_TEST, N_TEST_TIMES, device)
    y_true = make_reference_values(t_test, x_test, nv)
    x_shock, t_shock = build_shock_grid(t_test, device)
    y_true_shock = make_reference_values(t_shock, x_shock, nv)

    model.eval()
    with torch.no_grad():
        y_pred = model(x_test, t_test).cpu().numpy().reshape(-1)
        y_pred_shock = model(x_shock, t_shock).cpu().numpy().reshape(-1)
    final_metrics = evaluate_error_metrics(x_test, y_true, y_pred, y_true_shock, y_pred_shock)

    x_ic = torch.linspace(-1, 1, IC_GRID_N_X).reshape(-1, 1).to(device)
    t_ic = torch.zeros_like(x_ic).to(device)
    y_ic_true = g(x_ic).detach().cpu().numpy().reshape(-1)
    with torch.no_grad():
        y_ic_pred = model(x_ic, t_ic).cpu().numpy().reshape(-1)
    ic_relative_l2 = relative_l2(y_ic_true, y_ic_pred)

    diag_stats = save_component_diagnostics(
        model, "two_stage_frozen_rf", x_test, t_test, y_true, x_shock, t_shock, y_true_shock, OUTPUT_DIR
    )

    write_summary(history, grad_d_initial, d_after_first_step, final_metrics, ic_relative_l2, diag_stats, frozen_diffs)

    print("[test_two_stage_linear_local_feature] done.")


if __name__ == "__main__":
    main()
