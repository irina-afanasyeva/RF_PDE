"""Shared trainer for three modes: the conventional-PINN "baseline", the
reference-faithful "adaptive" SA-PINN, and the "fixed_weight" loss-scale-
matched control. A single parameterized entry point (`run(mode=...)`) is
used for all three on purpose, so they are structurally guaranteed to
share architecture, training points, and seed -- the only difference
between them is the attention-weight path (see attention.py):
  - baseline:      weights fixed at 1.0 (effective mask 1**2=1, i.e.
                    exactly unweighted MSE).
  - adaptive:       weights trainable, initialized as in the reference
                    (col_weights=100, u_weights~U(0,1)), updated by
                    gradient ASCENT during the Adam phase, frozen during
                    L-BFGS.
  - fixed_weight:   weights initialized IDENTICALLY to adaptive (same
                    values, same RNG draw), but requires_grad=False for
                    the entire run -- isolates whether any
                    adaptive-vs-baseline difference comes from the
                    per-point ADAPTIVITY of the weights, or merely from
                    the larger fixed effective loss scale that the
                    lambda=100 initialization introduces on its own.

Safeguards enforced here (see task spec):
  - "weighted_pde_loss" and "unweighted_pde_mse" are always separate,
    distinctly named fields -- never conflated.
  - The reference solution is used ONLY inside evaluate_relative_l2(),
    never inside compute_losses() (physics-only training).
  - L-BFGS is PyTorch's native optimizer (torch.optim.LBFGS), never
    silently swapped for more Adam iterations.

OPERATIONAL FEATURES (added for long-running GPU experiments):
  - Output directories include the seed (outputs/.../{mode}/{budget}/seed{N}/)
    and a completed run (config.json present) is never silently overwritten.
  - Periodic, atomically-written checkpoints during the Adam phase, capturing
    model/attention-weight/all-optimizer state, RNG state, and the completed
    epoch count -- resumable via run(..., resume_from=<checkpoint path>).
  - L-BFGS is executed in CHUNKS (repeated .step(closure) calls on the SAME
    optimizer instance) purely to enable checkpointing and progress
    visibility. This was NOT assumed safe -- it was empirically verified
    bit-identical to one uninterrupted call for this problem via
    scripts/test_sapinn_lbfgs_chunking.py (chunk sizes 1, 10, and 20 all
    tested against single-call budgets up to 300 iterations; see that
    script and docs/RESEARCH_LOG.md for the full equivalence writeup and
    its one disclosed untested edge case: a chunk boundary coinciding
    exactly with the optimizer's own tolerance-based early-stopping point
    was never empirically triggered in testing).
  - Per-iteration nonfinite checks only touch the scalar total_objective
    (cheap); the full col_weights/u_weights tensors are only checked at the
    logging cadence, to reduce GPU synchronization overhead.
  - Cheap loss-only logging (LOG_EVERY) is decoupled from the more
    expensive full relative-L2 evaluation (EVAL_EVERY_ADAM).
  - closure_evals is tracked as its own CSV column, separate from
    epoch/n_iter.
"""
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

BURGERS_DIR = Path(__file__).resolve().parents[1]
if str(BURGERS_DIR) not in sys.path:
    sys.path.insert(0, str(BURGERS_DIR))
if str(BURGERS_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(BURGERS_DIR / "scripts"))

from burgers_rf.data import make_test_grid, make_reference_values  # noqa: E402
from burgers_rf.evaluation import evaluate_error_metrics  # noqa: E402
from sweep_local_width import build_shock_grid  # noqa: E402

from . import attention, config, sampling
from .model import PINN
from . import physics

LOSS_KEYS = (
    "unweighted_pde_mse", "weighted_pde_loss",
    "unweighted_ic_mse", "weighted_ic_loss",
    "bc_mse", "total_objective",
)
EVAL_KEYS = ("global_relative_l2_error", "smooth_relative_l2_error", "shock_relative_l2_error")
CSV_HEADER = (
    ["phase", "epoch", "closure_evals", "wall_clock_s"]
    + list(LOSS_KEYS)
    + list(EVAL_KEYS)
    + ["col_weight_mean", "col_weight_std", "col_weight_min", "col_weight_max",
       "u_weight_mean", "u_weight_std", "nonfinite"]
)


def compute_losses(model, tensors, col_weights, u_weights, nv=None):
    """Physics-only loss computation. Does NOT use the analytic reference
    solution anywhere -- see evaluate_relative_l2 for the (evaluation-only)
    use of the reference solution."""
    nv = config.NV if nv is None else nv
    f_pred = physics.pde_residual(model, tensors["x_f"], tensors["t_f"], nv)
    ic_err = physics.ic_prediction_error(model, tensors["x0"], tensors["t0"])
    u_lb, u_ub = physics.boundary_predictions(
        model, tensors["x_lb"], tensors["t_lb"], tensors["x_ub"], tensors["t_ub"]
    )

    unweighted_pde_mse = torch.mean(f_pred**2)
    weighted_pde_loss = torch.mean((attention.identity_mask(col_weights) * f_pred) ** 2)
    unweighted_ic_mse = torch.mean(ic_err**2)
    weighted_ic_loss = torch.mean((attention.identity_mask(u_weights) * ic_err) ** 2)
    # [CODE] BC loss has NO self-adaptive weight in the reference Burgers example.
    bc_mse = torch.mean(u_lb**2) + torch.mean(u_ub**2)

    total_objective = weighted_pde_loss + weighted_ic_loss + bc_mse
    return {
        "unweighted_pde_mse": unweighted_pde_mse,
        "weighted_pde_loss": weighted_pde_loss,
        "unweighted_ic_mse": unweighted_ic_mse,
        "weighted_ic_loss": weighted_ic_loss,
        "bc_mse": bc_mse,
        "total_objective": total_objective,
    }


def evaluate_relative_l2(model, device):
    """EVALUATION ONLY -- the analytic reference solution must never be
    used inside compute_losses()/training. Reuses burgers_rf's existing
    reference-solution and evaluation-metric machinery unmodified."""
    x_test, t_test = make_test_grid(config.M_TEST, config.N_TEST_TIMES, device)
    x_test = x_test.to(dtype=config.DTYPE)
    t_test = t_test.to(dtype=config.DTYPE)
    y_true = make_reference_values(t_test, x_test, config.NV)

    x_shock, t_shock = build_shock_grid(t_test, device)
    x_shock = x_shock.to(dtype=config.DTYPE)
    t_shock = t_shock.to(dtype=config.DTYPE)
    y_true_shock = make_reference_values(t_shock, x_shock, config.NV)

    with torch.no_grad():
        y_pred = model(x_test, t_test).cpu().numpy().reshape(-1)
        y_pred_shock = model(x_shock, t_shock).cpu().numpy().reshape(-1)

    return evaluate_error_metrics(x_test, y_true, y_pred, y_true_shock, y_pred_shock)


def _scalar_nonfinite(tensor):
    """Cheap, scalar-only check -- safe to call every iteration without
    meaningfully adding GPU synchronization overhead."""
    return not torch.isfinite(tensor).all().item()


def _any_nonfinite(*tensors):
    """Full-tensor check -- only call at the logging cadence, not every
    iteration (see module docstring)."""
    return any(not torch.isfinite(t).all().item() for t in tensors)


def _weight_stats(col_weights, u_weights):
    return {
        "col_weight_mean": col_weights.mean().item(),
        "col_weight_std": col_weights.std().item(),
        "col_weight_min": col_weights.min().item(),
        "col_weight_max": col_weights.max().item(),
        "u_weight_mean": u_weights.mean().item(),
        "u_weight_std": u_weights.std().item(),
    }


def _atomic_save(obj, path):
    """Write-to-temp-then-rename so a crash mid-write never leaves a
    corrupted checkpoint at `path` -- os.replace is atomic on POSIX and
    Windows (same filesystem)."""
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def _save_attention_snapshot(output_dir, tag, col_weights, u_weights):
    snap_dir = Path(output_dir) / "attention_snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        snap_dir / f"{tag}.npz",
        col_weights=col_weights.detach().cpu().numpy().reshape(-1),
        u_weights=u_weights.detach().cpu().numpy().reshape(-1),
    )


def run(
    mode, budget_name, seed, output_dir, device=None,
    resume_from=None, overwrite=False,
    checkpoint_every_adam=None, eval_every_adam=None, lbfgs_chunk_size=None,
):
    assert mode in ("baseline", "adaptive", "fixed_weight"), mode
    assert budget_name in config.BUDGETS, budget_name
    device = device or config.get_device()
    checkpoint_every_adam = checkpoint_every_adam or config.CHECKPOINT_EVERY_ADAM
    eval_every_adam = eval_every_adam or config.EVAL_EVERY_ADAM
    lbfgs_chunk_size = lbfgs_chunk_size or config.LBFGS_CHUNK_SIZE

    output_dir = Path(output_dir)
    done_marker = output_dir / "config.json"
    if resume_from is None and done_marker.exists() and not overwrite:
        raise FileExistsError(
            f"{output_dir} already contains a completed or in-progress run "
            f"({done_marker} exists). Refusing to overwrite. Pass overwrite=True "
            "(CLI: --force) to intentionally replace it, or resume_from=<checkpoint> "
            "(CLI: --resume) to continue an interrupted run."
        )
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)

    resume_state = None
    if resume_from is not None:
        # ALWAYS load onto CPU first, regardless of the target training
        # device. A checkpoint is a heterogeneous bundle: model/optimizer
        # tensors are fine to relocate, but torch_rng_state (a CPU
        # ByteTensor, per torch.get_rng_state()) and cuda_rng_state (a list
        # of CPU ByteTensors, per torch.cuda.get_rng_state_all() -- these
        # describe CUDA generator state but are themselves always CPU-side
        # byte buffers) MUST stay on CPU to remain valid inputs to
        # torch.set_rng_state()/torch.cuda.set_rng_state_all(). Passing
        # map_location=device (e.g. "cuda:0") here blindly relocated EVERY
        # tensor in the pickle, including these, which is exactly what
        # caused "TypeError: RNG state must be a torch.ByteTensor" on the
        # GPU server (a CUDA-resident tensor is no longer accepted there).
        # Model/optimizer state tensors still end up on the correct
        # training device below, via net.load_state_dict()/
        # optimizer.load_state_dict() (which copy into the already
        # device-resident net/optimizers constructed further down) and via
        # explicit .to(device) casts for col_weights/u_weights -- none of
        # that depends on the checkpoint itself being pre-relocated.
        resume_state = torch.load(resume_from, map_location="cpu", weights_only=False)
        mode = resume_state["mode"]
        budget_name = resume_state["budget_name"]
        seed = resume_state["seed"]
    budget = config.BUDGETS[budget_name]

    # Fixed training points for the WHOLE run (see sampling.py docstring).
    # Built BEFORE any mode-dependent branching so all three modes share
    # identical collocation/IC/BC points given the same seed. Also used to
    # deterministically reconstruct the pre-training state when resuming
    # (the points themselves are never saved in the checkpoint).
    config.set_all_seeds(seed)
    tensors = sampling.build_training_tensors(seed, device)

    # Network construction consumes RNG (Xavier-normal init) BEFORE any
    # mode-dependent attention-weight RNG consumption, so all three modes
    # also share identical initial network weights given the same seed.
    net = PINN().to(device=device, dtype=config.DTYPE)

    # Seeded fresh, right before the mode branch, unconditionally -- so
    # adaptive and fixed_weight consume IDENTICAL random draws for
    # u_weights (same generator, same seed, same call shape), giving them
    # bit-identical initial attention parameters.
    cpu_generator = torch.Generator(device="cpu").manual_seed(seed)
    if mode == "adaptive":
        col_weights, u_weights = attention.make_adaptive_weights(
            config.N_F, config.N_0, device, cpu_generator
        )
    elif mode == "fixed_weight":
        col_weights, u_weights = attention.make_fixed_weights(
            config.N_F, config.N_0, device, cpu_generator
        )
    else:
        col_weights, u_weights = attention.make_baseline_weights(config.N_F, config.N_0, device)

    net_optimizer = torch.optim.Adam(net.parameters(), lr=config.ADAM_LR, betas=config.ADAM_BETAS)
    col_optimizer = u_optimizer = None
    if mode == "adaptive":
        # Gradient ASCENT on the attention weights: PyTorch's native
        # maximize=True is functionally identical to the reference's
        # manual gradient negation (apply_gradients(zip([-grads], ...))).
        col_optimizer = torch.optim.Adam([col_weights], lr=config.ADAM_LR, betas=config.ADAM_BETAS, maximize=True)
        u_optimizer = torch.optim.Adam([u_weights], lr=config.ADAM_LR, betas=config.ADAM_BETAS, maximize=True)
    # fixed_weight: no ascent optimizer at all -- col_weights/u_weights have
    # requires_grad=False, so there is nothing to step; frozen by
    # construction, not merely by omission of an update rule.

    start_phase = "adam"
    start_adam_epoch = 0          # next Adam epoch to run
    start_lbfgs_n_iter_done = 0   # L-BFGS iterations already completed
    lbfgs_optimizer_state_to_restore = None

    if resume_state is not None:
        net.load_state_dict(resume_state["model_state_dict"])
        with torch.no_grad():
            col_weights.copy_(resume_state["col_weights"].to(device=device, dtype=config.DTYPE))
            u_weights.copy_(resume_state["u_weights"].to(device=device, dtype=config.DTYPE))
        net_optimizer.load_state_dict(resume_state["net_optimizer_state_dict"])
        if mode == "adaptive":
            col_optimizer.load_state_dict(resume_state["col_optimizer_state_dict"])
            u_optimizer.load_state_dict(resume_state["u_optimizer_state_dict"])
        torch.set_rng_state(resume_state["torch_rng_state"])
        np.random.set_state(resume_state["numpy_rng_state"])
        if torch.cuda.is_available() and resume_state.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(resume_state["cuda_rng_state"])

        start_phase = resume_state["phase"]
        if start_phase == "adam":
            start_adam_epoch = resume_state["epoch"] + 1
        else:
            start_lbfgs_n_iter_done = resume_state["lbfgs_n_iter"]
            lbfgs_optimizer_state_to_restore = resume_state.get("lbfgs_optimizer_state_dict")
        print(
            f"[{mode}/{budget_name}] RESUMED from {resume_from}: phase={start_phase}, "
            f"adam_epoch={start_adam_epoch}, lbfgs_n_iter_done={start_lbfgs_n_iter_done}"
        )

    metrics_rows = []
    warnings_log = []
    start_time = time.time()
    stopped_early = False
    closure_evals = resume_state["closure_evals"] if resume_state is not None else 0

    def save_checkpoint(phase, epoch_or_niter, lbfgs_opt=None):
        ckpt = {
            "mode": mode, "budget_name": budget_name, "seed": seed,
            "phase": phase,
            "epoch": epoch_or_niter if phase == "adam" else None,
            "lbfgs_n_iter": epoch_or_niter if phase == "lbfgs" else None,
            "model_state_dict": net.state_dict(),
            "col_weights": col_weights.detach().cpu(),
            "u_weights": u_weights.detach().cpu(),
            "net_optimizer_state_dict": net_optimizer.state_dict(),
            "col_optimizer_state_dict": col_optimizer.state_dict() if col_optimizer is not None else None,
            "u_optimizer_state_dict": u_optimizer.state_dict() if u_optimizer is not None else None,
            "lbfgs_optimizer_state_dict": lbfgs_opt.state_dict() if lbfgs_opt is not None else None,
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "closure_evals": closure_evals,
        }
        tag = f"{phase}_{epoch_or_niter}"
        _atomic_save(ckpt, output_dir / "checkpoints" / f"{tag}.pt")
        _atomic_save(ckpt, output_dir / "checkpoints" / "latest.pt")
        _save_attention_snapshot(output_dir, tag, col_weights, u_weights)

    # ------------------------------------------------------------------
    # Adam phase
    # ------------------------------------------------------------------
    if start_phase == "adam":
        print(f"[{mode}/{budget_name}] starting Adam phase ({budget['adam_iters']} iters, "
              f"resuming at epoch {start_adam_epoch})" if start_adam_epoch else
              f"[{mode}/{budget_name}] starting Adam phase ({budget['adam_iters']} iters)")
        losses = None
        for epoch in range(start_adam_epoch, budget["adam_iters"]):
            net_optimizer.zero_grad()
            if mode == "adaptive":
                col_optimizer.zero_grad()
                u_optimizer.zero_grad()

            losses = compute_losses(net, tensors, col_weights, u_weights)
            losses["total_objective"].backward()
            closure_evals += 1

            net_optimizer.step()
            if mode == "adaptive":
                col_optimizer.step()
                u_optimizer.step()

            # Cheap, scalar-only check every iteration (see module docstring).
            nonfinite = _scalar_nonfinite(losses["total_objective"])
            if nonfinite:
                warnings_log.append(f"epoch {epoch}: nonfinite total_objective, stopping Adam phase early")

            do_log = epoch % config.LOG_EVERY == 0 or epoch == budget["adam_iters"] - 1 or nonfinite
            if do_log:
                # Full-tensor nonfinite check only at logging cadence.
                nonfinite = nonfinite or _any_nonfinite(col_weights, u_weights)
                do_eval = epoch % eval_every_adam == 0 or epoch == budget["adam_iters"] - 1 or nonfinite
                eval_metrics = evaluate_relative_l2(net, device) if do_eval else dict.fromkeys(EVAL_KEYS, float("nan"))
                row = {
                    "phase": "adam", "epoch": epoch, "closure_evals": closure_evals,
                    "wall_clock_s": time.time() - start_time,
                    **{k: losses[k].item() for k in LOSS_KEYS},
                    **eval_metrics,
                    **_weight_stats(col_weights, u_weights),
                    "nonfinite": nonfinite,
                }
                metrics_rows.append(row)
                print(
                    f"  adam epoch {epoch}: total={row['total_objective']:.4e} "
                    f"pde_mse={row['unweighted_pde_mse']:.4e}"
                    + (f" global_L2={row['global_relative_l2_error']:.4e}" if do_eval else "")
                )

            if (epoch + 1) % checkpoint_every_adam == 0 or epoch == budget["adam_iters"] - 1 or nonfinite:
                save_checkpoint("adam", epoch)

            if nonfinite:
                stopped_early = True
                break
    else:
        print(f"[{mode}/{budget_name}] Adam phase already complete (resuming directly into L-BFGS).")

    # ------------------------------------------------------------------
    # L-BFGS phase -- executed in CHUNKS on a single optimizer instance,
    # empirically verified equivalent to one uninterrupted call (see
    # module docstring and scripts/test_sapinn_lbfgs_chunking.py).
    # ------------------------------------------------------------------
    if not stopped_early and budget["lbfgs_iters"] > 0:
        # IMPORTANT: torch.optim.LBFGS's max_iter is fixed at construction
        # and applies PER .step() call -- it must never exceed the budget's
        # remaining lbfgs_iters, or a single chunk call would silently run
        # far more iterations than requested (caught by the smoke test: an
        # uncapped chunk_size=500 against a 5-iteration smoke budget caused
        # visible divergence). All three defined budgets (smoke=5,
        # repo_default=100, paper=10000) divide evenly by this clamp, so no
        # chunk ever needs to be reconstructed mid-phase with a different
        # max_iter (that would be a different, NOT yet empirically verified,
        # code path -- see scripts/test_sapinn_lbfgs_chunking.py, which only
        # tested same-instance/fixed-max_iter chunking).
        effective_chunk_size = min(lbfgs_chunk_size, budget["lbfgs_iters"])
        assert budget["lbfgs_iters"] % effective_chunk_size == 0, (
            f"lbfgs_iters={budget['lbfgs_iters']} is not evenly divisible by "
            f"effective_chunk_size={effective_chunk_size} -- the last chunk would "
            "overshoot the requested budget. Choose a chunk size that divides evenly."
        )
        print(f"[{mode}/{budget_name}] starting L-BFGS phase "
              f"({budget['lbfgs_iters']} iters total, chunk_size={effective_chunk_size}, "
              f"resuming at n_iter={start_lbfgs_n_iter_done})")
        lbfgs_optimizer = torch.optim.LBFGS(
            net.parameters(), lr=config.LBFGS_LR, max_iter=effective_chunk_size,
            history_size=config.LBFGS_HISTORY_SIZE, line_search_fn=None,
        )
        if lbfgs_optimizer_state_to_restore is not None:
            lbfgs_optimizer.load_state_dict(lbfgs_optimizer_state_to_restore)

        call_stats = {"last_losses": None}

        def closure():
            lbfgs_optimizer.zero_grad()
            step_losses = compute_losses(net, tensors, col_weights, u_weights)
            step_losses["total_objective"].backward()
            call_stats["last_losses"] = step_losses
            return step_losses["total_objective"]

        n_iter_done = start_lbfgs_n_iter_done
        # When resuming mid-L-BFGS, `closure_evals` (restored from the
        # checkpoint) ALREADY includes `start_lbfgs_n_iter_done` worth of
        # L-BFGS closure calls, and the reconstructed optimizer's own
        # restored `state["func_evals"]` ALSO already includes that same
        # count (load_state_dict restores it verbatim) and will keep
        # growing from there. Subtracting it back out here avoids
        # double-counting those already-completed iterations once the two
        # are added together below. For a fresh (non-resumed) start,
        # start_lbfgs_n_iter_done=0, so this is simply closure_evals
        # unchanged.
        closure_evals_before_lbfgs = closure_evals - start_lbfgs_n_iter_done
        lbfgs_nonfinite = False
        converged_early = False
        while n_iter_done < budget["lbfgs_iters"] and not lbfgs_nonfinite:
            lbfgs_optimizer.step(closure)
            state = lbfgs_optimizer.state[lbfgs_optimizer._params[0]]
            # func_evals is the optimizer's OWN cumulative closure-call
            # counter (persisted in state across chunks); with
            # line_search_fn=None this equals n_iter exactly (verified
            # empirically in test_sapinn_lbfgs_chunking.py), but using the
            # optimizer's own count rather than a manually-incremented one
            # is correct even if that ever changes.
            closure_evals = closure_evals_before_lbfgs + state.get("func_evals", 0)
            new_n_iter_done = state.get("n_iter", n_iter_done)
            requested_this_chunk = min(effective_chunk_size, budget["lbfgs_iters"] - n_iter_done)
            iters_this_chunk = new_n_iter_done - n_iter_done
            if iters_this_chunk < requested_this_chunk:
                converged_early = True
            n_iter_done = new_n_iter_done

            losses = call_stats["last_losses"]
            lbfgs_nonfinite = _scalar_nonfinite(losses["total_objective"]) or _any_nonfinite(col_weights, u_weights)
            if lbfgs_nonfinite:
                warnings_log.append(f"L-BFGS n_iter={n_iter_done}: nonfinite value detected")

            eval_metrics = evaluate_relative_l2(net, device)
            row = {
                "phase": "lbfgs", "epoch": n_iter_done, "closure_evals": closure_evals,
                "wall_clock_s": time.time() - start_time,
                **{k: losses[k].item() for k in LOSS_KEYS},
                **eval_metrics,
                **_weight_stats(col_weights, u_weights),
                "nonfinite": lbfgs_nonfinite,
            }
            metrics_rows.append(row)
            print(
                f"  lbfgs n_iter={n_iter_done}/{budget['lbfgs_iters']}: "
                f"total={row['total_objective']:.4e} global_L2={row['global_relative_l2_error']:.4e}"
                + (" [CONVERGED EARLY]" if converged_early else "")
            )

            save_checkpoint("lbfgs", n_iter_done, lbfgs_opt=lbfgs_optimizer)

            if converged_early:
                print(f"[{mode}/{budget_name}] L-BFGS converged early at n_iter={n_iter_done} "
                      f"(chunk advanced by only {iters_this_chunk} of {requested_this_chunk} "
                      "requested iterations) -- stopping rather than issuing further no-op chunks.")
                break
    else:
        if stopped_early:
            print(f"[{mode}/{budget_name}] SKIPPING L-BFGS phase: Adam phase stopped early due to nonfinite values.")
        elif budget["lbfgs_iters"] == 0:
            print(f"[{mode}/{budget_name}] budget has lbfgs_iters=0 -- skipping L-BFGS phase.")

    run_config = {
        "mode": mode, "budget_name": budget_name, "budget": budget, "seed": seed,
        "device": str(device), "dtype": str(config.DTYPE),
        "nv": config.NV, "x_min": config.X_MIN, "x_max": config.X_MAX,
        "t_min": config.T_MIN, "t_max": config.T_MAX,
        "layer_sizes": config.LAYER_SIZES, "activation": config.ACTIVATION,
        "n_f": config.N_F, "n_0": config.N_0, "n_b_per_side": config.N_B_PER_SIDE,
        "col_weight_init": config.COL_WEIGHT_INIT, "u_weight_init_dist": config.U_WEIGHT_INIT_DIST,
        "adam_lr": config.ADAM_LR, "adam_betas": list(config.ADAM_BETAS),
        "lbfgs_history_size": config.LBFGS_HISTORY_SIZE, "lbfgs_lr": config.LBFGS_LR,
        "checkpoint_every_adam": checkpoint_every_adam, "eval_every_adam": eval_every_adam,
        "lbfgs_chunk_size": lbfgs_chunk_size,
        "warnings": warnings_log,
    }
    (output_dir / "config.json").write_text(json.dumps(run_config, indent=2))

    with (output_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        for row in metrics_rows:
            writer.writerow(row)

    _atomic_save(
        {
            "model_state_dict": net.state_dict(),
            "col_weights": col_weights.detach().cpu(),
            "u_weights": u_weights.detach().cpu(),
            "mode": mode, "budget_name": budget_name, "seed": seed,
        },
        output_dir / "checkpoints" / "final.pt",
    )

    _plot_loss_history(metrics_rows, output_dir / "plots" / "loss_history.png")
    _plot_attention_map(tensors, col_weights, mode, output_dir / "plots" / "attention_map.png")
    _plot_u_weights(tensors, u_weights, mode, output_dir / "plots" / "u_weights.png")
    _plot_solution_and_error(net, device, output_dir / "plots" / "solution_error_map.png")

    print(f"[{mode}/{budget_name}] done. Output: {output_dir}")
    return metrics_rows, run_config


def _plot_loss_history(rows, save_path):
    epochs_adam = [r["epoch"] for r in rows if r["phase"] == "adam"]
    total_adam = [r["total_objective"] for r in rows if r["phase"] == "adam"]
    pde_adam = [r["unweighted_pde_mse"] for r in rows if r["phase"] == "adam"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs_adam, total_adam, label="total objective (weighted)")
    ax.plot(epochs_adam, pde_adam, label="unweighted PDE residual MSE")
    ax.set_yscale("log")
    ax.set_xlabel("Adam epoch")
    ax.set_ylabel("loss")
    ax.set_title("Training loss history (Adam phase)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def _plot_attention_map(tensors, col_weights, mode, save_path):
    x_f = tensors["x_f"].detach().cpu().numpy().reshape(-1)
    t_f = tensors["t_f"].detach().cpu().numpy().reshape(-1)
    w = col_weights.detach().cpu().numpy().reshape(-1)

    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(x_f, t_f, c=w, s=6, cmap="viridis")
    fig.colorbar(sc, ax=ax, label="collocation attention weight")
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_title(f"Final spatial-temporal collocation attention weights ({mode})")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def _plot_u_weights(tensors, u_weights, mode, save_path):
    x0 = tensors["x0"].detach().cpu().numpy().reshape(-1)
    w = u_weights.detach().cpu().numpy().reshape(-1)

    fig, ax = plt.subplots(figsize=(8, 5))
    order = np.argsort(x0)
    ax.scatter(x0[order], w[order], s=10)
    ax.set_xlabel("x0 (initial-condition point)")
    ax.set_ylabel("IC attention weight")
    ax.set_title(f"Final IC attention weights ({mode})")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def _plot_solution_and_error(model, device, save_path):
    x_test, t_test = make_test_grid(config.M_TEST, config.N_TEST_TIMES, device)
    x_test = x_test.to(dtype=config.DTYPE)
    t_test = t_test.to(dtype=config.DTYPE)
    y_true = make_reference_values(t_test, x_test, config.NV)
    with torch.no_grad():
        y_pred = model(x_test, t_test).cpu().numpy().reshape(-1)

    n_times = config.N_TEST_TIMES
    n_x = config.M_TEST
    x2 = x_test.detach().cpu().numpy().reshape(n_times, n_x)
    t2 = t_test.detach().cpu().numpy().reshape(n_times, n_x)
    true2 = y_true.reshape(n_times, n_x)
    pred2 = y_pred.reshape(n_times, n_x)
    err2 = np.abs(true2 - pred2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, data, title in zip(axes, (true2, pred2, err2), ("u_true", "u_pred", "|error|")):
        im = ax.pcolormesh(x2, t2, data, shading="auto")
        fig.colorbar(im, ax=ax)
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
