"""Trainer for SA-RF's three controlled comparison modes (Phase 4):
  - baseline:     lambda_f/lambda_0 fixed at 1.0 (effective mask 1, i.e.
                  exactly unweighted MSE -- the original RF loss shape,
                  but WITHOUT burgers_rf's own IC_BC_WEIGHT=1000 penalty;
                  see module docstring in attention.py / the task spec).
  - fixed_weight: lambda_f/lambda_0 initialized identically to adaptive
                  (lambda_f=100, lambda_0~U(0,1)) but frozen throughout.
  - adaptive:     lambda_f/lambda_0 trainable, updated by gradient ASCENT
                  during Adam, frozen during L-BFGS.

A single parameterized entry point (run(mode=...)) is used for all three,
exactly mirroring sapinn/train.py's design, so they are structurally
guaranteed to share RF features, collocation points, model initialization,
optimizer settings, and evaluation grid -- the ONLY difference between
them is the attention-weight path.

Does not modify burgers_rf/ or sapinn/ in any way. The RF model itself
(burgers_rf.model.BurgersRF) is imported and used completely unmodified;
the PDE residual is burgers_rf.physics.burgers_residual, also unmodified.
"""
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

BURGERS_DIR = Path(__file__).resolve().parents[1]
if str(BURGERS_DIR) not in sys.path:
    sys.path.insert(0, str(BURGERS_DIR))
if str(BURGERS_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(BURGERS_DIR / "scripts"))

from burgers_rf.data import make_test_grid, make_reference_values  # noqa: E402
from burgers_rf.evaluation import evaluate_error_metrics  # noqa: E402
from burgers_rf.model import BurgersRF  # noqa: E402
from sweep_local_width import build_shock_grid  # noqa: E402

from . import attention, config, physics, sampling  # noqa: E402

LOSS_KEYS = (
    "unweighted_pde_mse", "weighted_pde_loss",
    "unweighted_ic_mse", "weighted_ic_loss",
    "bc_loss_left", "bc_loss_right", "bc_mse", "total_objective",
)
EVAL_KEYS = ("global_relative_l2_error", "smooth_relative_l2_error", "shock_relative_l2_error")
CSV_HEADER = (
    ["phase", "epoch", "closure_evals", "wall_clock_s"]
    + list(LOSS_KEYS)
    + list(EVAL_KEYS)
    + ["lambda_f_mean", "lambda_f_std", "lambda_f_min", "lambda_f_max",
       "lambda_0_mean", "lambda_0_std",
       "lambda_f_grad_norm", "lambda_0_grad_norm",
       "coef_grad_norm", "coef_norm", "nonfinite"]
)

MODES = ("baseline", "fixed_weight", "adaptive", *config.BALANCED_MODE_VALUES)


def build_model(nx, nt, device):
    """Constructs the UNMODIFIED BurgersRF model. torch.set_default_dtype
    must already be called before this (see run())."""
    model = BurgersRF(nx, nt, config.SIGMA_X, config.SIGMA_T, use_local=False, device=device).to(device)
    return model


def trainable_coefficients(model):
    """The RF model's only trainable parameter: the linear-layer weight
    (the `c_k` coefficients in u_RF = sum_k c_k phi_k). Wx/Wt/bx/bt are
    frozen (requires_grad=False) by BurgersRF's own construction -- never
    touched here."""
    return model.model[0].weight


def compute_losses(model, tensors, lambda_f, lambda_0, nv=None):
    """Physics-only loss computation -- never uses the analytic reference
    solution (that happens only in evaluate_relative_l2, evaluation-only).

    L = mean((lambda_f * r)^2)
      + mean((lambda_0 * (u_RF(x,0) + sin(pi*x)))^2)
      + L_BC_left + L_BC_right

    No IC_BC_WEIGHT=1000 penalty (burgers_rf's own convention) is added --
    the self-adaptive weights supersede it, per the task specification.
    """
    nv = config.NV if nv is None else nv
    r = physics.pde_residual(model, tensors["x_f"], tensors["t_f"], nv)
    ic_err = physics.ic_prediction_error(model, tensors["x0"], tensors["t0"])
    u_lb, u_ub = physics.boundary_predictions(
        model, tensors["x_lb"], tensors["t_lb"], tensors["x_ub"], tensors["t_ub"]
    )

    unweighted_pde_mse = torch.mean(r**2)
    weighted_pde_loss = torch.mean((attention.identity_mask(lambda_f) * r) ** 2)
    unweighted_ic_mse = torch.mean(ic_err**2)
    weighted_ic_loss = torch.mean((attention.identity_mask(lambda_0) * ic_err) ** 2)
    # [SAPINN] BC loss has NO self-adaptive weight, matching sapinn/train.py.
    bc_loss_left = torch.mean(u_lb**2)
    bc_loss_right = torch.mean(u_ub**2)
    bc_mse = bc_loss_left + bc_loss_right

    total_objective = weighted_pde_loss + weighted_ic_loss + bc_mse
    return {
        "unweighted_pde_mse": unweighted_pde_mse,
        "weighted_pde_loss": weighted_pde_loss,
        "unweighted_ic_mse": unweighted_ic_mse,
        "weighted_ic_loss": weighted_ic_loss,
        "bc_loss_left": bc_loss_left,
        "bc_loss_right": bc_loss_right,
        "bc_mse": bc_mse,
        "total_objective": total_objective,
    }


def evaluate_relative_l2(model, device):
    """EVALUATION ONLY. Reuses burgers_rf's existing reference-solution and
    evaluation-metric machinery unmodified."""
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
    return not torch.isfinite(tensor).all().item()


def _any_nonfinite(*tensors):
    return any(not torch.isfinite(t).all().item() for t in tensors)


def _weight_stats(lambda_f, lambda_0, coef, coef_grad_norm):
    lambda_f_grad_norm = lambda_f.grad.detach().norm().item() if lambda_f.grad is not None else float("nan")
    lambda_0_grad_norm = lambda_0.grad.detach().norm().item() if lambda_0.grad is not None else float("nan")
    return {
        "lambda_f_mean": lambda_f.mean().item(), "lambda_f_std": lambda_f.std().item(),
        "lambda_f_min": lambda_f.min().item(), "lambda_f_max": lambda_f.max().item(),
        "lambda_0_mean": lambda_0.mean().item(), "lambda_0_std": lambda_0.std().item(),
        "lambda_f_grad_norm": lambda_f_grad_norm, "lambda_0_grad_norm": lambda_0_grad_norm,
        "coef_grad_norm": coef_grad_norm, "coef_norm": coef.detach().norm().item(),
    }


def _atomic_save(obj, path):
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def run(
    mode, budget_name, seed, output_dir, nx=None, nt=None,
    n_f=None, n_0=None, n_b_per_side=None, device=None, overwrite=False,
    coef_lr=None, lambda_f_init=None, lambda_0_init=None,
):
    """coef_lr overrides the RF-coefficient Adam optimizer's learning rate
    only (default: config.ADAM_LR). The attention ascent optimizers
    (lambda_f_optimizer/lambda_0_optimizer) always use config.ADAM_LR,
    regardless of coef_lr -- they are NOT affected by this override, by
    design (see scripts/run_sa_rf_lr_sweep.py, which sweeps coef_lr only).

    lambda_f_init/lambda_0_init are OPTIONAL scalar overrides for the
    initial attention values, used ONLY by mode="adaptive" (see
    attention.make_adaptive_weights). Defaults (None) reproduce the exact
    original behavior (lambda_f=100, lambda_0~Uniform(0,1)). They do NOT
    affect RF features, RF coefficients, collocation points, or evaluation
    data -- those are all constructed before this override is applied (see
    scripts/run_sa_rf_attention_init_sweep.py, which sweeps these only)."""
    assert mode in MODES, mode
    assert budget_name in config.BUDGETS, budget_name
    device = device or config.get_device()
    nx = config.NX_MATCHED if nx is None else nx
    nt = config.NT_MATCHED if nt is None else nt
    coef_lr = config.ADAM_LR if coef_lr is None else coef_lr
    budget = config.BUDGETS[budget_name]

    output_dir = Path(output_dir)
    done_marker = output_dir / "config.json"
    if done_marker.exists() and not overwrite:
        raise FileExistsError(
            f"{output_dir} already contains a completed run ({done_marker} exists). "
            "Refusing to overwrite. Pass overwrite=True if this is intentional."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Explicit, not solely relying on any import-time side effect --
    # BurgersRF draws Wx/Wt/bx/bt at whatever the default dtype is when
    # constructed, so this MUST happen before build_model() below.
    torch.set_default_dtype(config.DTYPE)
    config.set_all_seeds(seed)

    # Fixed training points for the WHOLE run (see sampling.py docstring)
    # -- built before any mode-dependent branching so all three modes
    # share identical collocation/IC/BC points given the same seed.
    tensors = sampling.build_training_tensors(seed, device, n_f=n_f, n_0=n_0, n_b_per_side=n_b_per_side)
    n_f_actual = tensors["x_f"].shape[0]
    n_0_actual = tensors["x0"].shape[0]

    # Model construction consumes RNG (Wx/Wt/bx/bt draws) BEFORE any
    # mode-dependent attention-weight RNG consumption, so all three modes
    # also share identical initial RF basis AND identical initial
    # coefficients (nn.Linear's default init) given the same seed.
    model = build_model(nx, nt, device)
    coef = trainable_coefficients(model)

    # Seeded fresh, right before the mode branch, unconditionally -- so
    # adaptive and fixed_weight consume IDENTICAL random draws for
    # lambda_0, giving them bit-identical initial attention parameters.
    cpu_generator = torch.Generator(device="cpu").manual_seed(seed)
    if mode == "adaptive":
        lambda_f, lambda_0 = attention.make_adaptive_weights(
            n_f_actual, n_0_actual, device, cpu_generator,
            lambda_f_init=lambda_f_init, lambda_0_init=lambda_0_init,
        )
    elif mode == "fixed_weight":
        lambda_f, lambda_0 = attention.make_fixed_weights(n_f_actual, n_0_actual, device, cpu_generator)
    elif mode in config.BALANCED_MODE_VALUES:
        lambda_f, lambda_0 = attention.make_frozen_constant_weights(
            n_f_actual, n_0_actual, device, config.BALANCED_MODE_VALUES[mode]
        )
    else:
        lambda_f, lambda_0 = attention.make_baseline_weights(n_f_actual, n_0_actual, device)

    coef_optimizer = torch.optim.Adam([coef], lr=coef_lr, betas=config.ADAM_BETAS)
    lambda_f_optimizer = lambda_0_optimizer = None
    if mode == "adaptive":
        # Gradient ASCENT on the attention weights -- maximize=True,
        # matching sapinn/train.py's convention exactly. Always uses
        # config.ADAM_LR (NOT coef_lr) -- the attention optimizer's
        # learning rate is deliberately held fixed across the coef_lr
        # sweep (see scripts/run_sa_rf_lr_sweep.py).
        lambda_f_optimizer = torch.optim.Adam([lambda_f], lr=config.ADAM_LR, betas=config.ADAM_BETAS, maximize=True)
        lambda_0_optimizer = torch.optim.Adam([lambda_0], lr=config.ADAM_LR, betas=config.ADAM_BETAS, maximize=True)

    metrics_rows = []
    warnings_log = []
    start_time = time.time()
    stopped_early = False
    closure_evals = 0

    # ------------------------------------------------------------------
    # True pre-training snapshot (phase="initial", epoch=-1), computed
    # BEFORE any optimizer.step() call -- bit-identical across coef_lr
    # values given the same seed/mode, since nothing optimizer-dependent
    # has touched the model/attention weights yet.
    #
    # BUG FOUND AND FIXED HERE: the per-epoch "adam epoch N" rows below
    # (inherited unchanged from sapinn/train.py's identical convention)
    # log eval_metrics (global/shock relative L2) via evaluate_relative_l2()
    # called AFTER coef_optimizer.step() has already mutated `coef` for
    # that epoch, while the LOSS fields in the same row were computed
    # BEFORE that step. So "epoch 0"'s loss fields reflect the untrained
    # state but its L2 fields reflect the state after ONE gradient step --
    # and since the step SIZE depends on coef_lr, the "epoch 0" L2 varies
    # across coef_lr values even though the true initial weights are
    # identical. This row is the genuine fix: a single, internally
    # consistent, lr-independent pre-training snapshot. The existing
    # "adam epoch N" rows are left UNCHANGED (same formula, same
    # semantics, same numbers as before) to avoid altering any
    # already-saved comparison -- do not treat "epoch 0" as the initial
    # state; use this "initial" row instead.
    # ------------------------------------------------------------------
    initial_losses = compute_losses(model, tensors, lambda_f, lambda_0)
    initial_eval = evaluate_relative_l2(model, device)
    metrics_rows.append({
        "phase": "initial", "epoch": -1, "closure_evals": 0, "wall_clock_s": 0.0,
        **{k: initial_losses[k].item() for k in LOSS_KEYS},
        **initial_eval,
        **_weight_stats(lambda_f, lambda_0, coef, float("nan")),
        "nonfinite": _scalar_nonfinite(initial_losses["total_objective"]),
    })
    print(f"  initial (pre-training): total={initial_losses['total_objective'].item():.4e} "
          f"global_L2={initial_eval['global_relative_l2_error']:.4e}")

    # ------------------------------------------------------------------
    # Adam phase
    # ------------------------------------------------------------------
    print(f"[sa_rf/{mode}/{budget_name}] starting Adam phase ({budget['adam_iters']} iters)")
    losses = None
    for epoch in range(budget["adam_iters"]):
        coef_optimizer.zero_grad()
        if mode == "adaptive":
            lambda_f_optimizer.zero_grad()
            lambda_0_optimizer.zero_grad()

        losses = compute_losses(model, tensors, lambda_f, lambda_0)
        losses["total_objective"].backward()
        closure_evals += 1
        coef_grad_norm = coef.grad.detach().norm().item() if coef.grad is not None else float("nan")

        coef_optimizer.step()
        if mode == "adaptive":
            lambda_f_optimizer.step()
            lambda_0_optimizer.step()

        nonfinite = _scalar_nonfinite(losses["total_objective"])
        if nonfinite:
            warnings_log.append(f"epoch {epoch}: nonfinite total_objective, stopping Adam phase early")

        do_log = epoch % config.LOG_EVERY == 0 or epoch == budget["adam_iters"] - 1 or nonfinite
        if do_log:
            nonfinite = nonfinite or _any_nonfinite(lambda_f, lambda_0, coef)
            do_eval = epoch % config.EVAL_EVERY_ADAM == 0 or epoch == budget["adam_iters"] - 1 or nonfinite
            eval_metrics = evaluate_relative_l2(model, device) if do_eval else dict.fromkeys(EVAL_KEYS, float("nan"))
            row = {
                "phase": "adam", "epoch": epoch, "closure_evals": closure_evals,
                "wall_clock_s": time.time() - start_time,
                **{k: losses[k].item() for k in LOSS_KEYS},
                **eval_metrics,
                **_weight_stats(lambda_f, lambda_0, coef, coef_grad_norm),
                "nonfinite": nonfinite,
            }
            metrics_rows.append(row)
            print(f"  adam epoch {epoch}: total={row['total_objective']:.4e} "
                  f"pde_mse={row['unweighted_pde_mse']:.4e}"
                  + (f" global_L2={row['global_relative_l2_error']:.4e}" if do_eval else ""))

        if nonfinite:
            stopped_early = True
            break

    # ------------------------------------------------------------------
    # L-BFGS phase -- ONE continuous step(closure, max_iter=N) call.
    # Chunking is deliberately NOT used: chunked torch.optim.LBFGS calls
    # were empirically found NOT bit-identical to one continuous call for
    # the pure-RF model (docs/RESEARCH_LOG.md Section 42,
    # scripts/test_rf_lbfgs_chunking_equivalence.py) -- unlike SA-PINN,
    # where chunking was verified safe. Attention weights are NEVER passed
    # to this optimizer, matching "frozen during L-BFGS".
    # ------------------------------------------------------------------
    if not stopped_early and budget["lbfgs_iters"] > 0:
        print(f"[sa_rf/{mode}/{budget_name}] starting L-BFGS phase ({budget['lbfgs_iters']} iters, continuous)")
        lbfgs_optimizer = torch.optim.LBFGS(
            [coef], lr=config.LBFGS_LR, max_iter=budget["lbfgs_iters"],
            history_size=config.LBFGS_HISTORY_SIZE, line_search_fn=None,
        )
        call_stats = {"last_losses": None}

        def closure():
            lbfgs_optimizer.zero_grad()
            step_losses = compute_losses(model, tensors, lambda_f, lambda_0)
            step_losses["total_objective"].backward()
            call_stats["last_losses"] = step_losses
            return step_losses["total_objective"]

        lbfgs_optimizer.step(closure)
        state = lbfgs_optimizer.state[lbfgs_optimizer._params[0]]
        n_iter_done = state.get("n_iter", 0)
        closure_evals += state.get("func_evals", 0)
        losses = call_stats["last_losses"]
        coef_grad_norm = coef.grad.detach().norm().item() if coef.grad is not None else float("nan")
        lbfgs_nonfinite = _scalar_nonfinite(losses["total_objective"]) or _any_nonfinite(lambda_f, lambda_0, coef)
        if lbfgs_nonfinite:
            warnings_log.append(f"L-BFGS n_iter={n_iter_done}: nonfinite value detected")

        eval_metrics = evaluate_relative_l2(model, device)
        row = {
            "phase": "lbfgs", "epoch": n_iter_done, "closure_evals": closure_evals,
            "wall_clock_s": time.time() - start_time,
            **{k: losses[k].item() for k in LOSS_KEYS},
            **eval_metrics,
            **_weight_stats(lambda_f, lambda_0, coef, coef_grad_norm),
            "nonfinite": lbfgs_nonfinite,
        }
        metrics_rows.append(row)
        print(f"  lbfgs n_iter={n_iter_done}/{budget['lbfgs_iters']}: "
              f"total={row['total_objective']:.4e} global_L2={row['global_relative_l2_error']:.4e}")
    else:
        if stopped_early:
            print(f"[sa_rf/{mode}/{budget_name}] SKIPPING L-BFGS phase: Adam phase stopped early.")
        elif budget["lbfgs_iters"] == 0:
            print(f"[sa_rf/{mode}/{budget_name}] budget has lbfgs_iters=0 -- skipping L-BFGS phase.")

    run_config = {
        "mode": mode, "budget_name": budget_name, "budget": budget, "seed": seed,
        "device": str(device), "dtype": str(config.DTYPE),
        "nx": nx, "nt": nt, "n_rf_coefficients": nx * nt,
        "n_f": n_f_actual, "n_0": n_0_actual,
        "n_b_per_side": tensors["x_lb"].shape[0],
        "n_attention_parameters": n_f_actual + n_0_actual,
        "nv": config.NV, "sigma_x": config.SIGMA_X, "sigma_t": config.SIGMA_T,
        # Actual resolved initialization values used for THIS run (not
        # just the global config default) -- records the real override
        # when lambda_f_init/lambda_0_init were passed to run().
        "lambda_f_init": config.LAMBDA_F_INIT if lambda_f_init is None else lambda_f_init,
        "lambda_0_init": config.LAMBDA_0_INIT_DIST if lambda_0_init is None else lambda_0_init,
        "adam_lr": config.ADAM_LR, "coef_lr": coef_lr, "adam_betas": list(config.ADAM_BETAS),
        "lbfgs_history_size": config.LBFGS_HISTORY_SIZE, "lbfgs_lr": config.LBFGS_LR,
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
            "model_state_dict": model.state_dict(),
            "Wx": model.Wx.detach().cpu(), "Wt": model.Wt.detach().cpu(),
            "bx": model.bx.detach().cpu(), "bt": model.bt.detach().cpu(),
            "rf_linear_weight": coef.detach().cpu(),
            "lambda_f": lambda_f.detach().cpu(), "lambda_0": lambda_0.detach().cpu(),
            "mode": mode, "budget_name": budget_name, "seed": seed,
        },
        output_dir / "final.pt",
    )

    print(f"[sa_rf/{mode}/{budget_name}] done. Output: {output_dir}")
    return metrics_rows, run_config
