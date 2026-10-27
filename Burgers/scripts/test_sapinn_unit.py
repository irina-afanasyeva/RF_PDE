import json
import shutil
import sys
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(PROJECT_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from sapinn import config, physics, sampling  # noqa: E402
from sapinn.model import PINN  # noqa: E402
from sapinn.train import run  # noqa: E402

# Lightweight assert-based tests, matching this project's existing
# convention (no pytest is used anywhere in this repository). Run via:
#   python Burgers/scripts/test_sapinn_unit.py


def test_pde_residual_analytic():
    """u(x,t) = x^2 + t has u_t=1, u_x=2x, u_xx=2, so the Burgers residual
    is exactly 1 + (x^2+t)*2x - nv*2 everywhere -- a closed-form check
    independent of any trained network. (A purely linear u(x,t)=x+t was
    tried first but rejected: u_x is then a constant with no graph
    dependency on x, so torch.autograd.grad(u_x, x) legitimately fails
    with "does not require grad" -- the quadratic term keeps u_x genuinely
    x-dependent so the second derivative can actually be taken.)"""
    torch.set_default_dtype(config.DTYPE)

    class QuadraticToy(torch.nn.Module):
        def forward(self, x, t):
            return x**2 + t

    model = QuadraticToy()
    x = torch.tensor([[0.3], [-0.7], [0.0]], dtype=config.DTYPE, requires_grad=True)
    t = torch.tensor([[0.1], [0.9], [0.5]], dtype=config.DTYPE, requires_grad=True)

    nv = config.NV
    residual = physics.pde_residual(model, x, t, nv=nv)
    expected = 1.0 + (x**2 + t) * (2 * x) - nv * 2.0
    assert torch.allclose(residual, expected, atol=1e-6), (residual, expected)
    print("test_pde_residual_analytic: PASS")


def test_initial_condition_matches_g():
    x = torch.tensor([[-1.0], [0.0], [0.5], [1.0]], dtype=config.DTYPE)
    expected = -torch.sin(torch.pi * x)
    actual = physics.initial_condition(x)
    assert torch.allclose(actual, expected), (actual, expected)
    # Spot check against the known closed-form values.
    assert abs(float(physics.initial_condition(torch.tensor([[0.0]]))) - 0.0) < 1e-8
    print("test_initial_condition_matches_g: PASS")


def test_boundary_points_at_domain_edges():
    rng = __import__("numpy").random.default_rng(0)
    x_lb, t_lb, x_ub, t_ub = sampling.sample_bc_points(config.N_B_PER_SIDE, rng)
    assert (x_lb == config.X_MIN).all()
    assert (x_ub == config.X_MAX).all()
    assert (t_lb >= config.T_MIN).all() and (t_lb <= config.T_MAX).all()
    assert (t_ub >= config.T_MIN).all() and (t_ub <= config.T_MAX).all()
    assert x_lb.shape == (config.N_B_PER_SIDE, 1)
    print("test_boundary_points_at_domain_edges: PASS")


def test_collocation_points_cover_domain():
    rng = __import__("numpy").random.default_rng(0)
    x_f, t_f = sampling.sample_collocation_points(config.N_F, rng)
    assert x_f.shape == (config.N_F, 1)
    assert (x_f >= config.X_MIN).all() and (x_f <= config.X_MAX).all()
    assert (t_f >= config.T_MIN).all() and (t_f <= config.T_MAX).all()
    print("test_collocation_points_cover_domain: PASS")


def test_attention_gradient_ascent_direction():
    """Construct a toy model with a KNOWN, large residual at one
    collocation point and near-zero residual at another; verify one Adam
    ascent step increases the weight at the large-residual point more than
    (or at least as much as) at the near-zero-residual point -- i.e. the
    self-adaptive weight moves toward emphasizing the harder point, per
    the paper's stated mechanism ("weights increase as the corresponding
    losses increase")."""
    torch.set_default_dtype(config.DTYPE)

    class ToyModel(torch.nn.Module):
        """u = 0 everywhere -> PDE residual u_t+u*u_x-nv*u_xx = 0 always.
        We instead directly fabricate residual-proportional weight
        gradients by using the IC term, whose residual is controllable via
        a trainable scalar bias per point."""

        def __init__(self):
            super().__init__()
            self.const = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x, t):
            return self.const * torch.ones_like(x)

    model = ToyModel()
    # Two IC points; model always predicts 0, so ic_err = 0 - g(x).
    # Point A: x=0.5 -> g(x) = -sin(pi*0.5) = -1 (large residual).
    # Point B: x=1e-6 -> g(x) ~ 0 (near-zero residual).
    x0 = torch.tensor([[0.5], [1e-6]], dtype=config.DTYPE)
    t0 = torch.zeros_like(x0)

    device = torch.device("cpu")
    generator = torch.Generator(device="cpu").manual_seed(0)
    _, u_weights = __import__("sapinn.attention", fromlist=["make_adaptive_weights"]).make_adaptive_weights(
        n_f=1, n_0=2, device=device, generator=generator
    )
    u_weights_before = u_weights.detach().clone()

    optimizer = torch.optim.Adam([u_weights], lr=0.1, maximize=True)
    optimizer.zero_grad()
    ic_err = physics.ic_prediction_error(model, x0, t0)
    weighted_ic_loss = torch.mean((u_weights * ic_err) ** 2)
    weighted_ic_loss.backward()
    optimizer.step()

    delta_a = (u_weights[0] - u_weights_before[0]).item()
    delta_b = (u_weights[1] - u_weights_before[1]).item()
    assert delta_a > delta_b, (
        f"expected the large-residual point's weight to increase more than the "
        f"near-zero-residual point's weight (ascent should favor larger loss); "
        f"got delta_a={delta_a}, delta_b={delta_b}"
    )
    print("test_attention_gradient_ascent_direction: PASS "
          f"(delta_large_residual={delta_a:.6f}, delta_near_zero_residual={delta_b:.6f})")


def test_end_to_end_smoke():
    """Full, tiny end-to-end run for both modes -- verifies the training
    loop, checkpoint/config/metrics/plot I/O, and finite losses, without
    any long training. Cleans up its own output directory afterward."""
    smoke_root = PROJECT_DIR / "outputs" / "sapinn_burgers" / "_unit_test_tmp"
    if smoke_root.exists():
        shutil.rmtree(smoke_root)

    for mode in ("baseline", "adaptive", "fixed_weight", *config.BALANCED_MODE_VALUES):
        output_dir = smoke_root / mode
        metrics_rows, run_config = run(mode=mode, budget_name="smoke", seed=0, output_dir=output_dir)
        assert len(metrics_rows) > 0
        last_row = metrics_rows[-1]
        for key in ("unweighted_pde_mse", "weighted_pde_loss", "bc_mse", "total_objective",
                    "global_relative_l2_error", "smooth_relative_l2_error", "shock_relative_l2_error"):
            value = last_row[key]
            assert value == value and abs(value) != float("inf"), f"{mode}: non-finite {key}={value}"
        assert not last_row["nonfinite"], f"{mode}: nonfinite flag set"
        assert (output_dir / "config.json").exists()
        assert (output_dir / "metrics.csv").exists()
        assert (output_dir / "checkpoints" / "final.pt").exists()
        assert (output_dir / "plots" / "loss_history.png").exists()
        assert (output_dir / "plots" / "attention_map.png").exists()
        assert (output_dir / "plots" / "solution_error_map.png").exists()
        print(f"test_end_to_end_smoke[{mode}]: PASS "
              f"(final total_objective={last_row['total_objective']:.4e}, "
              f"global_relative_l2_error={last_row['global_relative_l2_error']:.4e})")

    shutil.rmtree(smoke_root)


def test_baseline_and_adaptive_share_initial_state():
    """Verify the structural safeguard: given the same seed, baseline and
    adaptive runs must start from IDENTICAL collocation/IC/BC points and
    IDENTICAL initial network weights -- only the attention-weight path
    may differ."""
    device = torch.device("cpu")
    config.set_all_seeds(123)
    tensors_a = sampling.build_training_tensors(123, device)
    net_a = PINN().to(device=device, dtype=config.DTYPE)

    config.set_all_seeds(123)
    tensors_b = sampling.build_training_tensors(123, device)
    net_b = PINN().to(device=device, dtype=config.DTYPE)

    for key in tensors_a:
        assert torch.equal(tensors_a[key], tensors_b[key]), f"mismatched tensor: {key}"
    for p_a, p_b in zip(net_a.parameters(), net_b.parameters()):
        assert torch.equal(p_a, p_b), "mismatched initial network parameters"
    print("test_baseline_and_adaptive_share_initial_state: PASS")


def test_fixed_weight_and_adaptive_share_initial_attention_weights():
    """fixed_weight must start from IDENTICAL attention-weight values to
    adaptive (same lambda=100 constant, same U(0,1) draw for u_weights
    given the same seed) -- they must differ ONLY in requires_grad /
    trainability, never in their initial numeric values."""
    from sapinn import attention

    device = torch.device("cpu")
    seed = 7

    gen_adaptive = torch.Generator(device="cpu").manual_seed(seed)
    col_adaptive, u_adaptive = attention.make_adaptive_weights(
        config.N_F, config.N_0, device, gen_adaptive
    )

    gen_fixed = torch.Generator(device="cpu").manual_seed(seed)
    col_fixed, u_fixed = attention.make_fixed_weights(
        config.N_F, config.N_0, device, gen_fixed
    )

    assert torch.equal(col_adaptive.detach(), col_fixed.detach()), "mismatched initial col_weights"
    assert torch.equal(u_adaptive.detach(), u_fixed.detach()), "mismatched initial u_weights"
    assert col_adaptive.requires_grad is True and u_adaptive.requires_grad is True
    assert col_fixed.requires_grad is False and u_fixed.requires_grad is False
    print("test_fixed_weight_and_adaptive_share_initial_attention_weights: PASS")


def test_fixed_weight_attention_frozen_through_training():
    """Run a tiny fixed_weight training (Adam + L-BFGS) and verify the
    saved final col_weights/u_weights are BIT-IDENTICAL to what a fresh
    construction with the same seed produces -- i.e. attention parameters
    did not move at all during either phase."""
    from sapinn import attention

    output_dir = PROJECT_DIR / "outputs" / "sapinn_burgers" / "_unit_test_tmp_fixed_weight_frozen"
    if output_dir.exists():
        shutil.rmtree(output_dir)

    seed = 11
    run(mode="fixed_weight", budget_name="smoke", seed=seed, output_dir=output_dir)
    checkpoint = torch.load(output_dir / "checkpoints" / "final.pt", map_location="cpu", weights_only=False)

    device = torch.device("cpu")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    col_expected, u_expected = attention.make_fixed_weights(config.N_F, config.N_0, device, gen)

    assert torch.equal(checkpoint["col_weights"], col_expected.detach()), (
        "fixed_weight col_weights changed during training -- should have been frozen"
    )
    assert torch.equal(checkpoint["u_weights"], u_expected.detach()), (
        "fixed_weight u_weights changed during training -- should have been frozen"
    )
    shutil.rmtree(output_dir)
    print("test_fixed_weight_attention_frozen_through_training: PASS")


def test_balanced_mode_mask_initialization():
    """balanced_10/balanced_100: both col_weights and u_weights must be
    initialized to the SAME constant (10 or 100 respectively) -- not
    config.COL_WEIGHT_INIT, not a random draw -- and must be frozen
    (requires_grad=False) from construction, not merely left unstepped."""
    from sapinn import attention

    device = torch.device("cpu")
    for mode, value in config.BALANCED_MODE_VALUES.items():
        col_weights, u_weights = attention.make_frozen_constant_weights(
            config.N_F, config.N_0, device, value
        )
        assert col_weights.shape == (config.N_F, 1)
        assert u_weights.shape == (config.N_0, 1)
        assert torch.all(col_weights == value), f"{mode}: col_weights not all {value}"
        assert torch.all(u_weights == value), f"{mode}: u_weights not all {value}"
        assert col_weights.requires_grad is False, f"{mode}: col_weights must be frozen"
        assert u_weights.requires_grad is False, f"{mode}: u_weights must be frozen"
    print("test_balanced_mode_mask_initialization: PASS")


def test_balanced_mode_frozen_through_training():
    """Mirrors test_fixed_weight_attention_frozen_through_training for both
    balanced_10 and balanced_100: run a tiny training and verify the saved
    final col_weights/u_weights are BIT-IDENTICAL to the constant values
    they were initialized to -- i.e. truly frozen through Adam and L-BFGS,
    not just closely clustered."""
    from sapinn import attention

    seed = 13
    for mode, value in config.BALANCED_MODE_VALUES.items():
        output_dir = PROJECT_DIR / "outputs" / "sapinn_burgers" / f"_unit_test_tmp_{mode}_frozen"
        shutil.rmtree(output_dir, ignore_errors=True)
        try:
            run(mode=mode, budget_name="smoke", seed=seed, output_dir=output_dir)
            checkpoint = torch.load(output_dir / "checkpoints" / "final.pt", map_location="cpu", weights_only=False)
            col_expected, u_expected = attention.make_frozen_constant_weights(
                config.N_F, config.N_0, torch.device("cpu"), value
            )
            assert torch.equal(checkpoint["col_weights"], col_expected), (
                f"{mode}: col_weights changed during training -- should have been frozen"
            )
            assert torch.equal(checkpoint["u_weights"], u_expected), (
                f"{mode}: u_weights changed during training -- should have been frozen"
            )
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)
    print("test_balanced_mode_frozen_through_training: PASS")


def test_balanced_mode_squared_loss_scaling():
    """Directly verify the squared-mask loss convention for balanced_10/100:
    weighted_pde_loss == value**2 * unweighted_pde_mse and
    weighted_ic_loss == value**2 * unweighted_ic_mse exactly (same
    mean((mask*residual)**2) = mean(mask**2 * residual**2) identity used by
    every other mode -- see sapinn/train.py's compute_losses)."""
    from sapinn import attention
    from sapinn.train import compute_losses

    device = torch.device("cpu")
    config.set_all_seeds(0)
    tensors = sampling.build_training_tensors(0, device)
    net = PINN().to(device=device, dtype=config.DTYPE)

    for mode, value in config.BALANCED_MODE_VALUES.items():
        col_weights, u_weights = attention.make_frozen_constant_weights(
            config.N_F, config.N_0, device, value
        )
        losses = compute_losses(net, tensors, col_weights, u_weights)
        expected_pde = (value ** 2) * losses["unweighted_pde_mse"].item()
        expected_ic = (value ** 2) * losses["unweighted_ic_mse"].item()
        assert abs(losses["weighted_pde_loss"].item() - expected_pde) < 1e-6 * max(1.0, abs(expected_pde)), (
            f"{mode}: weighted_pde_loss={losses['weighted_pde_loss'].item()} != "
            f"value**2 * unweighted_pde_mse={expected_pde}"
        )
        assert abs(losses["weighted_ic_loss"].item() - expected_ic) < 1e-6 * max(1.0, abs(expected_ic)), (
            f"{mode}: weighted_ic_loss={losses['weighted_ic_loss'].item()} != "
            f"value**2 * unweighted_ic_mse={expected_ic}"
        )
    print("test_balanced_mode_squared_loss_scaling: PASS")


def test_balanced_mode_checkpoint_resume():
    """Checkpoint/resume compatibility for a balanced mode, mirroring
    test_adam_resume_consistency: 10 Adam epochs uninterrupted vs. 5+resume+5
    (same budget name, temporarily shortened for the first half) must give
    BIT-IDENTICAL final network/attention state."""
    mode = "balanced_10"
    dir_full = PROJECT_DIR / "outputs" / "sapinn_burgers" / "_unit_test_tmp_balanced_resume_full"
    dir_half = PROJECT_DIR / "outputs" / "sapinn_burgers" / "_unit_test_tmp_balanced_resume_half"
    dir_resumed = PROJECT_DIR / "outputs" / "sapinn_burgers" / "_unit_test_tmp_balanced_resume_resumed"
    for d in (dir_full, dir_half, dir_resumed):
        shutil.rmtree(d, ignore_errors=True)

    config.BUDGETS["_test_balanced_resume"] = {"adam_iters": 10, "lbfgs_iters": 0}
    try:
        rows_full, _ = run(mode=mode, budget_name="_test_balanced_resume", seed=0, output_dir=dir_full)
        ckpt_full = torch.load(dir_full / "checkpoints" / "final.pt", map_location="cpu", weights_only=False)

        config.BUDGETS["_test_balanced_resume"] = {"adam_iters": 5, "lbfgs_iters": 0}
        run(mode=mode, budget_name="_test_balanced_resume", seed=0, output_dir=dir_half)
        halfway_ckpt = dir_half / "checkpoints" / "latest.pt"

        config.BUDGETS["_test_balanced_resume"] = {"adam_iters": 10, "lbfgs_iters": 0}
        rows_resumed, _ = run(
            mode=mode, budget_name="_test_balanced_resume", seed=0, output_dir=dir_resumed,
            resume_from=halfway_ckpt,
        )
        ckpt_resumed = torch.load(dir_resumed / "checkpoints" / "final.pt", map_location="cpu", weights_only=False)

        for key in ckpt_full["model_state_dict"]:
            assert torch.equal(ckpt_full["model_state_dict"][key], ckpt_resumed["model_state_dict"][key]), (
                f"{mode}: resumed run's {key} differs from the uninterrupted run"
            )
        assert torch.equal(ckpt_full["col_weights"], ckpt_resumed["col_weights"])
        assert torch.equal(ckpt_full["u_weights"], ckpt_resumed["u_weights"])
        assert rows_full[-1]["total_objective"] == rows_resumed[-1]["total_objective"], (
            f"{mode}: final loss differs between uninterrupted and resumed runs"
        )
        print(f"test_balanced_mode_checkpoint_resume: PASS (bit-identical final state, "
              f"total_objective={rows_full[-1]['total_objective']:.6e})")
    finally:
        config.BUDGETS.pop("_test_balanced_resume", None)
        for d in (dir_full, dir_half, dir_resumed):
            shutil.rmtree(d, ignore_errors=True)


def test_checkpoint_save_load_roundtrip():
    """Run a tiny training, load the saved checkpoint, and confirm it
    contains everything Priority 2 requires: network state_dict, attention
    weights, all relevant optimizer state_dicts, phase/iteration count, and
    RNG states."""
    output_dir = PROJECT_DIR / "outputs" / "sapinn_burgers" / "_unit_test_tmp_ckpt_roundtrip"
    if output_dir.exists():
        shutil.rmtree(output_dir)

    config.BUDGETS["_test_ckpt"] = {"adam_iters": 6, "lbfgs_iters": 0}
    try:
        run(mode="adaptive", budget_name="_test_ckpt", seed=0, output_dir=output_dir,
            checkpoint_every_adam=3)
        ckpt_path = output_dir / "checkpoints" / "adam_5.pt"
        assert ckpt_path.exists(), "expected a checkpoint at the final epoch (5)"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        for key in (
            "model_state_dict", "col_weights", "u_weights",
            "net_optimizer_state_dict", "col_optimizer_state_dict", "u_optimizer_state_dict",
            "torch_rng_state", "numpy_rng_state", "phase", "epoch", "mode", "budget_name", "seed",
        ):
            assert key in ckpt, f"checkpoint missing required key: {key}"
        assert ckpt["phase"] == "adam" and ckpt["epoch"] == 5
        latest = torch.load(output_dir / "checkpoints" / "latest.pt", map_location="cpu", weights_only=False)
        assert latest["epoch"] == ckpt["epoch"]
    finally:
        del config.BUDGETS["_test_ckpt"]
        shutil.rmtree(output_dir, ignore_errors=True)
    print("test_checkpoint_save_load_roundtrip: PASS")


def test_adam_resume_consistency():
    """Train 10 Adam epochs uninterrupted vs. 5 epochs + checkpoint +
    resume for 5 more epochs (same budget name, temporarily shortened for
    the first half to simulate an interruption at epoch 5) -- the resumed
    run's final network/attention state must be BIT-IDENTICAL to the
    uninterrupted run's, since nothing in this trainer's Adam loop consumes
    per-epoch randomness (collocation/IC/BC points are fixed once, not
    resampled) -- optimizer state (including Adam's step-dependent bias
    correction) is the only thing that must carry over correctly, and it is
    saved/restored via net_optimizer.state_dict()."""
    dir_full = PROJECT_DIR / "outputs" / "sapinn_burgers" / "_unit_test_tmp_resume_full"
    dir_half = PROJECT_DIR / "outputs" / "sapinn_burgers" / "_unit_test_tmp_resume_half"
    dir_resumed = PROJECT_DIR / "outputs" / "sapinn_burgers" / "_unit_test_tmp_resume_resumed"
    for d in (dir_full, dir_half, dir_resumed):
        shutil.rmtree(d, ignore_errors=True)

    config.BUDGETS["_test_resume"] = {"adam_iters": 10, "lbfgs_iters": 0}
    try:
        rows_full, _ = run(mode="adaptive", budget_name="_test_resume", seed=0, output_dir=dir_full)
        ckpt_full = torch.load(dir_full / "checkpoints" / "final.pt", map_location="cpu", weights_only=False)

        # Simulate "interrupted after 5 epochs": temporarily shorten the
        # SAME budget name, run it to completion (5 epochs), then restore
        # the budget to its full 10 epochs before resuming -- the resumed
        # run looks up config.BUDGETS[budget_name] fresh, and the
        # checkpoint's own recorded budget_name/mode/seed are what resume
        # uses, so this reproduces a genuine same-budget interruption.
        config.BUDGETS["_test_resume"] = {"adam_iters": 5, "lbfgs_iters": 0}
        run(mode="adaptive", budget_name="_test_resume", seed=0, output_dir=dir_half)
        halfway_ckpt = dir_half / "checkpoints" / "latest.pt"  # full resumable checkpoint, not the minimal final.pt

        config.BUDGETS["_test_resume"] = {"adam_iters": 10, "lbfgs_iters": 0}
        rows_resumed, _ = run(
            mode="adaptive", budget_name="_test_resume", seed=0, output_dir=dir_resumed,
            resume_from=halfway_ckpt,
        )
        ckpt_resumed = torch.load(dir_resumed / "checkpoints" / "final.pt", map_location="cpu", weights_only=False)

        for key in ckpt_full["model_state_dict"]:
            assert torch.equal(ckpt_full["model_state_dict"][key], ckpt_resumed["model_state_dict"][key]), (
                f"resumed run's {key} differs from the uninterrupted run"
            )
        assert torch.equal(ckpt_full["col_weights"], ckpt_resumed["col_weights"])
        assert torch.equal(ckpt_full["u_weights"], ckpt_resumed["u_weights"])
        assert rows_full[-1]["total_objective"] == rows_resumed[-1]["total_objective"], (
            f"final loss differs: uninterrupted={rows_full[-1]['total_objective']} "
            f"resumed={rows_resumed[-1]['total_objective']}"
        )
        print("test_adam_resume_consistency: PASS (bit-identical final state, "
              f"total_objective={rows_full[-1]['total_objective']:.6e})")
    finally:
        config.BUDGETS.pop("_test_resume", None)
        for d in (dir_full, dir_half, dir_resumed):
            shutil.rmtree(d, ignore_errors=True)


def _values_equal(a, b):
    if torch.is_tensor(a) or torch.is_tensor(b):
        return torch.equal(torch.as_tensor(a), torch.as_tensor(b))
    return a == b


def test_lbfgs_resume_consistency():
    """FINAL VALIDATION before GPU deployment: train L-BFGS for 40 chunked
    iterations (chunk_size=10, 4 chunks) uninterrupted (case A) vs.
    interrupted after 2 completed chunks (20 iterations) and resumed from
    the saved checkpoint for the remaining 2 chunks (case B). Both start
    from IDENTICAL initial network weights, collocation/IC/BC points, and
    attention weights (same seed=0, same short adam_iters=5 warmup so both
    cases enter L-BFGS from the identical post-Adam state). Requires EXACT
    equality throughout, since this problem has a deterministic closure and
    line_search_fn=None (no stochasticity anywhere in this trainer)."""
    dir_full = PROJECT_DIR / "outputs" / "sapinn_burgers" / "_unit_test_tmp_lbfgs_resume_full"
    dir_half = PROJECT_DIR / "outputs" / "sapinn_burgers" / "_unit_test_tmp_lbfgs_resume_half"
    dir_resumed = PROJECT_DIR / "outputs" / "sapinn_burgers" / "_unit_test_tmp_lbfgs_resume_resumed"
    for d in (dir_full, dir_half, dir_resumed):
        shutil.rmtree(d, ignore_errors=True)

    config.BUDGETS["_test_lbfgs_resume"] = {"adam_iters": 5, "lbfgs_iters": 40}
    try:
        # Case A: one uninterrupted run (Adam warmup + all 4 L-BFGS chunks
        # within a single run() call, same optimizer instance throughout --
        # the already-verified same-instance chunking regime).
        rows_full, _ = run(
            mode="adaptive", budget_name="_test_lbfgs_resume", seed=0,
            output_dir=dir_full, lbfgs_chunk_size=10,
        )
        ckpt_full = torch.load(dir_full / "checkpoints" / "latest.pt", map_location="cpu", weights_only=False)
        assert ckpt_full["phase"] == "lbfgs" and ckpt_full["lbfgs_n_iter"] == 40

        # Case B, part 1: simulate "interrupted after 2 chunks" by
        # temporarily shortening the SAME budget name to 20 L-BFGS
        # iterations and running it to completion.
        config.BUDGETS["_test_lbfgs_resume"] = {"adam_iters": 5, "lbfgs_iters": 20}
        run(mode="adaptive", budget_name="_test_lbfgs_resume", seed=0,
            output_dir=dir_half, lbfgs_chunk_size=10)
        halfway_ckpt = dir_half / "checkpoints" / "latest.pt"
        ckpt_half = torch.load(halfway_ckpt, map_location="cpu", weights_only=False)
        assert ckpt_half["phase"] == "lbfgs" and ckpt_half["lbfgs_n_iter"] == 20

        # Case B, part 2: restore the full budget and resume -- this
        # constructs a FRESH torch.optim.LBFGS instance and restores it via
        # load_state_dict() from the checkpoint, the one path NOT covered by
        # the earlier same-instance chunking experiment.
        config.BUDGETS["_test_lbfgs_resume"] = {"adam_iters": 5, "lbfgs_iters": 40}
        rows_resumed, _ = run(
            mode="adaptive", budget_name="_test_lbfgs_resume", seed=0,
            output_dir=dir_resumed, resume_from=halfway_ckpt, lbfgs_chunk_size=10,
        )
        ckpt_resumed = torch.load(dir_resumed / "checkpoints" / "latest.pt", map_location="cpu", weights_only=False)
        assert ckpt_resumed["phase"] == "lbfgs" and ckpt_resumed["lbfgs_n_iter"] == 40

        # --- Final network parameters ---
        for key in ckpt_full["model_state_dict"]:
            assert torch.equal(ckpt_full["model_state_dict"][key], ckpt_resumed["model_state_dict"][key]), (
                f"resumed network parameter {key} differs from the uninterrupted run"
            )

        # --- Frozen attention weights ---
        assert torch.equal(ckpt_full["col_weights"], ckpt_resumed["col_weights"]), "col_weights differ"
        assert torch.equal(ckpt_full["u_weights"], ckpt_resumed["u_weights"]), "u_weights differ"

        # --- Final objective and global relative L2 ---
        assert rows_full[-1]["total_objective"] == rows_resumed[-1]["total_objective"], (
            f"total_objective differs: {rows_full[-1]['total_objective']} vs {rows_resumed[-1]['total_objective']}"
        )
        assert rows_full[-1]["global_relative_l2_error"] == rows_resumed[-1]["global_relative_l2_error"], (
            f"global_relative_l2_error differs: {rows_full[-1]['global_relative_l2_error']} "
            f"vs {rows_resumed[-1]['global_relative_l2_error']}"
        )

        # --- L-BFGS optimizer state, including curvature history ---
        state_full = ckpt_full["lbfgs_optimizer_state_dict"]["state"][0]
        state_resumed = ckpt_resumed["lbfgs_optimizer_state_dict"]["state"][0]
        assert state_full["n_iter"] == state_resumed["n_iter"] == 40, "cumulative n_iter mismatch"
        assert state_full["func_evals"] == state_resumed["func_evals"], "cumulative func_evals mismatch"
        assert len(state_full["old_dirs"]) == len(state_resumed["old_dirs"]), "curvature history length mismatch"
        for a, b in zip(state_full["old_dirs"], state_resumed["old_dirs"]):
            assert torch.equal(a, b), "old_dirs (curvature history) differ"
        for a, b in zip(state_full["old_stps"], state_resumed["old_stps"]):
            assert torch.equal(a, b), "old_stps (curvature history) differ"
        for a, b in zip(state_full["ro"], state_resumed["ro"]):
            assert _values_equal(a, b), "ro (curvature history) differ"
        assert _values_equal(state_full["d"], state_resumed["d"]), "search direction d differs"
        assert _values_equal(state_full["H_diag"], state_resumed["H_diag"]), "H_diag differs"
        assert _values_equal(state_full["prev_flat_grad"], state_resumed["prev_flat_grad"]), "prev_flat_grad differs"
        assert _values_equal(state_full["prev_loss"], state_resumed["prev_loss"]), "prev_loss differs"

        # --- Cumulative iterations and closure evaluations ---
        assert ckpt_full["closure_evals"] == ckpt_resumed["closure_evals"], "closure_evals differ"

        print("test_lbfgs_resume_consistency: PASS (exact equality: network params, "
              "attention weights, total_objective, global_relative_l2_error, full L-BFGS "
              f"curvature history, n_iter={state_full['n_iter']}, "
              f"closure_evals={ckpt_full['closure_evals']})")
    finally:
        config.BUDGETS.pop("_test_lbfgs_resume", None)
        for d in (dir_full, dir_half, dir_resumed):
            shutil.rmtree(d, ignore_errors=True)


def test_seed_specific_output_directories():
    """Two runs differing only in seed must land in different directories
    and both must survive (neither overwrites the other)."""
    base = PROJECT_DIR / "outputs" / "sapinn_burgers" / "baseline" / "smoke"
    dir_seed0 = base / "seed0"
    dir_seed1 = base / "seed1"
    for d in (dir_seed0, dir_seed1):
        shutil.rmtree(d, ignore_errors=True)

    try:
        run(mode="baseline", budget_name="smoke", seed=0, output_dir=dir_seed0)
        run(mode="baseline", budget_name="smoke", seed=1, output_dir=dir_seed1)
        assert dir_seed0.exists() and dir_seed1.exists()
        assert (dir_seed0 / "config.json").exists() and (dir_seed1 / "config.json").exists()
        ckpt0 = torch.load(dir_seed0 / "checkpoints" / "final.pt", map_location="cpu", weights_only=False)
        ckpt1 = torch.load(dir_seed1 / "checkpoints" / "final.pt", map_location="cpu", weights_only=False)
        # Different seeds -> different collocation points -> different trained weights.
        assert not torch.equal(ckpt0["model_state_dict"]["linears.0.weight"], ckpt1["model_state_dict"]["linears.0.weight"])
        print("test_seed_specific_output_directories: PASS")
    finally:
        for d in (dir_seed0, dir_seed1):
            shutil.rmtree(d, ignore_errors=True)


def test_overwrite_protection():
    """A completed run directory must refuse a second run unless
    overwrite=True is passed explicitly."""
    output_dir = PROJECT_DIR / "outputs" / "sapinn_burgers" / "_unit_test_tmp_overwrite_guard"
    shutil.rmtree(output_dir, ignore_errors=True)
    try:
        run(mode="baseline", budget_name="smoke", seed=0, output_dir=output_dir)
        try:
            run(mode="baseline", budget_name="smoke", seed=0, output_dir=output_dir)
            raise AssertionError("expected FileExistsError on re-run without overwrite=True")
        except FileExistsError:
            pass
        # overwrite=True must succeed.
        run(mode="baseline", budget_name="smoke", seed=0, output_dir=output_dir, overwrite=True)
        print("test_overwrite_protection: PASS")
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_checkpoint_optimizer_state_dict_structure():
    """CPU-executable structural check, added after a CUDA-only regression
    test bug (KeyError: 'param_groups') slipped past local testing: every
    saved *_optimizer_state_dict in a checkpoint must be the COMPLETE
    torch optimizer state_dict -- i.e. a dict with BOTH top-level keys
    "state" and "param_groups" -- never just the nested per-parameter
    "state" sub-dict on its own. torch.optim.Optimizer.load_state_dict()
    requires both keys and raises KeyError('param_groups') if only "state"
    is passed. This does not require CUDA and would have caught that bug
    before it ever reached the GPU server."""
    output_dir = PROJECT_DIR / "outputs" / "sapinn_burgers" / "_unit_test_tmp_opt_structure"
    shutil.rmtree(output_dir, ignore_errors=True)
    config.BUDGETS["_test_opt_structure"] = {"adam_iters": 3, "lbfgs_iters": 0}
    try:
        run(mode="adaptive", budget_name="_test_opt_structure", seed=0, output_dir=output_dir)
        # final.pt is intentionally minimal (no optimizer state); the full,
        # resumable checkpoint is latest.pt.
        latest = torch.load(output_dir / "checkpoints" / "latest.pt", map_location="cpu", weights_only=False)

        for key in ("net_optimizer_state_dict", "col_optimizer_state_dict", "u_optimizer_state_dict"):
            sd = latest[key]
            assert sd is not None, f"{key} unexpectedly None for mode=adaptive"
            assert isinstance(sd, dict) and "state" in sd and "param_groups" in sd, (
                f"{key} is missing required top-level keys (found: "
                f"{sorted(sd.keys()) if isinstance(sd, dict) else type(sd)}) -- this must be the "
                "COMPLETE optimizer state_dict, not just its 'state' sub-dict"
            )
            assert len(sd["state"]) > 0, f"{key}['state'] is empty"

        # A freshly constructed optimizer must accept it without error --
        # this is the exact call that previously raised KeyError('param_groups').
        net = PINN().to(device="cpu", dtype=config.DTYPE)
        reloaded = torch.optim.Adam(net.parameters(), lr=config.ADAM_LR, betas=config.ADAM_BETAS)
        reloaded.load_state_dict(latest["net_optimizer_state_dict"])
        print("test_checkpoint_optimizer_state_dict_structure: PASS")
    finally:
        config.BUDGETS.pop("_test_opt_structure", None)
        shutil.rmtree(output_dir, ignore_errors=True)


def test_cuda_checkpoint_resume_consistency():
    """Regression test for the GPU-server bug: torch.load(resume_from,
    map_location=device) with device="cuda:0" relocated torch_rng_state
    (must stay a CPU ByteTensor) and cuda_rng_state onto CUDA, breaking
    torch.set_rng_state() with "TypeError: RNG state must be a
    torch.ByteTensor". Skips gracefully if no CUDA device is available
    (e.g. this Mac) -- only meaningful on the GPU server.

    Mirrors test_adam_resume_consistency's CPU logic but on
    device=torch.device("cuda:0"), and additionally verifies:
      - the run actually used CUDA (model parameters live on a cuda device)
      - CPU RNG state (torch.get_rng_state()) matches after both runs
      - CUDA RNG state (torch.cuda.get_rng_state_all()) matches after both runs
      - optimizer state tensors (Adam exp_avg/exp_avg_sq) ended up on CUDA,
        not silently left on CPU
    """
    if not torch.cuda.is_available():
        print("test_cuda_checkpoint_resume_consistency: SKIPPED (no CUDA device available)")
        return

    device = torch.device("cuda:0")
    dir_full = PROJECT_DIR / "outputs" / "sapinn_burgers" / "_unit_test_tmp_cuda_resume_full"
    dir_half = PROJECT_DIR / "outputs" / "sapinn_burgers" / "_unit_test_tmp_cuda_resume_half"
    dir_resumed = PROJECT_DIR / "outputs" / "sapinn_burgers" / "_unit_test_tmp_cuda_resume_resumed"
    for d in (dir_full, dir_half, dir_resumed):
        shutil.rmtree(d, ignore_errors=True)

    config.BUDGETS["_test_cuda_resume"] = {"adam_iters": 10, "lbfgs_iters": 0}
    try:
        rows_full, _ = run(mode="adaptive", budget_name="_test_cuda_resume", seed=0,
                            output_dir=dir_full, device=device)
        ckpt_full = torch.load(dir_full / "checkpoints" / "final.pt", map_location="cpu", weights_only=False)
        rng_cpu_full = torch.get_rng_state()
        rng_cuda_full = torch.cuda.get_rng_state_all()

        config.BUDGETS["_test_cuda_resume"] = {"adam_iters": 5, "lbfgs_iters": 0}
        run(mode="adaptive", budget_name="_test_cuda_resume", seed=0, output_dir=dir_half, device=device)
        halfway_ckpt = dir_half / "checkpoints" / "latest.pt"

        # This is the exact line that failed on the GPU server before the
        # fix (map_location=device inside run()'s resume path).
        config.BUDGETS["_test_cuda_resume"] = {"adam_iters": 10, "lbfgs_iters": 0}
        rows_resumed, _ = run(mode="adaptive", budget_name="_test_cuda_resume", seed=0,
                               output_dir=dir_resumed, device=device, resume_from=halfway_ckpt)
        ckpt_resumed = torch.load(dir_resumed / "checkpoints" / "final.pt", map_location="cpu", weights_only=False)
        rng_cpu_resumed = torch.get_rng_state()
        rng_cuda_resumed = torch.cuda.get_rng_state_all()

        # --- Confirm CUDA was actually used ---
        # Check the run's OWN recorded device, not just that we can
        # construct something on CUDA ourselves after the fact (a tensor we
        # place on CUDA here would trivially be "on CUDA" regardless of
        # what device training actually used -- that would not be a
        # meaningful check).
        run_config_full = json.loads((dir_full / "config.json").read_text())
        assert run_config_full["device"] == str(device), (
            f"run_config recorded device={run_config_full['device']!r}, expected {str(device)!r} "
            "-- the uninterrupted run did not actually train on CUDA"
        )

        latest_full = torch.load(dir_full / "checkpoints" / "latest.pt", map_location="cpu", weights_only=False)
        net_check = PINN().to(device=device, dtype=config.DTYPE)
        net_check.load_state_dict({k: v.to(device) for k, v in latest_full["model_state_dict"].items()})
        assert all(p.is_cuda for p in net_check.parameters()), "model parameters were not on CUDA after loading"

        # IMPORTANT: load the COMPLETE optimizer state_dict
        # ({"state": ..., "param_groups": ...}), not just its nested
        # "state" sub-dict -- torch.optim.Optimizer.load_state_dict()
        # requires both top-level keys and raises KeyError('param_groups')
        # otherwise. This was the exact bug this test previously had.
        full_optimizer_state_dict = latest_full["net_optimizer_state_dict"]
        assert len(full_optimizer_state_dict["state"]) > 0, "optimizer has no per-parameter state to check"
        reloaded_net_optimizer = torch.optim.Adam(net_check.parameters(), lr=config.ADAM_LR, betas=config.ADAM_BETAS)
        reloaded_net_optimizer.load_state_dict(full_optimizer_state_dict)
        for param_state in reloaded_net_optimizer.state.values():
            for key, v in param_state.items():
                if key == "step":
                    # PyTorch deliberately keeps Adam's "step" counter on
                    # CPU when capturable/fused are both off (the default
                    # here) to avoid costly CUDA kernel launches -- this is
                    # correct, documented behavior, NOT a sign that state
                    # wasn't moved to CUDA. Checking it with the same
                    # is_cuda assertion as exp_avg/exp_avg_sq would be a
                    # false-positive test failure, not a real defect.
                    continue
                if torch.is_tensor(v):
                    assert v.is_cuda, f"optimizer state tensor '{key}' was not moved to CUDA on load"

        # --- Final model parameters: exact equality ---
        for key in ckpt_full["model_state_dict"]:
            assert torch.equal(ckpt_full["model_state_dict"][key], ckpt_resumed["model_state_dict"][key]), (
                f"resumed CUDA run's {key} differs from the uninterrupted CUDA run"
            )
        assert torch.equal(ckpt_full["col_weights"], ckpt_resumed["col_weights"]), "col_weights differ"
        assert torch.equal(ckpt_full["u_weights"], ckpt_resumed["u_weights"]), "u_weights differ"

        # --- Final objective ---
        assert rows_full[-1]["total_objective"] == rows_resumed[-1]["total_objective"], (
            f"total_objective differs: {rows_full[-1]['total_objective']} vs {rows_resumed[-1]['total_objective']}"
        )

        # --- CPU and CUDA RNG states restored/advanced consistently ---
        assert torch.equal(rng_cpu_full, rng_cpu_resumed), "final CPU RNG state differs between the two runs"
        assert len(rng_cuda_full) == len(rng_cuda_resumed)
        for a, b in zip(rng_cuda_full, rng_cuda_resumed):
            assert torch.equal(a, b), "final CUDA RNG state differs between the two runs"

        print("test_cuda_checkpoint_resume_consistency: PASS (CUDA confirmed in use; "
              f"total_objective={rows_full[-1]['total_objective']:.6e}; "
              "CPU and CUDA RNG states match between uninterrupted and resumed runs)")
    finally:
        config.BUDGETS.pop("_test_cuda_resume", None)
        for d in (dir_full, dir_half, dir_resumed):
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    test_pde_residual_analytic()
    test_initial_condition_matches_g()
    test_boundary_points_at_domain_edges()
    test_collocation_points_cover_domain()
    test_attention_gradient_ascent_direction()
    test_baseline_and_adaptive_share_initial_state()
    test_fixed_weight_and_adaptive_share_initial_attention_weights()
    test_fixed_weight_attention_frozen_through_training()
    test_balanced_mode_mask_initialization()
    test_balanced_mode_frozen_through_training()
    test_balanced_mode_squared_loss_scaling()
    test_balanced_mode_checkpoint_resume()
    test_checkpoint_save_load_roundtrip()
    test_adam_resume_consistency()
    test_lbfgs_resume_consistency()
    test_checkpoint_optimizer_state_dict_structure()
    test_cuda_checkpoint_resume_consistency()
    test_seed_specific_output_directories()
    test_overwrite_protection()
    test_end_to_end_smoke()
    print("ALL SA-PINN UNIT TESTS PASSED")
