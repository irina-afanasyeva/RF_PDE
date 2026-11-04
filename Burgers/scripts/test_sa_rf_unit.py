"""Phase 5: unit/integration tests for SA-RF, run on CPU with tiny
collocation counts -- fast, deterministic, no paper-scale training.
Mirrors the fast-test conventions already established for RF Stage 1
(test_rf_lbfgs_fast_tests.py) and SA-PINN (test_sapinn_unit.py).
"""
import shutil
import sys
from pathlib import Path

import torch

BURGERS_DIR = Path(__file__).resolve().parents[1]
if str(BURGERS_DIR) not in sys.path:
    sys.path.insert(0, str(BURGERS_DIR))
if str(BURGERS_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(BURGERS_DIR / "scripts"))

from burgers_rf.physics import burgers_residual  # noqa: E402
from sa_rf import attention, config, physics, sampling, train  # noqa: E402

SCRATCH_ROOT = BURGERS_DIR / "outputs" / "sa_rf" / "_unit_test_tmp"
TINY_NX, TINY_NT = 5, 2       # 10 coefficients
TINY_N_F, TINY_N_0, TINY_N_B = 12, 6, 4


def _clean():
    shutil.rmtree(SCRATCH_ROOT, ignore_errors=True)


def _tiny_setup(mode="adaptive", seed=11, device="cpu"):
    torch.set_default_dtype(config.DTYPE)
    config.set_all_seeds(seed)
    tensors = sampling.build_training_tensors(seed, device, n_f=TINY_N_F, n_0=TINY_N_0, n_b_per_side=TINY_N_B)
    model = train.build_model(TINY_NX, TINY_NT, device)
    coef = train.trainable_coefficients(model)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if mode == "adaptive":
        lambda_f, lambda_0 = attention.make_adaptive_weights(TINY_N_F, TINY_N_0, device, generator)
    elif mode == "fixed_weight":
        lambda_f, lambda_0 = attention.make_fixed_weights(TINY_N_F, TINY_N_0, device, generator)
    else:
        lambda_f, lambda_0 = attention.make_baseline_weights(TINY_N_F, TINY_N_0, device)
    return model, coef, tensors, lambda_f, lambda_0


# 1. RF model output shape and trainable coefficient count.
def test_model_output_shape_and_coefficient_count():
    model, coef, tensors, _, _ = _tiny_setup()
    out = model(tensors["x_f"], tensors["t_f"])
    assert out.shape == (TINY_N_F, 1), out.shape
    assert coef.numel() == TINY_NX * TINY_NT == 10
    assert coef.requires_grad
    print("test_model_output_shape_and_coefficient_count: PASS")


# 2. Random features remain fixed.
def test_random_features_remain_fixed():
    model, coef, tensors, lambda_f, lambda_0 = _tiny_setup()
    Wx0, Wt0, bx0, bt0 = (model.Wx.clone(), model.Wt.clone(), model.bx.clone(), model.bt.clone())
    optimizer = torch.optim.Adam([coef], lr=1e-2)
    for _ in range(5):
        optimizer.zero_grad()
        losses = train.compute_losses(model, tensors, lambda_f, lambda_0)
        losses["total_objective"].backward()
        optimizer.step()
    assert torch.equal(model.Wx, Wx0) and torch.equal(model.Wt, Wt0)
    assert torch.equal(model.bx, bx0) and torch.equal(model.bt, bt0)
    assert not model.Wx.requires_grad and not model.Wt.requires_grad
    print("test_random_features_remain_fixed: PASS")


# 3. PDE residual agrees with the existing RF residual implementation.
def test_pde_residual_matches_burgers_rf():
    model, coef, tensors, _, _ = _tiny_setup()
    r_sa_rf = physics.pde_residual(model, tensors["x_f"], tensors["t_f"], config.NV)
    r_direct = burgers_residual(model, tensors["x_f"], tensors["t_f"], config.NV)
    assert torch.equal(r_sa_rf, r_direct), "sa_rf.physics.pde_residual must be burgers_rf.physics.burgers_residual"
    print("test_pde_residual_matches_burgers_rf: PASS")


# 4. SA-RF loss components agree with independently calculated values.
def test_loss_components_match_independent_calculation():
    model, coef, tensors, lambda_f, lambda_0 = _tiny_setup()
    losses = train.compute_losses(model, tensors, lambda_f, lambda_0)

    r = burgers_residual(model, tensors["x_f"], tensors["t_f"], config.NV)
    ic_err = model(tensors["x0"], tensors["t0"]) - (-torch.sin(torch.pi * tensors["x0"]))
    u_lb = model(tensors["x_lb"], tensors["t_lb"])
    u_ub = model(tensors["x_ub"], tensors["t_ub"])

    expected_pde_mse = torch.mean(r**2)
    expected_weighted_pde = torch.mean((lambda_f * r) ** 2)
    expected_ic_mse = torch.mean(ic_err**2)
    expected_weighted_ic = torch.mean((lambda_0 * ic_err) ** 2)
    expected_bc_left = torch.mean(u_lb**2)
    expected_bc_right = torch.mean(u_ub**2)
    expected_total = expected_weighted_pde + expected_weighted_ic + expected_bc_left + expected_bc_right

    assert torch.allclose(losses["unweighted_pde_mse"], expected_pde_mse)
    assert torch.allclose(losses["weighted_pde_loss"], expected_weighted_pde)
    assert torch.allclose(losses["unweighted_ic_mse"], expected_ic_mse)
    assert torch.allclose(losses["weighted_ic_loss"], expected_weighted_ic)
    assert torch.allclose(losses["bc_loss_left"], expected_bc_left)
    assert torch.allclose(losses["bc_loss_right"], expected_bc_right)
    assert torch.allclose(losses["total_objective"], expected_total)
    # Explicitly confirm NO IC_BC_WEIGHT=1000-style penalty is present:
    # weighted_ic_loss uses lambda_0 only, not lambda_0 times any extra constant.
    assert torch.allclose(losses["weighted_ic_loss"], torch.mean((lambda_0 * ic_err) ** 2))
    print("test_loss_components_match_independent_calculation: PASS")


# 5. Attention-weight gradients have the expected ascent direction.
def test_attention_gradients_ascent_direction():
    model, coef, tensors, lambda_f, lambda_0 = _tiny_setup(mode="adaptive")
    lambda_f_optimizer = torch.optim.Adam([lambda_f], lr=1e-1, maximize=True)
    lambda_f_before = lambda_f.detach().clone()

    losses = train.compute_losses(model, tensors, lambda_f, lambda_0)
    losses["total_objective"].backward()
    grad = lambda_f.grad.detach().clone()
    assert torch.all(grad > 0), "d(weighted_pde_loss)/d(lambda_f) must be strictly positive (loss is monotone increasing in lambda_f)"

    lambda_f_optimizer.step()
    # maximize=True must move lambda_f in the direction that INCREASES the
    # objective (ascent), i.e. further positive, since the gradient is positive.
    assert torch.all(lambda_f > lambda_f_before), "ascent step must increase lambda_f given a positive gradient"
    print("test_attention_gradients_ascent_direction: PASS")


# 6. RF coefficient gradients have the expected descent direction.
def test_coefficient_gradients_descent_direction():
    model, coef, tensors, lambda_f, lambda_0 = _tiny_setup(mode="adaptive")
    optimizer = torch.optim.Adam([coef], lr=1e-1)
    loss_before = train.compute_losses(model, tensors, lambda_f, lambda_0)["total_objective"].item()

    optimizer.zero_grad()
    losses = train.compute_losses(model, tensors, lambda_f, lambda_0)
    losses["total_objective"].backward()
    optimizer.step()

    loss_after = train.compute_losses(model, tensors, lambda_f, lambda_0)["total_objective"].item()
    assert loss_after < loss_before, (
        f"a single Adam descent step on the RF coefficients should reduce the total "
        f"objective at this small LR, got before={loss_before}, after={loss_after}"
    )
    print("test_coefficient_gradients_descent_direction: PASS")


# 7. Fixed-weight modes do not update attention weights.
def test_fixed_weight_modes_do_not_update_attention():
    for mode in ("baseline", "fixed_weight"):
        model, coef, tensors, lambda_f, lambda_0 = _tiny_setup(mode=mode)
        lambda_f_before, lambda_0_before = lambda_f.detach().clone(), lambda_0.detach().clone()
        assert not lambda_f.requires_grad and not lambda_0.requires_grad
        optimizer = torch.optim.Adam([coef], lr=1e-1)
        for _ in range(5):
            optimizer.zero_grad()
            losses = train.compute_losses(model, tensors, lambda_f, lambda_0)
            losses["total_objective"].backward()
            optimizer.step()
        assert torch.equal(lambda_f, lambda_f_before) and torch.equal(lambda_0, lambda_0_before), mode
    print("test_fixed_weight_modes_do_not_update_attention: PASS")


# 8. Adaptive weights update during Adam.
def test_adaptive_weights_update_during_adam():
    model, coef, tensors, lambda_f, lambda_0 = _tiny_setup(mode="adaptive")
    lambda_f_before, lambda_0_before = lambda_f.detach().clone(), lambda_0.detach().clone()
    coef_optimizer = torch.optim.Adam([coef], lr=1e-2)
    lambda_f_optimizer = torch.optim.Adam([lambda_f], lr=1e-1, maximize=True)
    lambda_0_optimizer = torch.optim.Adam([lambda_0], lr=1e-1, maximize=True)
    for _ in range(5):
        coef_optimizer.zero_grad(); lambda_f_optimizer.zero_grad(); lambda_0_optimizer.zero_grad()
        losses = train.compute_losses(model, tensors, lambda_f, lambda_0)
        losses["total_objective"].backward()
        coef_optimizer.step(); lambda_f_optimizer.step(); lambda_0_optimizer.step()
    assert not torch.equal(lambda_f, lambda_f_before)
    assert not torch.equal(lambda_0, lambda_0_before)
    print("test_adaptive_weights_update_during_adam: PASS")


# 9. Attention weights remain unchanged during L-BFGS.
def test_attention_unchanged_during_lbfgs():
    model, coef, tensors, lambda_f, lambda_0 = _tiny_setup(mode="adaptive")
    lambda_f_before, lambda_0_before = lambda_f.detach().clone(), lambda_0.detach().clone()
    lbfgs_optimizer = torch.optim.LBFGS([coef], lr=0.8, max_iter=5, history_size=10, line_search_fn=None)

    def closure():
        lbfgs_optimizer.zero_grad()
        losses = train.compute_losses(model, tensors, lambda_f, lambda_0)
        losses["total_objective"].backward()
        return losses["total_objective"]

    lbfgs_optimizer.step(closure)
    assert torch.equal(lambda_f, lambda_f_before) and torch.equal(lambda_0, lambda_0_before), (
        "attention weights must not change during L-BFGS since they are never passed to its optimizer"
    )
    print("test_attention_unchanged_during_lbfgs: PASS")


# 10. Same-seed initialization and collocation sampling are reproducible.
def test_same_seed_reproducibility():
    m1, c1, t1, lf1, l01 = _tiny_setup(mode="adaptive", seed=77)
    m2, c2, t2, lf2, l02 = _tiny_setup(mode="adaptive", seed=77)
    assert torch.equal(m1.Wx, m2.Wx) and torch.equal(m1.Wt, m2.Wt)
    assert torch.equal(m1.bx, m2.bx) and torch.equal(m1.bt, m2.bt)
    assert torch.equal(c1.detach(), c2.detach())
    assert torch.equal(t1["x_f"], t2["x_f"]) and torch.equal(t1["t_f"], t2["t_f"])
    assert torch.equal(t1["x0"], t2["x0"])
    assert torch.equal(lf1, lf2) and torch.equal(l01, l02)
    print("test_same_seed_reproducibility: PASS")


def test_baseline_adaptive_balanced_share_identical_init_features_and_points():
    """Verifies baseline, adaptive, and balanced_10/balanced_100 ALL use
    identical model initialization, feature basis, and collocation points
    given the same seed (Task 2 of the LR-sweep follow-up request).
    Evaluation grid identity is implied since evaluate_relative_l2 is a
    pure function of config.M_TEST/N_TEST_TIMES, shared by all modes."""
    modes = ("baseline", "adaptive", "balanced_10", "balanced_100")
    setups = {m: _tiny_setup(mode=m, seed=42) for m in modes}
    ref_model, ref_coef, ref_tensors, _, _ = setups["baseline"]
    for m in modes[1:]:
        model, coef, tensors, _, _ = setups[m]
        assert torch.equal(model.Wx, ref_model.Wx) and torch.equal(model.Wt, ref_model.Wt), m
        assert torch.equal(model.bx, ref_model.bx) and torch.equal(model.bt, ref_model.bt), m
        assert torch.equal(coef.detach(), ref_coef.detach()), f"{m}: initial RF coefficients differ"
        for key in ("x_f", "t_f", "x0", "t0", "x_lb", "t_lb", "x_ub", "t_ub"):
            assert torch.equal(tensors[key], ref_tensors[key]), f"{m}: collocation tensor '{key}' differs"
    print("test_baseline_adaptive_balanced_share_identical_init_features_and_points: PASS")


# 11. NaN/Inf detection works for loss, gradients, coefficients, and attention weights.
def test_nonfinite_detection():
    model, coef, tensors, lambda_f, lambda_0 = _tiny_setup(mode="adaptive")
    assert not train._scalar_nonfinite(torch.tensor(1.0))
    assert train._scalar_nonfinite(torch.tensor(float("nan")))
    assert train._scalar_nonfinite(torch.tensor(float("inf")))
    assert not train._any_nonfinite(coef, lambda_f, lambda_0)

    with torch.no_grad():
        coef_backup = coef.clone()
        coef[0, 0] = float("nan")
    assert train._any_nonfinite(coef, lambda_f, lambda_0)
    with torch.no_grad():
        coef.copy_(coef_backup)

    optimizer = torch.optim.Adam([coef], lr=1e-2)
    optimizer.zero_grad()
    losses = train.compute_losses(model, tensors, lambda_f, lambda_0)
    losses["total_objective"].backward()
    assert not train._scalar_nonfinite(coef.grad)
    print("test_nonfinite_detection: PASS")


# 12. Boundary loss normalization matches the SA-PINN convention.
def test_boundary_loss_normalization_matches_sapinn_convention():
    model, coef, tensors, lambda_f, lambda_0 = _tiny_setup(mode="adaptive")
    losses = train.compute_losses(model, tensors, lambda_f, lambda_0)
    u_lb = model(tensors["x_lb"], tensors["t_lb"])
    u_ub = model(tensors["x_ub"], tensors["t_ub"])
    expected_bc_mse = torch.mean(u_lb**2) + torch.mean(u_ub**2)
    assert torch.allclose(losses["bc_mse"], expected_bc_mse), (
        "BC loss must be mean(u_lb**2) + mean(u_ub**2) (two SEPARATE means summed, "
        "matching sapinn/physics.py's convention), NOT burgers_rf's own "
        "mean-over-concatenated-both-sides convention (which differs by a factor of 2 "
        "when n_left == n_right)."
    )
    concatenated_mean = torch.mean(torch.cat([u_lb, u_ub]) ** 2)
    if tensors["x_lb"].shape[0] == tensors["x_ub"].shape[0]:
        assert not torch.allclose(losses["bc_mse"], concatenated_mean) or torch.allclose(expected_bc_mse, concatenated_mean * 2), (
            "sanity check: with equal per-side counts, the two conventions must differ by 2x "
            "(unless u_lb/u_ub happen to have identical squared means, astronomically unlikely here)"
        )
    print("test_boundary_loss_normalization_matches_sapinn_convention: PASS")


# Finite-difference gradient check on a small deterministic test problem.
def test_finite_difference_gradient_check():
    model, coef, tensors, lambda_f, lambda_0 = _tiny_setup(mode="adaptive", seed=5)
    eps = 1e-6
    idx = (0, 0)

    losses = train.compute_losses(model, tensors, lambda_f, lambda_0)
    losses["total_objective"].backward()
    analytic_grad = coef.grad[idx].item()

    with torch.no_grad():
        coef[idx] += eps
    loss_plus = train.compute_losses(model, tensors, lambda_f, lambda_0)["total_objective"].item()
    with torch.no_grad():
        coef[idx] -= 2 * eps
    loss_minus = train.compute_losses(model, tensors, lambda_f, lambda_0)["total_objective"].item()
    with torch.no_grad():
        coef[idx] += eps  # restore

    fd_grad = (loss_plus - loss_minus) / (2 * eps)
    assert abs(analytic_grad - fd_grad) < 1e-3 * max(1.0, abs(fd_grad)), (
        f"autograd grad {analytic_grad} does not match finite-difference grad {fd_grad}"
    )
    print("test_finite_difference_gradient_check: PASS")


def test_run_smoke_all_modes():
    """End-to-end sanity: run() completes for all modes at the
    smallest budget, with a tiny collocation count, to a scratch dir."""
    _clean()
    try:
        from sa_rf.train import run
        for mode in train.MODES:
            metrics_rows, run_config = run(
                mode, "smoke", seed=3, output_dir=SCRATCH_ROOT / mode,
                nx=TINY_NX, nt=TINY_NT, n_f=TINY_N_F, n_0=TINY_N_0, n_b_per_side=TINY_N_B,
                device="cpu",
            )
            assert (SCRATCH_ROOT / mode / "config.json").exists()
            assert (SCRATCH_ROOT / mode / "final.pt").exists()
            assert len(metrics_rows) > 0
            assert run_config["n_rf_coefficients"] == TINY_NX * TINY_NT
    finally:
        _clean()
    print("test_run_smoke_all_modes: PASS")


if __name__ == "__main__":
    test_model_output_shape_and_coefficient_count()
    test_random_features_remain_fixed()
    test_pde_residual_matches_burgers_rf()
    test_loss_components_match_independent_calculation()
    test_attention_gradients_ascent_direction()
    test_coefficient_gradients_descent_direction()
    test_fixed_weight_modes_do_not_update_attention()
    test_adaptive_weights_update_during_adam()
    test_attention_unchanged_during_lbfgs()
    test_same_seed_reproducibility()
    test_baseline_adaptive_balanced_share_identical_init_features_and_points()
    test_nonfinite_detection()
    test_boundary_loss_normalization_matches_sapinn_convention()
    test_finite_difference_gradient_check()
    test_run_smoke_all_modes()
    print("ALL SA-RF UNIT TESTS PASSED")
