import sys
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(PROJECT_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from burgers_rf.config import DTYPE as DEFAULT_DTYPE, M_TEST, N_TEST_TIMES, NV as DEFAULT_NV, set_seed  # noqa: E402
from burgers_rf.data import make_reference_values, make_test_grid, sample_training_points  # noqa: E402
from burgers_rf.model import BurgersRF  # noqa: E402
from burgers_rf.evaluation import evaluate_error_metrics  # noqa: E402
from sweep_local_width import build_shock_grid  # noqa: E402

# ONE-TIME utility: recover the fixed RF basis Wx/Wt/bx/bt that is MISSING
# from the GPU-trained checkpoint (root cause: nn.Parameter(...).to(device)
# silently downgrades to a plain Tensor when an actual CPU->CUDA conversion
# occurs, so Wx/Wt/bx/bt were never registered in state_dict() on the GPU
# server -- training itself was unaffected, only checkpoint completeness).
#
# This script does NOT train, does NOT use an optimizer, and does NOT modify
# checkpoint.pt. It reproduces the EXACT pre-model RNG sequence from the
# checkpoint's own recorded seed/config metadata, reconstructs Wx/Wt/bx/bt
# deterministically, loads ONLY the already-trained model.0.weight on top,
# verifies the result against the canonical PDE-trained baseline metrics,
# and ONLY on success saves a new, separate, complete checkpoint.

CHECKPOINT_PATH = PROJECT_DIR / "outputs" / "pde_trained_rf_residual" / "checkpoint.pt"
COMPLETE_CHECKPOINT_PATH = PROJECT_DIR / "outputs" / "pde_trained_rf_residual" / "checkpoint_complete.pt"

CANONICAL_GLOBAL = 0.27149430095802507
CANONICAL_SMOOTH = 0.2443827084159696
CANONICAL_SHOCK = 0.8390487998889169
VERIFY_RTOL = 1e-4

EXPECTED_STATE_DICT_KEYS = {"Wx", "Wt", "bx", "bt", "model.0.weight"}


def load_broken_checkpoint():
    return torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)


def resolve_dtype(checkpoint):
    if "DTYPE" in checkpoint:
        # checkpoint stores a readable string, e.g. "torch.float64"
        return getattr(torch, checkpoint["DTYPE"].split(".")[-1])
    return DEFAULT_DTYPE


def reconstruct_model(checkpoint):
    """Reproduce the EXACT pre-model RNG sequence
        torch.set_default_dtype(DTYPE)
        set_seed(seed)
        sample_training_points(M_TRAIN, device="cpu")
        BurgersRF(NX, NT, SIGMA_X, SIGMA_T, use_local=use_local, device="cpu")
    using the checkpoint's OWN recorded metadata (not re-duplicated literals),
    so Wx/Wt/bx/bt come out deterministically identical to the original
    GPU-trained run. Then load ONLY the trained model.0.weight on top --
    Wx/Wt/bx/bt are NEVER loaded from the checkpoint, only reconstructed.
    """
    dtype = resolve_dtype(checkpoint)
    torch.set_default_dtype(dtype)

    seed = checkpoint["seed"]
    m_train = checkpoint["M_TRAIN"]
    nx = checkpoint["NX"]
    nt = checkpoint["NT"]
    sigma_x = checkpoint["SIGMA_X"]
    sigma_t = checkpoint["SIGMA_T"]
    use_local = checkpoint["use_local"]

    set_seed(seed)
    sample_training_points(m_train, "cpu")  # burns the RNG exactly as the original run did; output unused

    model = BurgersRF(nx, nt, sigma_x, sigma_t, use_local=use_local, device="cpu")

    with torch.no_grad():
        model.model[0].weight.copy_(checkpoint["model_state_dict"]["model.0.weight"])
    model.eval()

    return model, dtype


def verify_canonical_metrics(model, checkpoint):
    """Recompute canonical global/smooth/shock metrics using the same
    evaluation machinery as test_pde_trained_rf_baseline.py, and compare
    against BOTH the hardcoded canonical reference and the checkpoint's own
    saved metrics. Raises (and does NOT save anything) if any check fails."""
    nv = checkpoint.get("NV", DEFAULT_NV)
    device = "cpu"

    x_test, t_test = make_test_grid(M_TEST, N_TEST_TIMES, device)
    y_true = make_reference_values(t_test, x_test, nv)
    x_shock, t_shock = build_shock_grid(t_test, device)
    y_true_shock = make_reference_values(t_shock, x_shock, nv)

    with torch.no_grad():
        y_pred = model(x_test, t_test).cpu().numpy().reshape(-1)
        y_pred_shock = model(x_shock, t_shock).cpu().numpy().reshape(-1)

    metrics = evaluate_error_metrics(x_test, y_true, y_pred, y_true_shock, y_pred_shock)

    reference_sets = {
        "canonical": {
            "global_relative_l2_error": CANONICAL_GLOBAL,
            "smooth_relative_l2_error": CANONICAL_SMOOTH,
            "shock_relative_l2_error": CANONICAL_SHOCK,
        },
    }
    if all(k in checkpoint for k in ("global_relative_l2_error", "smooth_relative_l2_error", "shock_relative_l2_error")):
        reference_sets["checkpoint"] = {
            "global_relative_l2_error": checkpoint["global_relative_l2_error"],
            "smooth_relative_l2_error": checkpoint["smooth_relative_l2_error"],
            "shock_relative_l2_error": checkpoint["shock_relative_l2_error"],
        }

    print("Reconstruction verification:")
    all_passed = True
    for ref_name, ref_values in reference_sets.items():
        print(f"  --- vs. {ref_name} ---")
        for key, ref_val in ref_values.items():
            recon_val = metrics[key]
            abs_diff = abs(recon_val - ref_val)
            rel_diff = abs_diff / abs(ref_val)
            passed = rel_diff <= VERIFY_RTOL
            all_passed = all_passed and passed
            status = "PASS" if passed else "FAIL"
            print(f"    {key}:")
            print(f"      reconstructed = {recon_val}")
            print(f"      {ref_name:<9} = {ref_val}")
            print(f"      abs diff      = {abs_diff:.6e}")
            print(f"      rel diff      = {rel_diff:.6e}  (tolerance {VERIFY_RTOL:.1e})  [{status}]")

    if not all_passed:
        raise RuntimeError(
            "Reconstruction verification FAILED: the reconstructed model does not "
            "reproduce the canonical PDE-trained RF baseline within tolerance. "
            "Refusing to save a complete checkpoint."
        )
    print("All verification checks PASSED.")
    return metrics


def save_complete_checkpoint(model, checkpoint, metrics):
    state_dict_keys = set(model.state_dict().keys())
    if state_dict_keys != EXPECTED_STATE_DICT_KEYS:
        raise RuntimeError(
            f"Reconstructed model.state_dict() keys {sorted(state_dict_keys)} do not "
            f"match the expected keys {sorted(EXPECTED_STATE_DICT_KEYS)}. Refusing to "
            "save a complete checkpoint."
        )
    print(f"Confirmed state_dict contains exactly: {sorted(state_dict_keys)}")

    complete = dict(checkpoint)  # preserves all original metadata fields unchanged
    complete["model_state_dict"] = model.state_dict()
    complete["reconstructed_from"] = str(CHECKPOINT_PATH)
    complete["reconstruction_method"] = (
        "Wx/Wt/bx/bt reconstructed by reproducing the exact pre-model RNG sequence "
        "(set_seed -> sample_training_points -> BurgersRF construction) using the "
        "checkpoint's own recorded seed/config metadata; model.0.weight loaded "
        "directly from the original checkpoint.pt. Verified against canonical "
        f"global/smooth/shock metrics within relative tolerance {VERIFY_RTOL:.1e}."
    )
    complete["verified_global_relative_l2_error"] = metrics["global_relative_l2_error"]
    complete["verified_smooth_relative_l2_error"] = metrics["smooth_relative_l2_error"]
    complete["verified_shock_relative_l2_error"] = metrics["shock_relative_l2_error"]

    torch.save(complete, COMPLETE_CHECKPOINT_PATH)
    print(f"Saved complete checkpoint to {COMPLETE_CHECKPOINT_PATH}")


def main():
    checkpoint = load_broken_checkpoint()
    print(f"Loaded (incomplete) checkpoint from {CHECKPOINT_PATH}")
    print(f"Original model_state_dict keys: {list(checkpoint['model_state_dict'].keys())}")

    model, dtype = reconstruct_model(checkpoint)
    print(f"Reconstructed model.state_dict() keys: {list(model.state_dict().keys())}")

    metrics = verify_canonical_metrics(model, checkpoint)

    save_complete_checkpoint(model, checkpoint, metrics)
    print("[reconstruct_pde_trained_rf_checkpoint] done.")


if __name__ == "__main__":
    main()
