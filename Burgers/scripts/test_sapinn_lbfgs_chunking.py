import copy
import sys
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(PROJECT_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from sapinn import attention, config, sampling  # noqa: E402
from sapinn.model import PINN  # noqa: E402
from sapinn.train import compute_losses, evaluate_relative_l2  # noqa: E402

# CONTROLLED EXPERIMENT: is calling torch.optim.LBFGS.step(closure) ONCE
# with max_iter=N equivalent to calling .step(closure) MULTIPLE times on
# the SAME optimizer instance with max_iter=N/k each (chunking)? Per the
# explicit task instruction, this is NOT assumed -- it is tested here
# BEFORE any chunking strategy is implemented in sapinn/train.py.
#
# Both cases start from IDENTICAL network parameters, training points, and
# attention weights (same seed, same construction order). Total requested
# iteration budget is identical (40) in both cases.

TOTAL_ITERS = 40
CHUNK_SIZES = [10, 10, 10, 10]  # case B: 4 calls of 10 -> same total as case A
SEED = 0
DEVICE = torch.device("cpu")


def build_state(seed):
    torch.set_default_dtype(config.DTYPE)
    config.set_all_seeds(seed)
    tensors = sampling.build_training_tensors(seed, DEVICE)
    net = PINN().to(device=DEVICE, dtype=config.DTYPE)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    col_weights, u_weights = attention.make_adaptive_weights(config.N_F, config.N_0, DEVICE, generator)
    # Freeze attention weights for THIS experiment (isolates L-BFGS network-
    # weight optimization behavior only -- attention weights are already
    # frozen during L-BFGS in the real trainer, so this matches that phase
    # exactly: col_weights/u_weights are constants here, not updated).
    col_weights = col_weights.detach().requires_grad_(False)
    u_weights = u_weights.detach().requires_grad_(False)
    return net, tensors, col_weights, u_weights


def make_closure(optimizer, net, tensors, col_weights, u_weights, call_stats):
    def closure():
        optimizer.zero_grad()
        losses = compute_losses(net, tensors, col_weights, u_weights)
        losses["total_objective"].backward()
        call_stats["closure_evals"] += 1
        call_stats["last_loss"] = losses["total_objective"].item()
        return losses["total_objective"]
    return closure


def run_case_a():
    net, tensors, col_weights, u_weights = build_state(SEED)
    optimizer = torch.optim.LBFGS(
        net.parameters(), lr=config.LBFGS_LR, max_iter=TOTAL_ITERS,
        history_size=config.LBFGS_HISTORY_SIZE, line_search_fn=None,
    )
    call_stats = {"closure_evals": 0, "last_loss": None}
    closure = make_closure(optimizer, net, tensors, col_weights, u_weights, call_stats)
    optimizer.step(closure)

    state = optimizer.state[optimizer._params[0]]
    n_iter_actual = state.get("n_iter", 0)
    func_evals = state.get("func_evals", 0)
    eval_metrics = evaluate_relative_l2(net, DEVICE)
    return {
        "final_loss": call_stats["last_loss"],
        "closure_evals": call_stats["closure_evals"],
        "n_iter_actual": n_iter_actual,
        "func_evals_state": func_evals,
        "n_calls": 1,
        "global_l2": eval_metrics["global_relative_l2_error"],
        "params": [p.detach().clone() for p in net.parameters()],
    }


def run_case_b():
    net, tensors, col_weights, u_weights = build_state(SEED)
    chunk_size = CHUNK_SIZES[0]
    assert all(c == chunk_size for c in CHUNK_SIZES), "this script assumes equal chunk sizes"
    optimizer = torch.optim.LBFGS(
        net.parameters(), lr=config.LBFGS_LR, max_iter=chunk_size,
        history_size=config.LBFGS_HISTORY_SIZE, line_search_fn=None,
    )
    call_stats = {"closure_evals": 0, "last_loss": None}
    closure = make_closure(optimizer, net, tensors, col_weights, u_weights, call_stats)

    per_call_n_iter = []
    for call_idx, chunk in enumerate(CHUNK_SIZES):
        optimizer.step(closure)
        state = optimizer.state[optimizer._params[0]]
        per_call_n_iter.append(state.get("n_iter", 0))

    state = optimizer.state[optimizer._params[0]]
    eval_metrics = evaluate_relative_l2(net, DEVICE)
    return {
        "final_loss": call_stats["last_loss"],
        "closure_evals": call_stats["closure_evals"],
        "n_iter_actual": state.get("n_iter", 0),
        "func_evals_state": state.get("func_evals", 0),
        "n_calls": len(CHUNK_SIZES),
        "per_call_n_iter": per_call_n_iter,
        "global_l2": eval_metrics["global_relative_l2_error"],
        "params": [p.detach().clone() for p in net.parameters()],
    }


def main():
    print(f"=== L-BFGS chunking equivalence experiment (total_iters={TOTAL_ITERS}) ===")
    result_a = run_case_a()
    result_b = run_case_b()

    print("\nCase A (one uninterrupted call):")
    print(f"  final_loss={result_a['final_loss']:.10e}")
    print(f"  global_relative_l2_error={result_a['global_l2']:.10e}")
    print(f"  n_iter (state, cumulative)={result_a['n_iter_actual']}")
    print(f"  func_evals (state, cumulative)={result_a['func_evals_state']}")
    print(f"  closure_evals (counted here)={result_a['closure_evals']}")

    print("\nCase B (4 calls x 10 iters, same optimizer instance):")
    print(f"  final_loss={result_b['final_loss']:.10e}")
    print(f"  global_relative_l2_error={result_b['global_l2']:.10e}")
    print(f"  n_iter (state, cumulative)={result_b['n_iter_actual']}")
    print(f"  per-call n_iter={result_b['per_call_n_iter']}")
    print(f"  func_evals (state, cumulative)={result_b['func_evals_state']}")
    print(f"  closure_evals (counted here)={result_b['closure_evals']}")

    loss_diff = abs(result_a["final_loss"] - result_b["final_loss"])
    l2_diff = abs(result_a["global_l2"] - result_b["global_l2"])
    max_param_diff = max(
        (pa - pb).abs().max().item() for pa, pb in zip(result_a["params"], result_b["params"])
    )

    print("\n=== Comparison ===")
    print(f"  |final_loss_A - final_loss_B| = {loss_diff:.6e}")
    print(f"  |global_L2_A - global_L2_B|   = {l2_diff:.6e}")
    print(f"  max |param_A - param_B|       = {max_param_diff:.6e}")
    print(f"  closure_evals: A={result_a['closure_evals']}  B={result_b['closure_evals']}")
    print(f"  n_iter (cumulative): A={result_a['n_iter_actual']}  B={result_b['n_iter_actual']}")

    identical = max_param_diff == 0.0
    print(f"\nBIT-IDENTICAL PARAMETERS: {identical}")
    if not identical:
        print("Chunking does NOT reproduce the single-call trajectory exactly.")
    else:
        print("Chunking reproduces the single-call trajectory exactly, for this configuration.")


if __name__ == "__main__":
    main()
