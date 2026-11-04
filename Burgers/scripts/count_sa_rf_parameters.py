"""Phase 3: programmatic parameter counting for SA-RF, reported separately
for RF coefficients (the function-approximator analog of SA-PINN's
network weights) and attention weights, plus an explicit check for any
OTHER trainable parameters in BurgersRF (bias terms, trainable feature
geometry, local Gaussian enrichment) that would need to be included.

This is a read-only diagnostic: it constructs models but trains nothing
and writes nothing to disk.
"""
import sys
from pathlib import Path

import torch

BURGERS_DIR = Path(__file__).resolve().parents[1]
if str(BURGERS_DIR) not in sys.path:
    sys.path.insert(0, str(BURGERS_DIR))

from sa_rf import config as sa_rf_config  # noqa: E402
from sa_rf.train import build_model  # noqa: E402

# SA-PINN reference counts (docs/RESEARCH_LOG.md Section 41), for comparison.
SAPINN_NETWORK_PARAMS = 3021
SAPINN_PDE_ATTENTION_PARAMS = 10000
SAPINN_IC_ATTENTION_PARAMS = 100
SAPINN_TOTAL_ADAM_PARAMS = SAPINN_NETWORK_PARAMS + SAPINN_PDE_ATTENTION_PARAMS + SAPINN_IC_ATTENTION_PARAMS


def count_rf_trainable_and_frozen_parameters(model):
    """Partitions ALL of BurgersRF's registered parameters into trainable
    vs. frozen, by name, so nothing (bias, local-feature coefficients,
    trainable geometry) is silently missed."""
    trainable = {}
    frozen = {}
    for name, p in model.named_parameters():
        (trainable if p.requires_grad else frozen)[name] = p.numel()
    return trainable, frozen


def main():
    torch.set_default_dtype(sa_rf_config.DTYPE)
    device = "cpu"
    nx, nt = sa_rf_config.NX_MATCHED, sa_rf_config.NT_MATCHED
    model = build_model(nx, nt, device)

    trainable, frozen = count_rf_trainable_and_frozen_parameters(model)
    n_coefficients = sum(trainable.values())
    n_frozen = sum(frozen.values())

    print(f"BurgersRF(Nx={nx}, Nt={nt}, use_local=False) parameter audit:")
    print(f"  Trainable parameters (by name): {trainable}")
    print(f"  Frozen parameters (by name):    {frozen}")
    print(f"  Total TRAINABLE (RF coefficients): {n_coefficients}")
    print(f"  Total FROZEN (Wx/Wt/bx/bt, basis geometry): {n_frozen}")
    assert n_coefficients == nx * nt, (
        f"expected exactly nx*nt={nx*nt} trainable coefficients (the linear-layer weight "
        f"only, bias=False), got {n_coefficients} -- investigate before trusting the "
        "parameter-matching prototype."
    )
    assert trainable.keys() == {"model.0.weight"}, (
        f"expected the ONLY trainable parameter to be the RF linear layer's weight, found "
        f"{list(trainable.keys())} -- this would mean BurgersRF has bias terms, trainable "
        "feature geometry, or local-enrichment coefficients that must be accounted for."
    )
    print(f"  CONFIRMED: no bias terms (bias=False), no trainable feature geometry "
          f"(Wx/Wt/bx/bt all requires_grad=False), no local-enrichment coefficients "
          f"(use_local=False) -- {n_coefficients} is the complete trainable-parameter count.")

    n_attention_sa_rf = sa_rf_config.N_F + sa_rf_config.N_0
    n_total_sa_rf = n_coefficients + n_attention_sa_rf

    print()
    print("=== Parameter count comparison (SA-PINN vs. SA-RF prototype) ===")
    print(f"{'':30s} {'SA-PINN':>12s} {'SA-RF':>12s}")
    print(f"{'Network / RF coefficients':30s} {SAPINN_NETWORK_PARAMS:>12d} {n_coefficients:>12d}")
    print(f"{'PDE attention (lambda_f/col)':30s} {SAPINN_PDE_ATTENTION_PARAMS:>12d} {sa_rf_config.N_F:>12d}")
    print(f"{'IC attention (lambda_0/u)':30s} {SAPINN_IC_ATTENTION_PARAMS:>12d} {sa_rf_config.N_0:>12d}")
    print(f"{'Total (Adam phase)':30s} {SAPINN_TOTAL_ADAM_PARAMS:>12d} {n_total_sa_rf:>12d}")
    print()
    print("NOTE: matching the 3,021 network/coefficient count is a PROPOSED interpretation "
          "of the professors' 'match trainable parameters' direction, NOT yet confirmed -- "
          "see docs/meetings/2026-10-28.md. The attention-parameter counts (10,000/100) are "
          "matched 'for free' here only because sa_rf.config currently defaults N_F/N_0 to "
          "SA-PINN's own current values, not because parameter-matching was separately "
          "required for the attention weights.")

    return {
        "sa_rf_coefficients": n_coefficients,
        "sa_rf_pde_attention": sa_rf_config.N_F,
        "sa_rf_ic_attention": sa_rf_config.N_0,
        "sa_rf_total_adam_params": n_total_sa_rf,
        "sapinn_network_params": SAPINN_NETWORK_PARAMS,
        "sapinn_pde_attention": SAPINN_PDE_ATTENTION_PARAMS,
        "sapinn_ic_attention": SAPINN_IC_ATTENTION_PARAMS,
        "sapinn_total_adam_params": SAPINN_TOTAL_ADAM_PARAMS,
    }


if __name__ == "__main__":
    main()
