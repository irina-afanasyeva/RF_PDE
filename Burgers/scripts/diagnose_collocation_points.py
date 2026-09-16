"""
Diagnostic-only script for the fixed interior PDE collocation sample.

Does not construct a model, does not train, and does not change sampling:
only reproduces the exact fixed draw via set_seed(SEED) +
sample_training_points(M_TRAIN, device="cpu") and reports its spatial
distribution near the Burgers shock.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from burgers_rf.config import M_TRAIN, SEED, set_seed  # noqa: E402
from burgers_rf.data import sample_training_points  # noqa: E402

SHOCK_HALF_WIDTH = 0.02
SIGMAS = [1e-4, 5e-4, 2.5e-3, 1.25e-2, 6.25e-2, 3.125e-1]
SUPPORT_MULTIPLES = [1, 2, 3]

OUTPUT_DIR = PROJECT_DIR / "outputs" / "diagnostics"


def expected_count(half_width, n_total):
    """P(|x| <= r) = r for Uniform(-1,1), clipped to the physical domain."""
    r_clipped = min(half_width, 1.0)
    return n_total * r_clipped


def main():
    set_seed(SEED)
    x_train, t_train = sample_training_points(M_TRAIN, "cpu")
    x = x_train.detach().numpy().reshape(-1)
    t = t_train.detach().numpy().reshape(-1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # (A) full-domain scatter
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, t, s=4, alpha=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"Interior collocation points (M_TRAIN={M_TRAIN}, SEED={SEED})")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "collocation_scatter_full.png")
    plt.close(fig)

    # (B) zoomed shock-region scatter
    shock_mask = np.abs(x) <= SHOCK_HALF_WIDTH
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x[shock_mask], t[shock_mask], s=12, alpha=0.7)
    ax.axvline(0.0, color="k", linestyle="-", linewidth=1, label="x=0")
    ax.axvline(-SHOCK_HALF_WIDTH, color="r", linestyle="--", linewidth=1, label=f"x=-{SHOCK_HALF_WIDTH}")
    ax.axvline(SHOCK_HALF_WIDTH, color="r", linestyle="--", linewidth=1, label=f"x=+{SHOCK_HALF_WIDTH}")
    ax.set_xlim(-SHOCK_HALF_WIDTH, SHOCK_HALF_WIDTH)
    ax.set_ylim(0, 1)
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_title(f"Interior collocation points, |x| <= {SHOCK_HALF_WIDTH}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "collocation_scatter_shock.png")
    plt.close(fig)

    # (C) shock-region summary
    shock_expected = expected_count(SHOCK_HALF_WIDTH, M_TRAIN)
    shock_actual = int(shock_mask.sum())
    shock_pct = 100 * shock_actual / M_TRAIN

    summary_lines = []
    summary_lines.append(f"M_TRAIN = {M_TRAIN}")
    summary_lines.append(f"SEED = {SEED}")
    summary_lines.append(f"x range: [{x.min():.6f}, {x.max():.6f}]")
    summary_lines.append(f"t range: [{t.min():.6f}, {t.max():.6f}]")
    summary_lines.append("")
    summary_lines.append(f"Shock region |x| <= {SHOCK_HALF_WIDTH}:")
    summary_lines.append(f"  expected_count = {shock_expected:.3f}")
    summary_lines.append(f"  actual_count   = {shock_actual}")
    summary_lines.append(f"  actual_percentage = {shock_pct:.4f}%")
    summary_lines.append("")

    csv_rows = []
    header = [
        "sigma",
        "support_multiple",
        "support_half_width",
        "expected_count",
        "actual_count",
        "actual_percentage",
    ]
    csv_rows.append(header)

    summary_lines.append(
        f"{'sigma':>10} {'mult':>5} {'half_width':>11} {'expected_n':>11} "
        f"{'actual_n':>9} {'actual_%':>9}"
    )
    for sigma in SIGMAS:
        for mult in SUPPORT_MULTIPLES:
            half_width = mult * sigma
            exp_n = expected_count(half_width, M_TRAIN)
            r_clipped = min(half_width, 1.0)
            actual_n = int((np.abs(x) <= r_clipped).sum())
            actual_pct = 100 * actual_n / M_TRAIN

            summary_lines.append(
                f"{sigma:>10.2e} {mult:>5d} {half_width:>11.4e} {exp_n:>11.3f} "
                f"{actual_n:>9d} {actual_pct:>8.4f}%"
            )
            csv_rows.append([sigma, mult, half_width, exp_n, actual_n, actual_pct])

    summary_text = "\n".join(summary_lines)
    (OUTPUT_DIR / "collocation_summary.txt").write_text(summary_text + "\n")

    with (OUTPUT_DIR / "collocation_summary.csv").open("w") as f:
        for row in csv_rows:
            f.write(",".join(str(v) for v in row) + "\n")

    print(summary_text)
    print(f"\nSaved plots and summary to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
