import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(PROJECT_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from sapinn import config  # noqa: E402
from sapinn.train import run  # noqa: E402

# Stage A, self-adaptive PINN Burgers benchmark (2026-10-21 advisor
# direction; see docs/meetings/2026-10-21.md, docs/RESEARCH_LOG.md Section
# 40). Isolated from burgers_rf/ -- does not modify or import any existing
# RF training script. A single parameterized entry point is used for both
# the conventional baseline and SA-PINN so they are structurally
# guaranteed to share architecture/data/seed; see sapinn/config.py for the
# full, explicitly-tagged provenance of every hyperparameter ([CODE] =
# verified from github.com/levimcclenny/SA-PINNs, [PAPER] = verified from
# arXiv:2009.04544, [CHOICE] = disclosed implementation decision).
#
# OUTPUT PATH NOTE: seed is now part of the output path
# (outputs/sapinn_burgers/{mode}/{budget}/seed{N}/), added when periodic
# checkpointing/resume/overwrite-protection were implemented. The
# PRE-EXISTING smoke/repo_default runs saved under the OLD flat path
# (outputs/sapinn_burgers/{mode}/{budget}/, no seed subfolder) are left
# exactly where they are -- not moved, not deleted, not touched by this
# change. Because the new path always includes a seed{N} segment, it is
# structurally distinct from the old flat path, so re-running the same
# mode/budget/seed combination can never collide with those earlier results.

OUTPUT_ROOT = PROJECT_DIR / "outputs" / "sapinn_burgers"


def parse_args():
    parser = argparse.ArgumentParser(description="SA-PINN / conventional-PINN Burgers benchmark")
    parser.add_argument("--mode", choices=["baseline", "adaptive", "fixed_weight"], required=True)
    parser.add_argument("--budget", choices=list(config.BUDGETS.keys()), default="smoke")
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    parser.add_argument("--resume", type=str, default=None,
                         help="Path to a checkpoint (e.g. .../checkpoints/latest.pt) to resume an interrupted run from.")
    parser.add_argument("--force", action="store_true",
                         help="Allow overwriting an existing completed/in-progress run directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = OUTPUT_ROOT / args.mode / args.budget / f"seed{args.seed}"
    run(
        mode=args.mode, budget_name=args.budget, seed=args.seed, output_dir=output_dir,
        resume_from=args.resume, overwrite=args.force,
    )


if __name__ == "__main__":
    main()
