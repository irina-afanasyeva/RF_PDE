"""Sequential, restartable multi-seed SA-PINN Burgers campaign runner.

Orchestrates repeated calls to the EXISTING, unmodified sapinn.train.run()
for every (mode, seed) combination -- it does not reimplement or alter
training in any way. Keeping this orchestration logic in a separate script
from sapinn/train.py is deliberate (see docs/RESEARCH_LOG.md / task spec
"keep analysis scripts separate from training code").

SAFETY / RESTART DESIGN:
  - Before touching a (mode, seed) run, its on-disk state is classified as
    one of: "completed", "interrupted", "corrupted", or "not_started" (see
    classify_existing_run). This NEVER deletes or overwrites anything.
  - "completed" runs are skipped (not retrained), after verifying
    config.json, metrics.csv (non-empty), and checkpoints/final.pt are all
    present and final.pt is actually loadable -- not just "a directory
    exists".
  - "interrupted" runs (a valid checkpoints/latest.pt exists but
    config.json does not, i.e. the process stopped mid-training) are
    RESUMED via sapinn.train.run(resume_from=...), reusing the existing,
    already-tested resume mechanism -- never silently restarted from
    scratch, which would both waste the partial progress and violate "do
    not silently retrain".
  - "corrupted" states (something exists on disk that doesn't match either
    of the above -- e.g. a checkpoint file that fails to load) STOP the
    whole campaign immediately with a clear error. Nothing is deleted;
    manual inspection is required before continuing.
  - A manifest (campaign/manifest.csv) is written after every single
    (mode, seed) outcome, not just at the end, so a crash of this ORCHESTRATOR
    itself (as opposed to an individual training run) still leaves an
    accurate, resumable record. Re-running this script re-reads the
    manifest and the actual output directories (the directories are the
    ground truth; the manifest is a convenience projection of them) and
    continues exactly where it left off.
  - Per-run stdout/stderr is captured to campaign/logs/{mode}_seed{N}.log,
    preserved regardless of success or failure.
"""
import argparse
import csv
import json
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(PROJECT_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from sapinn import config  # noqa: E402
from sapinn.train import run  # noqa: E402

ALL_MODES = ["baseline", "fixed_weight", "adaptive", "balanced_10", "balanced_100"]
MANIFEST_FIELDS = [
    "mode", "seed", "budget", "status", "output_dir",
    "elapsed_seconds", "started_at", "finished_at", "error_message",
]


def parse_seed_spec(spec):
    """'1-9' -> [1..9]; '1,2,5' -> [1,2,5]; '3' -> [3]."""
    seeds = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    return seeds


def run_output_dir(output_root, mode, budget, seed):
    return Path(output_root) / mode / budget / f"seed{seed}"


def _checkpoint_loadable(path):
    try:
        torch.load(path, map_location="cpu", weights_only=False)
        return True
    except Exception:
        return False


def classify_existing_run(output_dir):
    """Returns ('completed'|'interrupted'|'corrupted'|'not_started', detail_str)."""
    output_dir = Path(output_dir)
    config_path = output_dir / "config.json"
    metrics_path = output_dir / "metrics.csv"
    final_ckpt = output_dir / "checkpoints" / "final.pt"
    latest_ckpt = output_dir / "checkpoints" / "latest.pt"

    if config_path.exists():
        try:
            json.loads(config_path.read_text())
            if not metrics_path.exists():
                return "corrupted", f"{config_path} exists but {metrics_path} is missing"
            with metrics_path.open() as f:
                n_rows = sum(1 for _ in f) - 1
            if n_rows <= 0:
                return "corrupted", f"{metrics_path} has no data rows"
            if not final_ckpt.exists():
                return "corrupted", f"config.json exists but {final_ckpt} is missing"
            if not _checkpoint_loadable(final_ckpt):
                return "corrupted", f"{final_ckpt} exists but failed to load"
            return "completed", "config.json + non-empty metrics.csv + loadable final.pt"
        except Exception as exc:
            return "corrupted", f"failed to validate completed run: {exc}"

    if latest_ckpt.exists():
        if _checkpoint_loadable(latest_ckpt):
            return "interrupted", f"{latest_ckpt} present and loadable, no config.json yet"
        return "corrupted", f"{latest_ckpt} exists but failed to load"

    if output_dir.exists() and any(output_dir.iterdir()):
        return "corrupted", f"{output_dir} exists with unrecognized partial content"

    return "not_started", "no existing output directory"


def load_manifest(manifest_path):
    if not manifest_path.exists():
        return {}
    with manifest_path.open() as f:
        return {(r["mode"], int(r["seed"])): r for r in csv.DictReader(f)}


def save_manifest(manifest_path, entries):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = manifest_path.with_suffix(".tmp")
    with tmp_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for key in sorted(entries.keys()):
            writer.writerow(entries[key])
    tmp_path.replace(manifest_path)


@contextmanager
def tee_stdout(log_path):
    """Duplicate stdout/stderr to `log_path` (appended) while still
    printing to the console, for the duration of one training run."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    class _Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()

        def flush(self):
            for s in self.streams:
                s.flush()

    with log_path.open("a") as log_f:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = _Tee(old_stdout, log_f)
        sys.stderr = _Tee(old_stderr, log_f)
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr


def main():
    parser = argparse.ArgumentParser(description="Sequential, restartable multi-seed SA-PINN campaign runner")
    parser.add_argument("--seeds", type=str, default="1-9")
    parser.add_argument("--modes", type=str, default=",".join(ALL_MODES))
    parser.add_argument("--budget", type=str, default="paper")
    parser.add_argument("--output-root", type=str, default=str(PROJECT_DIR / "outputs" / "sapinn_burgers"))
    parser.add_argument("--campaign-dir", type=str, default=None,
                         help="Defaults to <output-root>/campaign")
    parser.add_argument("--dry-run", action="store_true",
                         help="Force budget='smoke' regardless of --budget, for validating this "
                              "runner's logic with real-but-tiny training. Does NOT launch real "
                              "paper-scale training. Combine with --output-root/--campaign-dir "
                              "pointed at a scratch location to avoid touching real results.")
    args = parser.parse_args()

    budget = "smoke" if args.dry_run else args.budget
    seeds = parse_seed_spec(args.seeds)
    modes = args.modes.split(",")
    output_root = Path(args.output_root)
    campaign_dir = Path(args.campaign_dir) if args.campaign_dir else output_root / "campaign"
    manifest_path = campaign_dir / "manifest.csv"
    logs_dir = campaign_dir / "logs"

    if args.dry_run:
        print(f"[DRY RUN] budget forced to 'smoke'. output_root={output_root}, campaign_dir={campaign_dir}")

    entries = load_manifest(manifest_path)
    for seed in seeds:
        for mode in modes:
            key = (mode, seed)
            if key not in entries:
                entries[key] = {
                    "mode": mode, "seed": seed, "budget": budget, "status": "pending",
                    "output_dir": str(run_output_dir(output_root, mode, budget, seed)),
                    "elapsed_seconds": "", "started_at": "", "finished_at": "", "error_message": "",
                }
    save_manifest(manifest_path, entries)

    print(f"Campaign plan: {len(seeds)} seeds x {len(modes)} modes = {len(seeds) * len(modes)} runs "
          f"(budget={budget})")

    for seed in seeds:
        for mode in modes:
            key = (mode, seed)
            output_dir = run_output_dir(output_root, mode, budget, seed)
            status, detail = classify_existing_run(output_dir)

            if status == "completed":
                print(f"[{mode}/seed{seed}] SKIP (already completed) -- {detail}")
                entries[key].update({"status": "already_completed", "output_dir": str(output_dir),
                                      "error_message": ""})
                save_manifest(manifest_path, entries)
                continue

            if status == "corrupted":
                msg = (f"[{mode}/seed{seed}] CORRUPTED/AMBIGUOUS state at {output_dir}: {detail}. "
                       "Refusing to proceed automatically -- nothing has been deleted or modified. "
                       "Inspect manually, then re-run this campaign script (completed runs before "
                       "this one are preserved and will be skipped).")
                print(msg)
                entries[key].update({"status": "corrupted", "error_message": detail})
                save_manifest(manifest_path, entries)
                raise RuntimeError(msg)

            resume_from = None
            if status == "interrupted":
                resume_from = output_dir / "checkpoints" / "latest.pt"
                print(f"[{mode}/seed{seed}] RESUMING interrupted run from {resume_from} -- {detail}")
            else:
                print(f"[{mode}/seed{seed}] starting fresh run -> {output_dir}")

            entries[key].update({
                "status": "running", "budget": budget, "output_dir": str(output_dir),
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "error_message": "",
            })
            save_manifest(manifest_path, entries)

            log_path = logs_dir / f"{mode}_seed{seed}.log"
            start_time = time.time()
            try:
                with tee_stdout(log_path):
                    run(mode=mode, budget_name=budget, seed=seed, output_dir=output_dir,
                        resume_from=resume_from)
                elapsed = time.time() - start_time
                entries[key].update({
                    "status": "completed", "elapsed_seconds": f"{elapsed:.1f}",
                    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                })
                save_manifest(manifest_path, entries)
                print(f"[{mode}/seed{seed}] COMPLETED in {elapsed:.1f}s")
            except Exception as exc:
                elapsed = time.time() - start_time
                tb_text = traceback.format_exc()
                (logs_dir / f"{mode}_seed{seed}.error.log").write_text(tb_text)
                entries[key].update({
                    "status": "failed", "elapsed_seconds": f"{elapsed:.1f}",
                    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "error_message": str(exc),
                })
                save_manifest(manifest_path, entries)
                print(f"[{mode}/seed{seed}] FAILED after {elapsed:.1f}s: {exc}\n"
                      f"Full traceback: {logs_dir}/{mode}_seed{seed}.error.log\n"
                      "Stopping campaign. All prior completed runs and this run's partial "
                      "checkpoints/logs are preserved. Fix the issue and re-run this same "
                      "command to resume.")
                raise

    print("Campaign finished: all requested (mode, seed) runs are completed or were already complete.")


if __name__ == "__main__":
    main()
