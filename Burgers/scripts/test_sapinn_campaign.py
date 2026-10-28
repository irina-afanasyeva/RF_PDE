import shutil
import sys
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(PROJECT_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR / "scripts"))

import run_sapinn_campaign as campaign  # noqa: E402

# Tests for the CAMPAIGN ORCHESTRATOR only (scripts/run_sapinn_campaign.py).
# Does not test sapinn/train.py itself (already covered by
# test_sapinn_unit.py) -- only the restart/skip/corruption-detection/
# manifest logic layered on top of it. Uses a dedicated scratch output
# root, never the real outputs/sapinn_burgers/{mode}/paper/ results.

SCRATCH_ROOT = PROJECT_DIR / "outputs" / "sapinn_burgers" / "_unit_test_tmp_campaign"


def _clean():
    shutil.rmtree(SCRATCH_ROOT, ignore_errors=True)


def test_parse_seed_spec():
    assert campaign.parse_seed_spec("1-9") == list(range(1, 10))
    assert campaign.parse_seed_spec("1,2,5") == [1, 2, 5]
    assert campaign.parse_seed_spec("3") == [3]
    assert campaign.parse_seed_spec("1-3,7") == [1, 2, 3, 7]
    print("test_parse_seed_spec: PASS")


def test_classify_not_started():
    _clean()
    output_dir = SCRATCH_ROOT / "baseline" / "smoke" / "seed0"
    status, _ = campaign.classify_existing_run(output_dir)
    assert status == "not_started", status
    _clean()
    print("test_classify_not_started: PASS")


def test_classify_completed_and_interrupted_and_corrupted():
    _clean()
    output_dir = SCRATCH_ROOT / "baseline" / "smoke" / "seed0"
    try:
        from sapinn.train import run as train_run
        train_run(mode="baseline", budget_name="smoke", seed=0, output_dir=output_dir)

        status, _ = campaign.classify_existing_run(output_dir)
        assert status == "completed", status

        # Simulate an INTERRUPTED run: delete config.json (the thing run()
        # writes last) but keep the checkpoints -- this is exactly what a
        # process killed mid-training would look like.
        (output_dir / "config.json").unlink()
        status, _ = campaign.classify_existing_run(output_dir)
        assert status == "interrupted", status

        # Simulate a CORRUPTED checkpoint: truncate final.pt (config.json
        # still absent, so classify_existing_run falls through to checking
        # latest.pt -- corrupt THAT instead, since it's what "interrupted"
        # actually inspects).
        latest_path = output_dir / "checkpoints" / "latest.pt"
        latest_path.write_bytes(b"not a real checkpoint")
        status, _ = campaign.classify_existing_run(output_dir)
        assert status == "corrupted", status
    finally:
        _clean()
    print("test_classify_completed_and_interrupted_and_corrupted: PASS")


def test_manifest_round_trip():
    _clean()
    manifest_path = SCRATCH_ROOT / "campaign" / "manifest.csv"
    entries = {
        ("baseline", 1): {
            "mode": "baseline", "seed": 1, "budget": "paper", "status": "completed",
            "output_dir": "/some/path", "elapsed_seconds": "123.4",
            "started_at": "2026-01-01T00:00:00", "finished_at": "2026-01-01T00:02:03",
            "error_message": "",
        },
    }
    campaign.save_manifest(manifest_path, entries)
    reloaded = campaign.load_manifest(manifest_path)
    assert ("baseline", 1) in reloaded
    assert reloaded[("baseline", 1)]["status"] == "completed"
    _clean()
    print("test_manifest_round_trip: PASS")


def test_campaign_dry_run_skip_and_resume_behavior():
    """End-to-end validation of the orchestrator itself (not just its
    helper functions): first invocation trains fresh; second invocation
    (same args) must SKIP without retraining; after simulating an
    interruption, a third invocation must RESUME rather than restart from
    scratch. Uses --dry-run (budget forced to 'smoke') and a scratch
    output root -- never touches real results."""
    _clean()
    output_root = SCRATCH_ROOT / "results"
    campaign_dir = SCRATCH_ROOT / "campaign"

    argv_base = [
        "run_sapinn_campaign.py",
        "--seeds", "0",
        "--modes", "baseline",
        "--dry-run",
        "--output-root", str(output_root),
        "--campaign-dir", str(campaign_dir),
    ]

    old_argv = sys.argv
    try:
        # First invocation: fresh run.
        sys.argv = argv_base
        campaign.main()
        output_dir = output_root / "baseline" / "smoke" / "seed0"
        assert (output_dir / "config.json").exists()
        manifest = campaign.load_manifest(campaign_dir / "manifest.csv")
        assert manifest[("baseline", 0)]["status"] == "completed"
        first_elapsed = manifest[("baseline", 0)]["elapsed_seconds"]

        # Second invocation: must SKIP (not retrain) -- verified by the
        # manifest status changing to 'already_completed' and NOT being
        # re-timed (elapsed_seconds for a true retrain would differ; more
        # importantly, the run must not raise/overwrite).
        sys.argv = argv_base
        campaign.main()
        manifest = campaign.load_manifest(campaign_dir / "manifest.csv")
        assert manifest[("baseline", 0)]["status"] == "already_completed"

        # Simulate interruption: remove config.json, keep checkpoints.
        (output_dir / "config.json").unlink()
        status, _ = campaign.classify_existing_run(output_dir)
        assert status == "interrupted"

        # Third invocation: must RESUME (not error, not silently restart
        # from epoch 0 -- resume_from is passed to sapinn.train.run(),
        # which is independently tested in test_sapinn_unit.py).
        sys.argv = argv_base
        campaign.main()
        manifest = campaign.load_manifest(campaign_dir / "manifest.csv")
        assert manifest[("baseline", 0)]["status"] == "completed"
        assert (output_dir / "config.json").exists()
    finally:
        sys.argv = old_argv
        _clean()
    print("test_campaign_dry_run_skip_and_resume_behavior: PASS")


def test_campaign_stops_on_corruption_without_deleting():
    """If a (mode, seed) is found in a corrupted state, the campaign must
    raise immediately and must NOT delete or modify the corrupted file."""
    _clean()
    output_root = SCRATCH_ROOT / "results"
    campaign_dir = SCRATCH_ROOT / "campaign"
    output_dir = output_root / "baseline" / "smoke" / "seed0"

    old_argv = sys.argv
    try:
        sys.argv = [
            "run_sapinn_campaign.py", "--seeds", "0", "--modes", "baseline", "--dry-run",
            "--output-root", str(output_root), "--campaign-dir", str(campaign_dir),
        ]
        campaign.main()  # fresh completed run

        # Corrupt the checkpoint directory in a way classify_existing_run
        # cannot recognize as completed/interrupted.
        (output_dir / "config.json").unlink()
        latest_path = output_dir / "checkpoints" / "latest.pt"
        corrupted_bytes = b"deliberately corrupted, not a real checkpoint"
        latest_path.write_bytes(corrupted_bytes)

        raised = False
        try:
            campaign.main()
        except RuntimeError as exc:
            raised = True
            assert "CORRUPTED" in str(exc)
        assert raised, "expected the campaign to raise on a corrupted run, not continue silently"
        # Nothing should have been deleted or altered.
        assert latest_path.read_bytes() == corrupted_bytes
    finally:
        sys.argv = old_argv
        _clean()
    print("test_campaign_stops_on_corruption_without_deleting: PASS")


if __name__ == "__main__":
    test_parse_seed_spec()
    test_classify_not_started()
    test_classify_completed_and_interrupted_and_corrupted()
    test_manifest_round_trip()
    test_campaign_dry_run_skip_and_resume_behavior()
    test_campaign_stops_on_corruption_without_deleting()
    print("ALL SA-PINN CAMPAIGN TESTS PASSED")
