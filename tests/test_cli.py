import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from xangle_4dgs import cli
from xangle_4dgs.state import load_job, save_json


class FixtureRuntime:
    def __init__(self):
        self.fail_export = False

    def train(self, job, npz, attempt, source, log, observe):
        step = job["steps"] if source else 3
        if source:
            assert json.loads(source.with_suffix(".json").read_text())["completedSteps"] == 3
        else:
            cli.stop(attempt.parent)
            assert (attempt / "stop-request").is_file()
        name = f"viewer_snapshots/snap_s{step:07d}.pt"
        (attempt / "viewer_snapshots").mkdir()
        (attempt / name).write_bytes(b"snapshot")
        (attempt / "xangle-training.model.pt").write_bytes(b"model")
        save_json(attempt / "xangle-training.json", {
            "format": "xangle_trigs_training", "version": 1, "completedSteps": step,
            "checkpointFile": "xangle-training.model.pt", "snapshotFile": name,
        })
        if source is None:
            observe("[XANGLE_STOPPED] step=3")

    def export(self, snapshot, output, log):
        assert snapshot.name == "snap_s0000006.pt"
        save_json(output / "export_meta.json", {"fps": 120, "n_frames": 2, "start_frame": 0})
        for i in range(1 if self.fail_export else 2):
            (output / f"frame_{i:06d}.ply").write_bytes(b"ply")


class JobTest(unittest.TestCase):
    def test_stop_resume_and_transactional_export(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            run = Path(temp) / "run"
            for folder in ("run", "capture/images", "capture/sparse/0"):
                (Path(temp) / folder).mkdir(parents=True)
            (Path(temp) / "sample.npz").write_bytes(b"sample")
            job = Path(temp) / "job.json"
            save_json(job, {"version": 1, "captureDir": "capture", "preparedNpz": "sample.npz",
                            "start": 10, "end": 11, "steps": 6, "fps": 120})
            rt = FixtureRuntime()
            with cli.run_lock(run):
                cli.train(run, rt, new_job=load_job(job))
            self.assertEqual(cli.status(run)["status"], "stopped")
            self.assertEqual(cli.status(run)["savedSteps"], 3)
            with cli.run_lock(run):
                cli.train(run, rt)
            self.assertEqual(cli.status(run)["status"], "trained")
            self.assertEqual(cli.status(run)["savedSteps"], 6)
            output = Path(temp) / "export"
            rt.fail_export = True
            with self.assertRaisesRegex(ValueError, "does not match"):
                cli.export(run, rt, output)
            self.assertFalse(output.exists())
            rt.fail_export = False
            cli.export(run, rt, output)
            self.assertEqual(len(list(output.glob("*.ply"))), 2)
