import os
from pathlib import Path
import re
import subprocess
import sys


CHECKPOINT_PROTOCOL = {"format": "xangle_trigs_training", "version": 1}


class Runtime:
    def __init__(self, path):
        self.root = Path(path).resolve(strict=True)
        self.python = self.root / "env" / "Scripts" / "python.exe"
        self.core = self.root / "open4dgs-core"
        self.env = {**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"}
        for key in ("PYTHONPATH", "XANGLE_TRIGS_STATE_IN", "XANGLE_TRIGS_STATE_OUT",
                    "XANGLE_TRIGS_STOP_REQUEST"):
            self.env.pop(key, None)

    def run(self, args, log, *, env=None, observe=None):
        with log.open("a", encoding="utf-8") as output:
            with subprocess.Popen(
                [str(self.python), *map(str, args)], stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                env={**self.env, **(env or {})},
            ) as child:
                try:
                    assert child.stdout is not None
                    for line in child.stdout:
                        output.write(line)
                        output.flush()
                        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
                        print(line.encode(encoding, "backslashreplace").decode(encoding), end="", flush=True)
                        if observe:
                            observe(line)
                except BaseException:
                    child.terminate()
                    raise
                code = child.wait()
        if code:
            raise RuntimeError(f"Open4DGS exited with code {code}. See {log}")

    def prepare(self, job, npz, log):
        reference = npz.parent / "reference.json"
        self.run([self.core / "prep" / "align" / "convert-to-colmap-ref-trigs.py",
                  "--sparse-dir", Path(job["captureDir"]) / "sparse" / "0",
                  "--out-json", reference], log)
        matching = self.root / "matching-v2"
        flow = matching / "memfof" / "6c6c9aa3ad64f93aee8efbc2f7a6e4535814ee96"
        if not (flow / "model.safetensors").is_file():
            raise ValueError(f"Install the matching runtime or supply preparedNpz: {flow}")
        self.run([self.core / "prep" / "init" / "prep-roma-trigs-memory.py",
                  "--reference-json", reference, "--images-root", Path(job["captureDir"]) / "images",
                  "--output-path", npz, "--fps", job["fps"], "--frame-start", job["start"],
                  "--frame-end", job["end"], "--input-frame-start", job["inputStart"],
                  "--input-frame-end", job["inputEnd"], "--frame-step", job["stride"],
                  "--temporal-frame-delta", job["stride"], "--roma-backend", "v2",
                  "--roma-v2-setting", "precise", "--temporal-backend", "memfof",
                  "--memfof-checkpoint", flow, "--roma-num-matches", 12000,
                  "--target-points-per-frame", 32000, "--log-memory"], log,
                 env={"PYTHONPATH": str(matching / "packages"), "TORCH_HOME": str(matching / "torch"),
                      "HF_HUB_OFFLINE": "1"})

    def train(self, job, npz, attempt, source, log, observe):
        from .state import read_json

        if read_json(self.root / "trigs" / "xangle_checkpoint_capability.json") != CHECKPOINT_PROTOCOL:
            raise ValueError("Install the companion's TRiGS checkpoint patch before training.")
        env = {
            "XANGLE_TRIGS_STATE_IN": str(source) if source else "",
            "XANGLE_TRIGS_STATE_OUT": str(attempt / "xangle-training"),
            "XANGLE_TRIGS_STOP_REQUEST": str(attempt / "stop-request"),
        }
        self.run([Path(__file__).with_name("worker.py"), "--data-dir", job["captureDir"],
                  "--npz-path", npz, "--output-dir", attempt, "--start-frame", job["start"],
                  "--end-frame", job["end"], "--data-factor", job["dataFactor"],
                  "--config.batch-cameras", job["batchCameras"],
                  "--config.target-count-per-keyframe", 40000,
                  "--config.frame-sampling-fast-visits", 0, "--config.n-steps", job["steps"]],
                 log, env=env, observe=observe)

    def export(self, snapshot, output, log):
        self.run(["-m", "trigs.cli.export_plys", "--ckpt-path", snapshot,
                  "--output-dir", output, "--start-frame", 0], log)


PROGRESS = re.compile(r"^TRiGS:\s+\d+%\|.*?\|\s*(\d+)/(\d+)")
