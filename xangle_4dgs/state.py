from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import re
import tempfile


def read_json(path):
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def save_json(path, value):
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as file:
        pending = Path(file.name)
        try:
            json.dump(value, file, indent=2, allow_nan=False)
            file.flush()
            os.fsync(file.fileno())
        except BaseException:
            file.close()
            pending.unlink()
            raise
    try:
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


@contextmanager
def run_lock(root):
    with (root / ".lock").open("a+b") as lock:
        try:
            if os.name == "nt":
                import msvcrt

                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(f"Another command owns this run: {root}") from error
        yield


def load_job(file):
    job = read_json(file)
    defaults = {"stride": 2, "dataFactor": 2, "batchCameras": 4}
    allowed = {*defaults, "version", "captureDir", "preparedNpz", "start", "end", "fps",
               "steps", "inputStart", "inputEnd"}
    if unknown := job.keys() - allowed:
        raise ValueError(f"Unknown job fields: {', '.join(sorted(unknown))}")
    job = {**defaults, **job}
    if type(job.get("version")) is not int or job["version"] != 1:
        raise ValueError("Expected job version 1.")
    for key in ("start", "end", "steps", "stride", "dataFactor", "batchCameras"):
        if type(job.get(key)) is not int or job[key] < (0 if key in ("start", "end") else 1):
            raise ValueError(f"Invalid integer field: {key}")
    if job["start"] >= job["end"] or job["steps"] > 1_000_000 or job["dataFactor"] not in (1, 2):
        raise ValueError("Expected at least two iterations, at most 1,000,000 steps, and dataFactor 1 or 2.")
    for key, default in (("inputStart", "start"), ("inputEnd", "end")):
        job.setdefault(key, job[default])
        if type(job[key]) is not int:
            raise ValueError(f"Invalid integer field: {key}")
    if not 0 <= job["inputStart"] <= job["start"] < job["end"] <= job["inputEnd"]:
        raise ValueError("The training range must be inside the input range.")
    fps = job.get("fps")
    if type(fps) not in (int, float) or not math.isfinite(fps) or not 1 <= fps <= 240:
        raise ValueError("Expected fps between 1 and 240.")
    for key in ("captureDir", "preparedNpz"):
        if key == "preparedNpz" and key not in job:
            continue
        if not isinstance(job.get(key), str) or not job[key]:
            raise ValueError(f"Expected a path for {key}.")
        job[key] = str((file.parent / job[key]).resolve(strict=True))
    capture = Path(job["captureDir"])
    if not (capture / "images").is_dir() or not (capture / "sparse" / "0").is_dir():
        raise ValueError("captureDir must contain images and sparse/0.")
    if job.get("preparedNpz") and not Path(job["preparedNpz"]).is_file():
        raise ValueError("preparedNpz must be a file.")
    return job


def checkpoint(root, state):
    from .runtime import CHECKPOINT_PROTOCOL

    for name in dict.fromkeys((state.get("attempt"), state.get("savedAttempt"))):
        if name is None:
            continue
        if not re.fullmatch(r"training-[a-f0-9]{32}", name):
            raise ValueError("Invalid training attempt name.")
        directory = root / name
        stem = directory / "xangle-training"
        marker = stem.with_suffix(".json")
        if not marker.is_file():
            continue
        payload = read_json(marker)
        step = payload.get("completedSteps")
        if (any(payload.get(k) != v for k, v in CHECKPOINT_PROTOCOL.items())
                or type(step) is not int or not state.get("savedSteps", 0) <= step <= state["targetSteps"]):
            raise ValueError(f"Invalid checkpoint marker: {marker}")
        model = payload.get("checkpointFile", "")
        snapshot = payload.get("snapshotFile", "")
        if not (isinstance(model, str) and re.fullmatch(r"xangle-training\.[a-z0-9.]+\.pt", model)
                and snapshot == f"viewer_snapshots/snap_s{step:07d}.pt"):
            raise ValueError(f"Invalid checkpoint artifact paths: {marker}")
        for file in (directory / model, directory / snapshot):
            if not file.is_file() or file.stat().st_size == 0:
                raise ValueError(f"Missing checkpoint artifact: {file}")
        return step, stem, directory / snapshot
    return None
