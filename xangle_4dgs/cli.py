import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from uuid import uuid4

from .runtime import PROGRESS, Runtime
from .state import checkpoint, load_job, read_json, run_lock, save_json


def train(root, runtime, *, new_job=None):
    record = root / "status.json"
    if new_job is not None:
        save_json(root / "job.json", new_job)
        state = {"version": 1, "targetSteps": new_job["steps"], "savedSteps": 0}
    else:
        state = read_json(record)
    job = load_job(root / "job.json")
    log = root / "run.log"
    prior = checkpoint(root, state)
    if prior and prior[0] == state["targetSteps"]:
        raise ValueError("The run has reached its target. Export its saved snapshot.")

    try:
        npz = Path(job["preparedNpz"]) if "preparedNpz" in job else root / "keyframes.npz"
        if not job.get("preparedNpz") and not state.get("prepared"):
            state["status"] = "preparing"
            save_json(record, state)
            runtime.prepare(job, npz, log)
            if not npz.is_file() or npz.stat().st_size == 0:
                raise RuntimeError("Preparation did not produce a keyframe archive.")
            state["prepared"] = True
            save_json(record, state)

        attempt = root / f"training-{uuid4().hex}"
        attempt.mkdir()
        if prior:
            state.update(savedAttempt=prior[1].parent.name, savedSteps=prior[0])
        state.update(status="training", attempt=attempt.name, observedSteps=state["savedSteps"])
        state.pop("error", None)
        save_json(record, state)
        stopped_step = None
        last_status_write = 0.0

        def observe(line):
            nonlocal stopped_step, last_status_write
            if match := PROGRESS.search(line):
                step = int(match[1])
                if state["observedSteps"] < step <= state["targetSteps"] and time.monotonic() - last_status_write >= 1:
                    state["observedSteps"] = step
                    save_json(record, state)
                    last_status_write = time.monotonic()
            if line.startswith("[XANGLE_STOPPED] step="):
                stopped_step = int(line.split("=", 1)[1])

        runtime.train(job, npz, attempt, prior[1] if prior else None, log, observe)
        saved = checkpoint(root, state)
        if (not saved or saved[1].parent != attempt
                or saved[0] not in (stopped_step, state["targetSteps"])):
            raise RuntimeError("Training exited without a verified stop or final checkpoint.")
        state.update(status="trained" if saved[0] == state["targetSteps"] else "stopped",
                     observedSteps=saved[0], savedSteps=saved[0], savedAttempt=attempt.name)
        save_json(record, state)
        print(json.dumps(state), flush=True)
    except (OSError, ValueError, RuntimeError, KeyboardInterrupt) as error:
        state.update(status="failed", error=str(error) or "Interrupted")
        save_json(record, state)
        raise


def stop(root):
    state = read_json(root / "status.json")
    if state["status"] != "training":
        raise ValueError("Stop is available while an optimizer attempt is running.")
    try:
        with run_lock(root):
            raise ValueError("The recorded training attempt is no longer active.")
    except RuntimeError:
        pass
    attempt = state["attempt"]
    if not attempt.startswith("training-") or Path(attempt).name != attempt:
        raise ValueError("Invalid training attempt.")
    (root / attempt / "stop-request").touch()
    print(json.dumps({"status": "stop-requested", "attempt": attempt}))


def status(root):
    state = read_json(root / "status.json")
    if saved := checkpoint(root, state):
        state["savedSteps"] = saved[0]
    if state["status"] in ("training", "preparing"):
        try:
            with run_lock(root):
                state["status"] = "interrupted"
        except RuntimeError:
            if state["status"] == "training" and (root / state["attempt"] / "stop-request").is_file():
                state["status"] = "stop-requested"
    return state


def export(root, runtime, destination):
    if destination.exists():
        raise ValueError(f"Export destination already exists: {destination}")
    state = read_json(root / "status.json")
    saved = checkpoint(root, state)
    if not saved:
        raise ValueError("The run has no committed snapshot to export.")
    job = load_job(root / "job.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".4dgs-export-", dir=destination.parent))
    try:
        runtime.export(saved[2], staging, root / "run.log")
        metadata = read_json(staging / "export_meta.json")
        count = job["end"] - job["start"] + 1
        names = {path.name for path in staging.glob("*.ply") if path.is_file() and path.stat().st_size > 0}
        if (metadata.get("n_frames") != count or metadata.get("start_frame") != 0
                or not math.isclose(metadata["fps"], job["fps"], abs_tol=0.0001)
                or names != {f"frame_{index:06d}.ply" for index in range(count)}):
            raise ValueError("PLY export does not match the capture range and FPS.")
        if destination.exists():
            raise ValueError(f"Export destination already exists: {destination}")
        staging.rename(destination)
    finally:
        if staging.is_dir():
            shutil.rmtree(staging)
    print(json.dumps({"status": "exported", "outputDir": str(destination), "savedSteps": saved[0]}))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="xangle-4dgs")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "stop", "resume", "export", "status"):
        command = commands.add_parser(name)
        command.add_argument("--run-dir", type=Path, required=True)
        if name in ("train", "resume", "export"):
            command.add_argument("--runtime", default=os.environ.get("XANGLE_4DGS_RUNTIME"))
        if name == "train":
            command.add_argument("--job", type=Path, required=True)
        elif name == "export":
            command.add_argument("--output-dir", type=Path, required=True)
        elif name == "status":
            command.add_argument("--json", action="store_true", help="Print JSON (the default).")
    args = parser.parse_args(argv)
    root = args.run_dir.resolve()
    try:
        if args.command == "status":
            print(json.dumps(status(root)))
        elif args.command == "stop":
            stop(root)
        else:
            if not args.runtime:
                raise ValueError("Set --runtime or XANGLE_4DGS_RUNTIME.")
            runtime = Runtime(args.runtime)
            if args.command == "train":
                job = load_job(args.job.resolve())
                root.mkdir(parents=True)
                with run_lock(root):
                    train(root, runtime, new_job=job)
            else:
                with run_lock(root):
                    if args.command == "resume":
                        train(root, runtime)
                    else:
                        export(root, runtime, args.output_dir.resolve())
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"xangle-4dgs: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0
