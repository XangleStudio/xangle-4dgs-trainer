import os
from pathlib import Path


class StopAfterCheckpoint(Exception):
    pass


def main():
    from trigs.cli.train_from_colmap import main as train
    from trigs.core.xangle_checkpoint import TrainingCheckpoint

    request = Path(os.environ["XANGLE_TRIGS_STOP_REQUEST"])
    original_due = TrainingCheckpoint.due
    original_write = TrainingCheckpoint.write
    stop_step = None

    def due(checkpoint, completed, target):
        nonlocal stop_step
        stop_step = completed if completed < target and request.is_file() else None
        return stop_step is not None or original_due(checkpoint, completed, target)

    def write(checkpoint, completed, *args, **kwargs):
        original_write(checkpoint, completed, *args, **kwargs)
        if completed == stop_step:
            raise StopAfterCheckpoint

    TrainingCheckpoint.due = due
    TrainingCheckpoint.write = write
    try:
        train()
    except StopAfterCheckpoint:
        print(f"[XANGLE_STOPPED] step={stop_step}", flush=True)


if __name__ == "__main__":
    main()
