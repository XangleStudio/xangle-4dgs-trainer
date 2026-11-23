# Xangle 4DGS Trainer

Open4DGS provides the matching, training, and PLY export pipeline. This companion adds a managed 4DGS job interface around it, with matching-data preparation, durable optimizer checkpoints, stop and resume from saved state, run status, and validated PLY-sequence export.

## Install

Run `python -m pip install .` from this repository to install the `xangle-4dgs` command. This installs the CLI, not the CUDA runtime, Open4DGS, or model weights.

Set `XANGLE_4DGS_RUNTIME` to the runtime directory, or pass `--runtime` on `train`, `resume`, and `export`. The runtime must contain `env\Scripts\python.exe`, `open4dgs-core`, and `trigs`. Use [Core revision `3cac7ba`](https://github.com/ASU-NEM-Labs/open4dgs-core/tree/3cac7ba62d930e3ce56cb532bb60870e4ebf6db5) and [TRiGS revision `ed7a84a`](https://github.com/ASU-NEM-Labs/open4dgs-trigs/tree/ed7a84aff1aeccd18dfb48d2806e521255ccf9ef). Follow the [TRiGS runtime setup](https://github.com/ASU-NEM-Labs/open4dgs-trigs/blob/ed7a84aff1aeccd18dfb48d2806e521255ccf9ef/SETUP.md) and apply the [checkpoint patch](patches/trigs-checkpoint.patch) to the TRiGS checkout before installing it into the runtime environment:

```powershell
git -C "$env:XANGLE_4DGS_RUNTIME\trigs" apply --check "$PWD\patches\trigs-checkpoint.patch"
git -C "$env:XANGLE_4DGS_RUNTIME\trigs" apply "$PWD\patches\trigs-checkpoint.patch"
```

For matching, follow [Core's preparation setup](https://github.com/ASU-NEM-Labs/open4dgs-core/blob/3cac7ba62d930e3ce56cb532bb60870e4ebf6db5/prep/SETUP.md). The CLI looks for RoMa v2 and MEMFOF packages in `matching-v2\packages`, the RoMa/DINOv3 Torch Hub cache in `matching-v2\torch`, and the [pinned MEMFOF checkpoint](https://huggingface.co/egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH/tree/6c6c9aa3ad64f93aee8efbc2f7a6e4535814ee96) in `matching-v2\memfof\6c6c9aa3ad64f93aee8efbc2f7a6e4535814ee96`. Alternatively, pass a compatible `preparedNpz` in the job to skip matching.

## Train, stop, resume, and export

Save a job file with paths relative to that file or absolute paths:

```json
{
  "version": 1,
  "captureDir": "capture",
  "start": 325,
  "end": 375,
  "fps": 120,
  "steps": 5000,
  "dataFactor": 2,
  "batchCameras": 4
}
```

The capture directory contains synchronized numbered images in `images\<camera>` and a compatible COLMAP model in `sparse\0`. Use an already normalized capture with image names and camera IDs matching its calibration; the CLI does not ingest native RealityScan projects or synchronize video. The start and end are inclusive. Optional `inputStart` and `inputEnd` identify a staged range wider than the training range; `stride` defaults to 2. `steps` is the total optimizer-step target.

```powershell
xangle-4dgs train --job .\capture-job.json --run-dir D:\4dgs\run-001
xangle-4dgs status --run-dir D:\4dgs\run-001 --json
xangle-4dgs stop --run-dir D:\4dgs\run-001
xangle-4dgs resume --run-dir D:\4dgs\run-001
xangle-4dgs export --run-dir D:\4dgs\run-001 --output-dir D:\4dgs\ply-001
```

Run `stop` from another terminal while the optimizer is active. It requests a save after a completed optimizer step. Wait for `status` to report `stopped` before resuming or exporting. If training reaches its target before the request can take effect, it finishes normally. A resume keeps the original target and requires the same capture inputs and compatible runtime. Use a new run directory to start a new job.

The run directory holds the log, checkpoints, and any generated matching data. Put it on a drive with enough free space. Export validates the frame count and FPS before publishing to a new directory. It does not activate the export in either Xangle app.

## License and credits

The companion is licensed under [AGPL-3.0](LICENSE). It uses the [MIT-licensed Open4DGS-Core](https://github.com/ASU-NEM-Labs/open4dgs-core) and [AGPL-licensed Open4DGS-Trigs](https://github.com/ASU-NEM-Labs/open4dgs-trigs). See the [TRiGS third-party notices](https://github.com/ASU-NEM-Labs/open4dgs-trigs/blob/ed7a84aff1aeccd18dfb48d2806e521255ccf9ef/THIRD_PARTY_NOTICES.md) for incorporated code. Model weights and runtime dependencies retain their own licenses. This repository does not contain model weights, training data, or a bundled runtime.
