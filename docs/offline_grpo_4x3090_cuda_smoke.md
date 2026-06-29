# Offline GRPO 4x RTX3090 Smoke Test

Date: 2026-06-29

## Goal

After committing the current offline GRPO workflow, test whether the experiment can avoid the `rtx3090-large` partition and run a small 4-GPU smoke test on the normal `rtx3090` partition with `MAX_STEPS=1`.

## Pre-test Commit

The current offline GRPO implementation was committed before the smoke test:

```bash
git commit -m "Add offline GRPO reward training workflow"
```

Commit:

```text
a371036 Add offline GRPO reward training workflow
```

## Environment Fix

The previous normal `rtx3090` training attempt failed because DeepSpeed could not find CUDA:

```text
CUDA_HOME is None
```

The conda environment now has CUDA 12.8 nvcc installed:

```text
/srv/shared/home/ycl/miniconda3/envs/env_isaaclab/bin/nvcc
Cuda compilation tools, release 12.8, V12.8.93
```

The smoke script exports:

```text
CUDA_HOME=/srv/shared/home/ycl/miniconda3/envs/env_isaaclab
CUDA_PATH=/srv/shared/home/ycl/miniconda3/envs/env_isaaclab
```

Several Slurm scripts were also adjusted so `set -u` is enabled only after `conda activate`. This avoids the conda CUDA nvcc activation hook failing on an unset `NVCC_PREPEND_FLAGS`.

## Smoke Script

Script:

```text
scripts/train/qwenvl_train/slurm_train_dual_system_r2r_offline_pref_4x3090_smoke.sbatch
```

Main settings:

```text
partition=rtx3090
nodelist=tr2
gres=gpu:rtx3090:4
MAX_STEPS=1
VLN_DATASET_USE=vln_pe_r2r_offline%1
OFFLINE_REWARD_PATH=logs/offline_grpo/r2r_offline_grpo_vln_pe_full_1gpu/rewards.jsonl
```

## Attempts

### Job 5051

Result:

```text
FAILED, elapsed 00:00:09, node tr2
```

Reason:

```text
CUDA_HOME=/usr/local/cuda
nvcc release 13.0
import deepspeed -> Bus error
```

This showed that relying on system CUDA 13.0 was not usable for the current PyTorch CUDA 12.8 environment.

### Job 5052

Result:

```text
FAILED, elapsed 00:00:02, node tr2
```

Reason:

```text
NVCC_PREPEND_FLAGS: unbound variable
```

This came from the conda CUDA nvcc activation hook under `set -u`. The Slurm scripts were patched to activate conda before re-enabling `set -u`.

### Job 5053

Result:

```text
FAILED, elapsed 00:00:56, node tr2
```

Positive checks:

```text
CUDA_HOME=/srv/shared/home/ycl/miniconda3/envs/env_isaaclab
nvcc release 12.8, V12.8.93
[check] torch CUDA_HOME=/srv/shared/home/ycl/miniconda3/envs/env_isaaclab
[check] deepspeed import ok: 0.19.2
```

Failure after entering distributed training:

```text
rank 3 local_rank 3 exitcode -7
Signal 7 (SIGBUS)
```

The output directory was only 4 KB:

```text
checkpoints/InternVLA-N1-VLNPE-OfflinePref-4x3090-CUDA-Smoke
```

No useful checkpoint was produced.

## Conclusion

The original CUDA_HOME/DeepSpeed problem on normal `rtx3090` is fixed. The environment can now find CUDA 12.8 from the conda env, and DeepSpeed imports correctly.

However, the 4-card `MAX_STEPS=1` training smoke has not completed. The current blocker is a new `SIGBUS` crash on rank 3 after model/checkpoint loading starts. Therefore, the experiment can avoid waiting for `rtx3090-large` in principle, but the normal `rtx3090` 4-GPU path still needs one more round of debugging before it can be used as a reliable training path.

Recommended next checks:

1. Try a smaller normal `rtx3090` smoke with 1 or 2 GPUs to isolate whether `SIGBUS` is caused by 4 parallel ranks loading the checkpoint.
2. Put DeepSpeed/Triton temporary cache on a node-local path instead of NFS.
3. If the crash persists, test whether copying the checkpoint to node-local storage before launch avoids shared-filesystem mmap/SIGBUS issues.
