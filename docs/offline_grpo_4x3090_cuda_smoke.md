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

## 2-GPU Follow-up

Job:

```text
5054
```

Command shape:

```bash
sbatch --gres=gpu:rtx3090:2 \
  --export=ALL,NPROC_PER_NODE=2,RUN_NAME=InternVLA-N1-VLNPE-OfflinePref-2x3090-CUDA-Smoke,OUTPUT_DIR=checkpoints/InternVLA-N1-VLNPE-OfflinePref-2x3090-CUDA-Smoke \
  scripts/train/qwenvl_train/slurm_train_dual_system_r2r_offline_pref_4x3090_smoke.sbatch
```

Result:

```text
FAILED, elapsed 00:00:34, node tr2
CUDA_VISIBLE_DEVICES=0,1
nproc_per_node=2
CUDA_HOME=/srv/shared/home/ycl/miniconda3/envs/env_isaaclab
[check] deepspeed import ok: 0.19.2
Loaded 546 offline reward weights.
sampling 16 examples from dataset vln_pe_r2r_offline%1
rank 1 local_rank 1 exitcode -7
Signal 7 (SIGBUS)
```

The 2-GPU run reproduced the same `SIGBUS` pattern as the 4-GPU run, so the failure is not specific to 4 ranks loading the checkpoint at the same time. The current evidence points more broadly to multi-process distributed startup/model loading/runtime state on this normal `rtx3090` node, or to shared filesystem/cache interaction under distributed launch.

The 2-GPU smoke output directory was also only 4 KB:

```text
checkpoints/InternVLA-N1-VLNPE-OfflinePref-2x3090-CUDA-Smoke
```

No useful checkpoint was produced.

Recommended next checks:

1. Try a 1-GPU normal `rtx3090` smoke to separate single-process training issues from distributed multi-process issues.
2. Put DeepSpeed/Triton temporary cache on a node-local path instead of NFS.
3. If the crash persists, test whether copying the checkpoint to node-local storage before launch avoids shared-filesystem mmap/SIGBUS issues.

## 1-GPU Follow-up

### Job 5055: tr2, default cache

Result:

```text
FAILED, elapsed 00:00:07, node tr2
CUDA_VISIBLE_DEVICES=0
nproc_per_node=1
CUDA_HOME=/srv/shared/home/ycl/miniconda3/envs/env_isaaclab
```

Failure:

```text
Bus error during preflight import deepspeed
```

This failed before entering `torchrun`.

### Job 5056: tr2, local Triton/Torch cache

Result:

```text
FAILED, elapsed 00:00:18, node tr2
CUDA_VISIBLE_DEVICES=0
nproc_per_node=1
TRITON_CACHE_DIR=/tmp/ycl_triton_cache_5056
TORCH_EXTENSIONS_DIR=/tmp/ycl_torch_extensions_5056
[check] deepspeed import ok: 0.19.2
rank 0 local_rank 0 exitcode -7
Signal 7 (SIGBUS)
```

Local Triton/Torch cache fixed the preflight `deepspeed import` in this attempt, but the trainer still crashed with `SIGBUS` after `torchrun` started.

### Job 5057: tr2, faulthandler

Result:

```text
FAILED, elapsed 00:00:08, node tr2
```

`PYTHONFAULTHANDLER=1` showed the `SIGBUS` happened during Python import of DeepSpeed/TorchDynamo/SymPy bytecode/source:

```text
Fatal Python error: Bus error
...
importlib._bootstrap_external._compile_bytecode
sympy/printing/__init__.py
torch/_dynamo
deepspeed/runtime/compiler.py
```

### Job 5058: tr2, local pycache

Result:

```text
FAILED, elapsed 00:00:12, node tr2
PYTHONPYCACHEPREFIX=/tmp/ycl_pycache_5058
```

Failure was still during `import deepspeed`, but the stack moved to source compilation:

```text
Fatal Python error: Bus error
...
importlib._bootstrap_external.source_to_code
torch/onnx/symbolic_caffe2.py
torch/_dynamo
deepspeed/comm
```

This suggests the `tr2` failures are not only from Python writing cache files. They are consistent with unstable source/bytecode reads from the shared environment or node-specific filesystem behavior.

### Job 5059: epyc3, local cache and local pycache

Result:

```text
CANCELLED after elapsed 00:06:21, node epyc3
CUDA_VISIBLE_DEVICES=0
nproc_per_node=1
TRITON_CACHE_DIR=/tmp/ycl_triton_cache_5059
TORCH_EXTENSIONS_DIR=/tmp/ycl_torch_extensions_5059
PYTHONPYCACHEPREFIX=/tmp/ycl_pycache_5059
```

Positive checks:

```text
[check] deepspeed import ok: 0.19.2
Loaded checkpoint shards
Completed 1/1 train step
{'loss': 1.0573, 'grad_norm': 69.89163970947266, 'learning_rate': 0.0, 'epoch': 0.06}
```

Output:

```text
checkpoints/InternVLA-N1-VLNPE-OfflinePref-1x3090-Epyc3-Smoke/checkpoint-1
16G total
model-00001-of-00004.safetensors
model-00002-of-00004.safetensors
model-00003-of-00004.safetensors
model-00004-of-00004.safetensors
```

The training step and checkpoint save succeeded on `epyc3`, but the process did not exit cleanly after several minutes of no new output, so it was manually cancelled to free the GPU. This means the normal `rtx3090` path is usable for at least a 1-GPU training step on `epyc3`, while `tr2` appears unreliable for this Python/DeepSpeed import/runtime path.

Updated recommendation:

1. Avoid `tr2` for offline GRPO training until its SIGBUS/import behavior is understood.
2. Prefer `epyc3` or another non-`tr2` normal `rtx3090` node for the next smoke.
3. Keep `TRITON_CACHE_DIR`, `TORCH_EXTENSIONS_DIR`, and `PYTHONPYCACHEPREFIX` on node-local `/tmp`.
4. Disable final save for pure smoke tests, or add an explicit timeout around cleanup, because `epyc3` completed the train step but hung during final exit.
