# Offline GRPO RTX3090 Multi-GPU Diagnosis Plan

Date: 2026-06-29

## Current Evidence

已有 smoke 结果说明当前问题不能直接归结为“多卡显存不够”：

```text
tr2 4卡：deepspeed import 通过，进入 torchrun 后 rank 3 SIGBUS
tr2 2卡：deepspeed import 通过，进入 torchrun 后 rank 1 SIGBUS
tr2 1卡：多次在 import deepspeed 或 torchrun 后 SIGBUS
epyc3 1卡：deepspeed import、checkpoint load、1/1 train step、checkpoint save 成功，但退出阶段不干净
epyc3 4卡：未真正启动，原因是 epyc3 DRAIN
```

因此当前优先假设顺序是：

1. `tr2` 节点或该节点访问共享文件系统/conda 环境不稳定。
2. 多 rank 会放大 Python import、checkpoint 读取、NFS cache、DeepSpeed/TorchDynamo/SymPy 源码读取问题。
3. DeepSpeed/NCCL cleanup 可能存在退出阶段 hang，因为 `epyc3` 1 卡完成 step 后没有干净退出。
4. 显存不足不是首要假设，因为主要错误是 `SIGBUS`，不是 CUDA OOM。

## Diagnostic Principle

每组实验只增加一个复杂度：

```text
Python import
-> 并发文件 IO / mmap / safetensors 读取
-> NCCL all_reduce
-> trainer 1 step 不保存 checkpoint
-> trainer 1 step 保存 checkpoint
```

判定时不要只看 Slurm 状态，要同时看：

```text
logs/slurm/<job>.out
logs/slurm/<job>.err
sacct state / exit code
是否完成 1/1 step
是否产生 checkpoint
是否正常打印 finished_at
```

## New Diagnostic Entrypoints

轻量 Python 诊断脚本：

```text
scripts/train/qwenvl_train/diagnose_3090_multigpu.py
```

Slurm wrapper：

```text
scripts/train/qwenvl_train/slurm_diag_3090_multigpu_matrix.sbatch
```

支持模式：

```text
DIAG_MODE=imports  # torchrun 下每个 rank 同时 import numpy/sympy/torch/torch._dynamo/deepspeed
DIAG_MODE=io       # import 后并发读/compile/mmap Python 源码、reward、checkpoint safetensors
DIAG_MODE=nccl     # torchrun 下 init_process_group + all_reduce + barrier + destroy
DIAG_MODE=all      # imports + io + nccl
```

默认会把缓存放到节点本地 `/tmp`：

```text
TRITON_CACHE_DIR
TORCH_EXTENSIONS_DIR
PYTHONPYCACHEPREFIX
```

## Experiment Matrix

### Phase A: 节点基线

目的：确认 `tr2` 和 `epyc3` 的差异是否在训练前就存在。

命令模板：

```bash
sbatch --nodelist=<node> \
  --gres=gpu:rtx3090:1 \
  --export=ALL,NPROC_PER_NODE=1,DIAG_MODE=imports \
  scripts/train/qwenvl_train/slurm_diag_3090_multigpu_matrix.sbatch
```

建议组合：

```text
A1: tr2,   1 GPU, DIAG_MODE=imports
A2: epyc3, 1 GPU, DIAG_MODE=imports
```

判定：

```text
tr2 imports 失败，epyc3 imports 成功
=> 节点/共享环境访问问题优先，先不要继续在 tr2 上做 trainer。

两个节点 imports 都成功
=> 基础 import 不是充分条件，继续 Phase B/C。
```

### Phase B: 并发 Python/Checkpoint IO

目的：判断 `SIGBUS` 是否来自共享文件系统上的 Python 源码、pycache、checkpoint shard、reward 文件并发读取。

命令模板：

```bash
sbatch --nodelist=<node> \
  --gres=gpu:rtx3090:<n> \
  --export=ALL,NPROC_PER_NODE=<n>,DIAG_MODE=io \
  scripts/train/qwenvl_train/slurm_diag_3090_multigpu_matrix.sbatch
```

建议组合：

```text
B1: tr2,   1 GPU, DIAG_MODE=io
B2: tr2,   2 GPU, DIAG_MODE=io
B3: epyc3, 1 GPU, DIAG_MODE=io
B4: epyc3, 2 GPU, DIAG_MODE=io
B5: epyc3, 4 GPU, DIAG_MODE=io   # 等 epyc3 退出 DRAIN 后再跑
```

判定：

```text
io 在 tr2 失败但 epyc3 成功
=> tr2 的文件系统/源码读取/checkpoint mmap 路径更可疑。

io 在 1 GPU 成功但 2/4 GPU 失败
=> 并发读取放大问题，优先考虑把 conda/env、checkpoint 或 cache 复制到节点本地盘。

io 全部成功
=> 继续 Phase C，问题更可能在 NCCL/DeepSpeed/trainer。
```

### Phase C: NCCL 基础通信

目的：确认多卡通信本身是否可用，和模型加载无关。

命令模板：

```bash
sbatch --nodelist=<node> \
  --gres=gpu:rtx3090:<n> \
  --export=ALL,NPROC_PER_NODE=<n>,DIAG_MODE=nccl,NCCL_DEBUG=INFO \
  scripts/train/qwenvl_train/slurm_diag_3090_multigpu_matrix.sbatch
```

建议组合：

```text
C1: epyc3, 2 GPU, DIAG_MODE=nccl
C2: epyc3, 4 GPU, DIAG_MODE=nccl
C3: tr2,   2 GPU, DIAG_MODE=nccl
C4: tr2,   4 GPU, DIAG_MODE=nccl
```

判定：

```text
NCCL 失败
=> 多卡通信/驱动/NCCL/拓扑问题，不应继续完整 trainer。

NCCL 成功但 trainer 失败
=> NCCL 基础通信可用，问题转向 DeepSpeed 初始化、模型加载、dataset 或 cleanup。
```

### Phase D: Trainer 1 Step 不保存 checkpoint

目的：判断多卡训练 step 是否能完成，先排除 checkpoint save 和 final save 影响。

命令模板：

```bash
sbatch --nodelist=<node> \
  --gres=gpu:rtx3090:<n> \
  --export=ALL,NPROC_PER_NODE=<n>,RUN_NAME=InternVLA-N1-VLNPE-OfflinePref-<n>x3090-<node>-NoSave-Smoke,OUTPUT_DIR=checkpoints/InternVLA-N1-VLNPE-OfflinePref-<n>x3090-<node>-NoSave-Smoke,TRITON_CACHE_DIR=/tmp/ycl_triton_cache_<n>x_<node>,TORCH_EXTENSIONS_DIR=/tmp/ycl_torch_extensions_<n>x_<node>,PYTHONPYCACHEPREFIX=/tmp/ycl_pycache_<n>x_<node>,PYTHONFAULTHANDLER=1,NCCL_DEBUG=INFO,SAVE_STEPS=999,INTERNVLA_SKIP_FINAL_SAVE=true \
  scripts/train/qwenvl_train/slurm_train_dual_system_r2r_offline_pref_4x3090_smoke.sbatch
```

建议组合：

```text
D1: epyc3, 2 GPU, trainer 1 step no-save
D2: epyc3, 4 GPU, trainer 1 step no-save
D3: tr2,   2 GPU, trainer 1 step no-save
D4: tr2,   4 GPU, trainer 1 step no-save
```

判定：

```text
trainer no-save 完成 step 且正常退出
=> 多卡训练本身可行，之前失败更可能来自 checkpoint save/final save/节点偶发。

trainer no-save 完成 step 但不退出
=> cleanup/barrier/DeepSpeed/NCCL destroy 问题。

trainer no-save 进入训练前失败
=> 仍是 import/checkpoint/dataset/model-load 问题。

trainer no-save 训练中失败
=> DeepSpeed 分布式训练路径或显存/CPU 内存问题，再查 stderr 中是否 OOM。
```

### Phase E: Trainer 1 Step 保存 checkpoint

目的：只在 Phase D 成功后测试 checkpoint 保存。

命令模板：

```bash
sbatch --nodelist=<node> \
  --gres=gpu:rtx3090:<n> \
  --export=ALL,NPROC_PER_NODE=<n>,RUN_NAME=InternVLA-N1-VLNPE-OfflinePref-<n>x3090-<node>-Save-Smoke,OUTPUT_DIR=checkpoints/InternVLA-N1-VLNPE-OfflinePref-<n>x3090-<node>-Save-Smoke,TRITON_CACHE_DIR=/tmp/ycl_triton_cache_<n>x_<node>_save,TORCH_EXTENSIONS_DIR=/tmp/ycl_torch_extensions_<n>x_<node>_save,PYTHONPYCACHEPREFIX=/tmp/ycl_pycache_<n>x_<node>_save,PYTHONFAULTHANDLER=1,NCCL_DEBUG=INFO,SAVE_STEPS=1,INTERNVLA_SKIP_FINAL_SAVE=true \
  scripts/train/qwenvl_train/slurm_train_dual_system_r2r_offline_pref_4x3090_smoke.sbatch
```

判定：

```text
no-save 成功，save 失败或 hang
=> checkpoint save / shared filesystem 写入 / ZeRO state 保存是主要问题。

no-save 和 save 都成功
=> 可以启动更长训练 smoke，例如 MAX_STEPS=10。
```

## Recommended Execution Order

优先顺序：

```text
1. 等 epyc3 退出 DRAIN 后，先跑 epyc3 2卡 imports/io/nccl。
2. epyc3 2卡轻量诊断都通过后，跑 epyc3 2卡 trainer no-save。
3. epyc3 2卡 no-save 通过后，跑 epyc3 4卡 imports/io/nccl。
4. epyc3 4卡轻量诊断通过后，跑 epyc3 4卡 trainer no-save。
5. 只有 no-save 成功后，才测试 save checkpoint。
6. tr2 只作为对照节点，不建议继续作为真实训练节点，除非管理员修复其 SIGBUS/import 行为。
```

## Result Interpretation Summary

```text
tr2 失败，epyc3 成功
=> 节点问题为主。

imports/io 失败
=> Python/conda/shared filesystem/checkpoint read 问题为主。

imports/io 成功，nccl 失败
=> NCCL/驱动/拓扑问题为主。

imports/io/nccl 成功，trainer 失败
=> DeepSpeed trainer/model/dataset/ZeRO 初始化问题为主。

trainer no-save 成功，save 失败
=> checkpoint 写入/final save/共享文件系统写入问题为主。

trainer step 成功但不退出
=> cleanup/barrier/process group destroy 问题为主。
```

## Current Blocker

截至当前记录，`epyc3` 处于：

```text
State=MIXED+DRAIN
Reason=Kill task failed
```

因此 4 卡 epyc3 训练 smoke 还没有被真正验证。等待 `epyc3` 恢复后，应优先从 2 卡轻量诊断开始，而不是直接重新跑完整 4 卡 trainer。
