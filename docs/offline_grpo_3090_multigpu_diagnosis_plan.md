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

## Current Entrypoint

轻量诊断脚本已移除。当前只保留直接训练 smoke 入口：

```text
scripts/train/qwenvl_train/slurm_train_dual_system_r2r_offline_pref_4x3090_smoke.sbatch
```

所有多卡验证都用这个脚本，通过 `sbatch --export` 覆盖节点、本地 cache、卡数、保存策略。

## Experiment Matrix

### Phase A: Trainer 1 Step 不保存 checkpoint

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
A1: epyc3, 2 GPU, trainer 1 step no-save
A2: epyc3, 4 GPU, trainer 1 step no-save
A3: tr2,   2 GPU, trainer 1 step no-save
A4: tr2,   4 GPU, trainer 1 step no-save
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

### Phase B: Trainer 1 Step 保存 checkpoint

目的：只在 Phase A 成功后测试 checkpoint 保存。

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
1. 等 epyc3 退出 DRAIN 后，先跑 epyc3 2卡 trainer no-save。
2. epyc3 2卡 no-save 通过后，跑 epyc3 4卡 trainer no-save。
3. 只有 no-save 成功后，才测试 save checkpoint。
4. tr2 只作为对照节点，不建议继续作为真实训练节点，除非管理员修复其 SIGBUS/import 行为。
```

## Result Interpretation Summary

```text
tr2 失败，epyc3 成功
=> 节点问题为主。

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

因此 4 卡 epyc3 训练 smoke 还没有被真正验证。等待 `epyc3` 恢复后，应优先从 2 卡 trainer no-save 开始，而不是直接重新跑完整 4 卡保存 checkpoint 的训练。
