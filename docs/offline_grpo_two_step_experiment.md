# Offline GRPO 实验步骤

## 1. 计算 reward

先用当前 DualVLN checkpoint 让 System1 对 `vln_pe_r2r_offline` 全量样本生成候选轨迹，并计算 offline GRPO reward：

```bash
RUN_NAME=r2r_offline_grpo_vln_pe_full_1gpu \
VLN_DATASET_USE=vln_pe_r2r_offline \
NUM_SAMPLES=1638 \
NUM_CANDIDATES=8 \
NUM_INFERENCE_STEPS=10 \
sbatch scripts/train/qwenvl_train/slurm_generate_offline_rewards_3090.sbatch
```

输出文件：

```bash
logs/offline_grpo/r2r_offline_grpo_vln_pe_full_1gpu/candidates.jsonl
logs/offline_grpo/r2r_offline_grpo_vln_pe_full_1gpu/rewards.jsonl
```

其中训练阶段主要使用 `rewards.jsonl`，里面包含每条候选轨迹的 `loss_weight`。

## 2. 开始训练

等 `rewards.jsonl` 生成完成后，用它作为 `OFFLINE_REWARD_PATH` 启动 System1 offline preference 后训练：

```bash
OFFLINE_REWARD_PATH=logs/offline_grpo/r2r_offline_grpo_vln_pe_full_1gpu/rewards.jsonl \
VLN_DATASET_USE=vln_pe_r2r_offline \
RUN_NAME=InternVLA-N1-VLNPE-OfflinePref-Full \
OUTPUT_DIR=checkpoints/InternVLA-N1-VLNPE-OfflinePref-Full \
sbatch scripts/train/qwenvl_train/slurm_train_dual_system_r2r_offline_pref_3090.sbatch
```

训练输入 checkpoint：

```bash
checkpoints/InternVLA-N1-DualVLN
```

训练输出 checkpoint：

```bash
checkpoints/InternVLA-N1-VLNPE-OfflinePref-Full
```
