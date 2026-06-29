# Offline GRPO large8 训练任务 5050 诊断

## 结论

任务 `5050` 没有真正失败，也没有进入训练代码。它仍处于 Slurm `PENDING` 状态。

阻塞原因是 `rtx3090-large` 分区唯一节点 `epyc1` 当前处于 `IDLE+DRAIN`，Slurm 给出的原因是：

```text
Kill task failed
```

因此当前没有生成训练 stdout/stderr 日志，也没有 Python traceback。

## 已确认信息

作业状态：

```text
JobId=5050
JobName=internvla_r2r_offline_pref_l8
JobState=PENDING
Reason=Nodes_required_for_job_are_DOWN,_DRAINED_or_reserved_for_jobs_in_higher_priority_partitions
```

资源请求：

```text
Partition=rtx3090-large
ReqTRES=cpu=16,mem=300G,node=1,billing=16,gres/gpu=8
TresPerNode=gres/gpu:rtx3090:8
```

节点状态：

```text
NodeName=epyc1
State=IDLE+DRAIN
Gres=gpu:rtx3090:8
Reason=Kill task failed
```

## 接口状态

训练脚本接口已在提交前验证过：

- 全量 reward 文件存在，且为 `13104` 行：

```bash
logs/offline_grpo/r2r_offline_grpo_vln_pe_full_1gpu/rewards.jsonl
```

- loader 能读取该文件，并生成 `traj_reward_weights`：

```text
traj_reward_weights=(1, 2)
```

因此当前问题不是 reward 文件或训练接口不对齐，而是 large8 节点没有被 Slurm 调度。

## 后续处理

保持 `5050` 等待即可；等管理员解除 `epyc1` 的 `DRAIN` 状态后，任务应自动开始运行。

查看状态：

```bash
squeue -j 5050
scontrol show node epyc1 | grep -E 'State=|Reason='
```

如果需要改用普通 `rtx3090` 分区重新提交，需要另建或修改脚本；本次没有做该修改。
