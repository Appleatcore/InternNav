# Offline GRPO 后训练实现日志

## 已完成的代码改动

1. 新增 `scripts/dataset_converters/prepare_internvla_traj_data.py`
   - 用于解压 `traj_data/vln_ce/r2r/*.tar.gz`。
   - 支持解压内层 `observation.images.rgb.tar.gz`。
   - 将 `scene_id/group_id/{data,meta,videos}` 结构整理为默认训练路径下的软链接：`traj_data/r2r/scene_id__group_id`。

2. 修改 `internnav/dataset/internvla_n1_lerobot_dataset.py`
   - 支持 mini 数据的嵌套 LeRobot layout。
   - 支持 `observation.images.rgb/000.jpg` 这类 RGB 帧路径 fallback。
   - depth 缺失时使用零深度图，避免 loader smoke test 直接失败。
   - `torchcodec` 和 `decord` 改为可选依赖，只有通用视频分支实际调用时才报错。
   - collator 支持可选 `traj_reward_weights`，用于后续离线 GRPO / preference loss 加权。
   - 新增 `sample_id` 规则：`unit_id:epXXXXXX:framesXXXXXX-XXXXXX`。
   - 支持从 offline reward JSONL 加载 `loss_weight`，并映射为每个 trajectory frame 的 `traj_reward_weights`。

3. 修改 `internnav/model/basemodel/internvla_n1/internvla_n1.py`
   - `forward` 新增可选输入 `traj_reward_weights`。
   - 在 System1 trajectory diffusion loss 中支持按 reward/advantage 权重加权。
   - 默认不传该字段时，不改变原始训练行为。

4. 新增 `scripts/train/qwenvl_train/build_offline_grpo_rewards.py`
   - 输入同一状态下多条候选轨迹和 GT 轨迹。
   - 计算 ADE、FDE、终点误差、平滑度 reward。
   - 输出 group-relative advantage 和 `loss_weight`。

5. 新增 `scripts/train/qwenvl_train/generate_offline_traj_candidates.py`
   - 加载 `InternVLA-N1-DualVLN` checkpoint 和 `NavPixelGoalDataset`。
   - 复用模型已有 `generate_latents()` 与 `generate_traj()`，让 System1 生成多条候选轨迹。
   - 输出 `build_offline_grpo_rewards.py` 可直接读取的 JSONL：`sample_id/candidate/gt/goal`。
   - 在 inference-only 场景屏蔽 Transformers 对 deepspeed 的自动导入，避免普通 `rtx3090` 节点因 `CUDA_HOME` 缺失失败。

6. 新增 `scripts/train/qwenvl_train/smoke_nav_pixel_goal_dataset.py`
   - 不加载完整模型，只加载 tokenizer/processor 和 `NavPixelGoalDataset`。
   - 用于检查 RGB、pose、goal、trajectory query 是否能被 dataset 正常构造。
   - 支持 `--offline-reward-path`、`--offline-reward-reduce`、`--find-sample-id`，用于验证 reward 权重是否进入 batch。

7. 新增 3090 Slurm 脚本
   - `scripts/train/qwenvl_train/slurm_smoke_r2r_dataset_3090.sbatch`
   - `scripts/train/qwenvl_train/slurm_generate_offline_rewards_3090.sbatch`
   - `scripts/train/qwenvl_train/slurm_train_dual_system_r2r_offline_pref_3090.sbatch`
   - 候选/reward 生成脚本默认使用普通 `rtx3090` 单卡。
   - 训练脚本默认使用 `vln_pe_r2r_offline%10` 和 `rtx3090-large`，并设置 `INTERNVLA_SKIP_FINAL_SAVE=true`，用于短步数 smoke。
   - 支持通过 `OFFLINE_REWARD_PATH` 和 `OFFLINE_REWARD_REDUCE` 接入离线 reward 文件。

8. 修改 `.gitignore`
   - 新增 `traj_data/`，避免将解压后的轨迹数据和软链接误提交到 git。

9. 修改 `internnav/trainer/internvla_n1_trainer.py`
   - 根据 checkpoint `config.json` 中的 `model_type=internvla_n1` 加载 `InternVLAN1ForCausalLM`，避免 DualVLN checkpoint 被误加载成 Qwen2VL。
   - 支持 `INTERNVLA_ATTN_IMPLEMENTATION=sdpa`，避免环境缺少 `flash_attn` 时失败。
   - `tabulate` 改为可选依赖。
   - 支持 `INTERNVLA_SKIP_FINAL_SAVE=true`，短训练 smoke 可跳过最终完整模型保存，只保留 step checkpoint。

10. 修改 `internnav/trainer/qwenvl_base.py`
   - `flash_attn` 改为可选懒依赖，只有 `data_flatten=True` 替换 attention 时才强制要求。

## 已运行的验证

1. Python 语法检查通过：

```bash
python -m py_compile \
  scripts/dataset_converters/prepare_internvla_traj_data.py \
  scripts/train/qwenvl_train/build_offline_grpo_rewards.py \
  scripts/train/qwenvl_train/generate_offline_traj_candidates.py \
  scripts/train/qwenvl_train/smoke_nav_pixel_goal_dataset.py \
  internnav/dataset/internvla_n1_lerobot_dataset.py \
  internnav/model/basemodel/internvla_n1/internvla_n1.py
```

2. Slurm 脚本语法检查通过：

```bash
bash -n \
  scripts/train/qwenvl_train/slurm_smoke_r2r_dataset_3090.sbatch \
  scripts/train/qwenvl_train/slurm_generate_offline_rewards_3090.sbatch \
  scripts/train/qwenvl_train/slurm_train_dual_system_r2r_offline_pref_3090.sbatch
```

3. 离线 reward 脚本最小样例通过，能够输出 group-relative advantage 和 loss weight。

4. 数据准备脚本已实际运行 1 个 scene：

```bash
python -u scripts/dataset_converters/prepare_internvla_traj_data.py \
  --src-dir traj_data/vln_ce/r2r \
  --extract-root traj_data/vln_ce/r2r_extracted \
  --flat-root traj_data/r2r \
  --manifest traj_data/vln_ce/r2r_units.txt \
  --limit-scenes 1
```

结果：

- 解压 `17DRP5sb8fy`。
- 解压 25 个内层 RGB 视频包。
- 创建 25 个 `traj_data/r2r/17DRP5sb8fy__*` 软链接。

5. 已给 `env_isaaclab` 安装 loader smoke test 所需依赖：

```bash
pip install pyarrow decord torchcodec
```

其中代理 `10.12.120.178:7890` 和 `10.12.120.119:7890` 都超时，最终使用清华 PyPI 镜像安装成功。

6. 已给 `env_isaaclab` 安装 `deepspeed`：

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple deepspeed
```

7. `vln_pe` offline trajectory smoke test 已通过：

```bash
python -u scripts/train/qwenvl_train/smoke_nav_pixel_goal_dataset.py \
  --model-name-or-path checkpoints/InternVLA-N1-DualVLN \
  --vln-dataset-use 'vln_pe_r2r_offline%10' \
  --num-samples 1
```

结果：

- dataset 长度：163。
- batch 中包含 `traj_images`、`traj_depths`、`traj_poses`。
- `traj_poses` shape 为 `(1, 2, 32, 3)`。

8. 3090 短训练 smoke 已跑通核心训练 step：

```bash
MAX_STEPS=1 SAVE_STEPS=1 \
RUN_NAME=InternVLA-N1-VLNPE-OfflinePref-1step \
OUTPUT_DIR=checkpoints/InternVLA-N1-VLNPE-OfflinePref-1step \
sbatch scripts/train/qwenvl_train/slurm_train_dual_system_r2r_offline_pref_3090.sbatch
```

有效运行 job：

- `5020`：在 `rtx3090-large/epyc1` 上使用 4 张 RTX 3090。
- 已完成 1 个训练 step。
- 日志记录：`loss=0.9139`，`grad_norm=29.4441`。
- 已写出 `checkpoints/InternVLA-N1-VLNPE-OfflinePref-1step/checkpoint-1/` 下 4 个 model shard。
- 因最终完整模型保存耗时过久，已手动取消该 smoke job；核心 forward/backward/checkpoint 已验证。

9. 其他 Slurm 尝试：

- `5016`：失败于 `flash_attn` 顶层 import，已通过懒依赖修复。
- `5017`：失败于 DualVLN checkpoint 类型误判，已通过读取 `config.json` 修复。
- `5018`：失败于缺少 `tabulate`，已通过可选依赖修复。
- `5019`：失败于 bf16/float32 dtype mismatch，已通过将 `images_dp_norm` cast 到 `rgb_model` dtype 修复。
- `5021`：落到普通 `rtx3090/epyc3`，失败于 `CUDA_HOME` 缺失导致 `deepspeed` import 错误。
- `5023`：固定 `rtx3090-large` 后提交，但节点当时不可用，任务 pending 后已取消。

10. reward JSONL 到 batch 的映射已验证：

先构造两条同一 `sample_id` 的候选轨迹并计算 reward：

```bash
python scripts/train/qwenvl_train/build_offline_grpo_rewards.py \
  --input tmp/offline_grpo_smoke/candidates.jsonl \
  --output tmp/offline_grpo_smoke/rewards.jsonl
```

再用指定 `sample_id` 跑 loader smoke：

```bash
python -u scripts/train/qwenvl_train/smoke_nav_pixel_goal_dataset.py \
  --model-name-or-path checkpoints/InternVLA-N1-DualVLN \
  --vln-dataset-use 'vln_pe_r2r_offline' \
  --num-samples 1 \
  --find-sample-id '17DRP5sb8fy__6380:ep000000:frames000032-000037' \
  --offline-reward-path tmp/offline_grpo_smoke/rewards.jsonl \
  --offline-reward-reduce max
```

结果：

- 成功加载 1 条 offline reward weight。
- 命中样本 `17DRP5sb8fy__6380:ep000000:frames000032-000037`。
- sample 中 `traj_reward_weights` 为 `[2.0, 2.0]`。
- batch 中包含 `traj_reward_weights`，shape 为 `(1, 2)`。

11. System1 候选轨迹生成到 reward 构造的 Slurm smoke 已通过：

```bash
NUM_SAMPLES=1 NUM_CANDIDATES=1 NUM_INFERENCE_STEPS=2 \
RUN_NAME=r2r_offline_grpo_candidate_smoke \
sbatch scripts/train/qwenvl_train/slurm_generate_offline_rewards_3090.sbatch
```

有效运行 job：

- `5028`：在普通 `rtx3090/epyc3` 上完成。
- Slurm 状态：`COMPLETED`，退出码 `0:0`，耗时 `00:03:51`。
- 写出 1 行 `logs/offline_grpo/r2r_offline_grpo_candidate_smoke/candidates.jsonl`。
- 写出 1 行 `logs/offline_grpo/r2r_offline_grpo_candidate_smoke/rewards.jsonl`。
- reward 记录包含 `ade/fde/reward/loss_weight`，示例 `loss_weight=1.0030500814096612`。
- 候选轨迹与 GT 轨迹长度均为 32。

12. `5028` 生成的真实 reward 文件已验证可以被训练 loader 消费：

```bash
python -u scripts/train/qwenvl_train/smoke_nav_pixel_goal_dataset.py \
  --model-name-or-path checkpoints/InternVLA-N1-DualVLN \
  --vln-dataset-use 'vln_pe_r2r_offline%10' \
  --num-samples 1 \
  --find-sample-id '17DRP5sb8fy__1945:ep000000:frames000116-000121' \
  --offline-reward-path logs/offline_grpo/r2r_offline_grpo_candidate_smoke/rewards.jsonl \
  --offline-reward-reduce max
```

结果：

- 成功加载 1 条 offline reward weight。
- 命中样本 `17DRP5sb8fy__1945:ep000000:frames000116-000121`。
- sample 中 `traj_reward_weights` 为 `[1.0030500888824463, 1.0030500888824463]`。
- batch 中包含 `traj_reward_weights`，shape 为 `(1, 2)`。

## 当前验证结论

当前 `vln_ce/r2r` mini 数据还不能跑 System1 后训练。

原因是已抽查的 parquet 只有以下列：

```text
action
timestamp
frame_index
episode_index
index
task_index
```

缺少当前 `NavPixelGoalDataset` 构造 trajectory query 所需字段：

```text
pose.<setting>
goal.<setting>
relative_goal_frame_id.<setting>
```

因此 smoke test 可以读到 parquet，但最终 `pixel_goal_list` 为空，dataset 长度为 0。

当前可运行的替代路径是 `vln_pe_r2r_offline`：

- 使用 `vln_pe` mini 数据中的 `observation.robot_position` 和 `observation.robot_yaw` 构造离线轨迹监督。
- 使用 `rgb.npy/depth.npy` 构造 `traj_images/traj_depths`。
- pixel goal 坐标只作为上下文占位，不作为 loss。
- 该路径已经完成 loader smoke test、System1 候选生成/reward 构造 smoke，以及 1-step 3090 训练 smoke。

## 下一步条件

要真正进入 R2R-only offline GRPO / preference 后训练，需要先满足下面任一条件：

1. 下载包含 `pose.<setting>`、`goal.<setting>`、`relative_goal_frame_id.<setting>` 的 InternData-N1 轨迹数据版本。
2. 或者新增一个数据转换器，从 simulator / raw trajectory 中补齐这些字段。
3. 或者改写训练目标，不再依赖 pixel goal trajectory query，但这会偏离当前 InternVLA-N1 DualVLN 的 System1 训练接口。
