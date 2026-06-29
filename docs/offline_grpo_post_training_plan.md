# Offline GRPO 后训练简要计划

1. 先不要直接做 simulator-in-loop GRPO，成本高、工程改动大、容易卡在 Habitat/InternUtopia 环境。
2. 先把 `vln_ce/r2r` 解压并整理到代码默认读取路径，必要时用软链接。
3. 用当前 checkpoint 做小规模 loader smoke test，确认 `NavPixelGoalDataset` 能读出 RGB、pose、goal 和 trajectory query。
4. 先做 offline GRPO / preference-style 后训练：让 System1 生成多条候选轨迹，用 GT 轨迹的 ADE/FDE、终点误差、平滑度和目标接近度构造 reward。
5. 用 group-relative advantage 给 System1 的 diffusion loss 加权，先只训练 System1，冻结 System2。
6. 每次只用小 split 评估 `success_distance=3m` 的 SR/SPL，确认有效后再扩大数据。
