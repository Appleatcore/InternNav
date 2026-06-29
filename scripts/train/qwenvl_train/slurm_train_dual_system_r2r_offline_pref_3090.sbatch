#!/usr/bin/env bash
#SBATCH --job-name=internvla_r2r_offline_pref
#SBATCH --partition=rtx3090-large
#SBATCH --gres=gpu:rtx3090:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=/srv/shared/home/ycl/workspace/InternNav/logs/slurm/%x_%j.out
#SBATCH --error=/srv/shared/home/ycl/workspace/InternNav/logs/slurm/%x_%j.err

set -eo pipefail

cd /srv/shared/home/ycl/workspace/InternNav

source /srv/shared/home/ycl/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-env_isaaclab}"
set -u

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export INTERNVLA_ATTN_IMPLEMENTATION="${INTERNVLA_ATTN_IMPLEMENTATION:-sdpa}"
export INTERNVLA_SKIP_FINAL_SAVE="${INTERNVLA_SKIP_FINAL_SAVE:-true}"

NPROC_PER_NODE="${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-4}}"
MASTER_PORT="${MASTER_PORT:-$((RANDOM % 101 + 20001))}"

DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-scripts/train/qwenvl_train/zero2.json}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-checkpoints/InternVLA-N1-DualVLN}"
VLN_DATASET_USE="${VLN_DATASET_USE:-vln_pe_r2r_offline%10}"
RUN_NAME="${RUN_NAME:-InternVLA-N1-R2R-OfflinePref-Smoke}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/${RUN_NAME}}"
MAX_STEPS="${MAX_STEPS:-20}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
LR="${LR:-1e-5}"
OFFLINE_REWARD_PATH="${OFFLINE_REWARD_PATH:-}"
OFFLINE_REWARD_REDUCE="${OFFLINE_REWARD_REDUCE:-mean}"

echo "[job] started at $(date)"
echo "[job] node=${SLURMD_NODENAME:-unknown}"
echo "[job] job_id=${SLURM_JOB_ID:-manual}"
echo "[job] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[job] conda_env=${CONDA_DEFAULT_ENV:-unknown}"
echo "[job] nproc_per_node=${NPROC_PER_NODE}"
echo "[job] model=${MODEL_NAME_OR_PATH}"
echo "[job] datasets=${VLN_DATASET_USE}"
echo "[job] output_dir=${OUTPUT_DIR}"
echo "[job] max_steps=${MAX_STEPS}"
echo "[job] attn=${INTERNVLA_ATTN_IMPLEMENTATION}"
echo "[job] skip_final_save=${INTERNVLA_SKIP_FINAL_SAVE}"
echo "[job] offline_reward_path=${OFFLINE_REWARD_PATH:-none}"
nvidia-smi || true

python - <<'PY'
import importlib.util
missing = [
    name
    for name in ("pyarrow", "deepspeed")
    if importlib.util.find_spec(name) is None
]
if missing:
    raise SystemExit(f"missing required dependencies: {', '.join(missing)}")
PY

torchrun --nproc_per_node="${NPROC_PER_NODE}" \
    internnav/trainer/internvla_n1_trainer.py \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --vln_dataset_use "${VLN_DATASET_USE}" \
    --data_flatten False \
    --tune_mm_vision False \
    --tune_mm_mlp False \
    --tune_mm_llm False \
    --bf16 \
    --num_history 8 \
    --data_augmentation True \
    --resize_h 384 \
    --resize_w 384 \
    --sample_step 4 \
    --num_future_steps 4 \
    --predict_step_num 32 \
    --pixel_goal_only True \
    --offline_reward_path "${OFFLINE_REWARD_PATH}" \
    --offline_reward_reduce "${OFFLINE_REWARD_REDUCE}" \
    --system1 nextdit_async \
    --output_dir "${OUTPUT_DIR}" \
    --max_steps "${MAX_STEPS}" \
    --per_device_train_batch_size "${BATCH_SIZE}" \
    --per_device_eval_batch_size "${BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
    --max_pixels 313600 \
    --min_pixels 3136 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps "${SAVE_STEPS:-20}" \
    --save_total_limit 2 \
    --learning_rate "${LR}" \
    --weight_decay 0 \
    --warmup_ratio 0.003 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --lr_scheduler_kwargs '{"min_lr": 1e-06}' \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name "${RUN_NAME}" \
    --report_to none

echo "[job] finished at $(date)"
