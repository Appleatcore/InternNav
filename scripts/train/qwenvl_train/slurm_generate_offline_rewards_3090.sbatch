#!/usr/bin/env bash
#SBATCH --job-name=internvla_r2r_offline_rewards
#SBATCH --partition=rtx3090
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --output=/srv/shared/home/ycl/workspace/InternNav/logs/slurm/%x_%j.out
#SBATCH --error=/srv/shared/home/ycl/workspace/InternNav/logs/slurm/%x_%j.err

set -euo pipefail

cd /srv/shared/home/ycl/workspace/InternNav

source /srv/shared/home/ycl/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-env_isaaclab}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export INTERNVLA_ATTN_IMPLEMENTATION="${INTERNVLA_ATTN_IMPLEMENTATION:-sdpa}"

RUN_NAME="${RUN_NAME:-r2r_offline_grpo_smoke}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-checkpoints/InternVLA-N1-DualVLN}"
VLN_DATASET_USE="${VLN_DATASET_USE:-vln_pe_r2r_offline%10}"
OUTPUT_DIR="${OUTPUT_DIR:-logs/offline_grpo/${RUN_NAME}}"
CANDIDATE_PATH="${CANDIDATE_PATH:-${OUTPUT_DIR}/candidates.jsonl}"
REWARD_PATH="${REWARD_PATH:-${OUTPUT_DIR}/rewards.jsonl}"

echo "[job] started at $(date)"
echo "[job] node=${SLURMD_NODENAME:-unknown}"
echo "[job] job_id=${SLURM_JOB_ID:-manual}"
echo "[job] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[job] conda_env=${CONDA_DEFAULT_ENV:-unknown}"
echo "[job] model=${MODEL_NAME_OR_PATH}"
echo "[job] datasets=${VLN_DATASET_USE}"
echo "[job] candidate_path=${CANDIDATE_PATH}"
echo "[job] reward_path=${REWARD_PATH}"
nvidia-smi || true

mkdir -p "${OUTPUT_DIR}"

python -u scripts/train/qwenvl_train/generate_offline_traj_candidates.py \
    --model-name-or-path "${MODEL_NAME_OR_PATH}" \
    --vln-dataset-use "${VLN_DATASET_USE}" \
    --output "${CANDIDATE_PATH}" \
    --num-samples "${NUM_SAMPLES:-8}" \
    --sample-index "${SAMPLE_INDEX:-0}" \
    --context-frames "${CONTEXT_FRAMES:-1}" \
    --num-candidates "${NUM_CANDIDATES:-8}" \
    --num-inference-steps "${NUM_INFERENCE_STEPS:-10}" \
    --guidance-scale "${GUIDANCE_SCALE:-1.0}" \
    --device cuda:0 \
    --bf16

python -u scripts/train/qwenvl_train/build_offline_grpo_rewards.py \
    --input "${CANDIDATE_PATH}" \
    --output "${REWARD_PATH}" \
    --smoothness-weight "${SMOOTHNESS_WEIGHT:-0.1}" \
    --goal-weight "${GOAL_WEIGHT:-0.5}"

echo "[job] finished at $(date)"
echo "[job] use this reward file for training:"
echo "OFFLINE_REWARD_PATH=${REWARD_PATH} sbatch scripts/train/qwenvl_train/slurm_train_dual_system_r2r_offline_pref_3090.sbatch"
