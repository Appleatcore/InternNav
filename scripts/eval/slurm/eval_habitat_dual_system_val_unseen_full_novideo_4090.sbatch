#!/usr/bin/env bash
#SBATCH --job-name=habitat_dual_val_unseen_full
#SBATCH --partition=rtx4090
#SBATCH --gres=gpu:rtx4090d:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/srv/shared/home/ycl/workspace/InternNav/logs/slurm/%x_%j.out
#SBATCH --error=/srv/shared/home/ycl/workspace/InternNav/logs/slurm/%x_%j.err

set -euo pipefail

cd /srv/shared/home/ycl/workspace/InternNav

source /srv/shared/home/ycl/miniconda3/etc/profile.d/conda.sh
conda activate internvla_habitat_eval

CONFIG="scripts/eval/configs/habitat_dual_system_val_unseen_full_novideo_cfg.py"
RUN_NAME="${RUN_NAME:-habitat_dual_val_unseen_full_novideo_${SLURM_JOB_ID:-manual}}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export INTERNVLA_ATTN_IMPLEMENTATION="${INTERNVLA_ATTN_IMPLEMENTATION:-sdpa}"
export INTERNNAV_HABITAT_CONFIG_PATH="${INTERNNAV_HABITAT_CONFIG_PATH:-scripts/eval/configs/vln_r2r_val_unseen_full.yaml}"
export INTERNNAV_OUTPUT_PATH="${INTERNNAV_OUTPUT_PATH:-./logs/habitat/${RUN_NAME}}"
export INTERNNAV_MAX_STEPS_PER_EPISODE="${INTERNNAV_MAX_STEPS_PER_EPISODE:-500}"
export INTERNNAV_VIS_DEBUG="${INTERNNAV_VIS_DEBUG:-false}"

mkdir -p logs/slurm "${INTERNNAV_OUTPUT_PATH}"

echo "[job] started at $(date)"
echo "[job] node=${SLURMD_NODENAME:-unknown}"
echo "[job] job_id=${SLURM_JOB_ID:-unknown}"
echo "[job] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[job] config=${CONFIG}"
echo "[job] habitat_config=${INTERNNAV_HABITAT_CONFIG_PATH}"
echo "[job] output_path=${INTERNNAV_OUTPUT_PATH}"
echo "[job] attn=${INTERNVLA_ATTN_IMPLEMENTATION}"
echo "[job] max_steps_per_episode=${INTERNNAV_MAX_STEPS_PER_EPISODE}"
echo "[job] save_video=false"
echo "[job] estimated_time=20-30h"

nvidia-smi || true

python -u scripts/eval/eval.py --config "${CONFIG}"

echo "[job] finished at $(date)"
echo "[job] result=${INTERNNAV_OUTPUT_PATH}/result.json"
echo "[job] progress=${INTERNNAV_OUTPUT_PATH}/progress.json"
