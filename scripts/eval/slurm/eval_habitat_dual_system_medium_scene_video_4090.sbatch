#!/usr/bin/env bash
#SBATCH --job-name=habitat_dual_medium_video
#SBATCH --partition=rtx4090
#SBATCH --gres=gpu:rtx4090d:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/srv/shared/home/ycl/workspace/InternNav/logs/slurm/%x_%j.out
#SBATCH --error=/srv/shared/home/ycl/workspace/InternNav/logs/slurm/%x_%j.err

set -euo pipefail

cd /srv/shared/home/ycl/workspace/InternNav

source /srv/shared/home/ycl/miniconda3/etc/profile.d/conda.sh
conda activate internvla_habitat_eval

CONFIG="scripts/eval/configs/habitat_dual_system_single_scene_video_cfg.py"
SCENE_ID="${SCENE_ID:-X7HyMhZNoso}"
SCENE_CONFIG="${SCENE_CONFIG:-scripts/eval/configs/vln_r2r_val_unseen_${SCENE_ID}.yaml}"
RUN_NAME="${RUN_NAME:-habitat_dual_medium_scene_${SCENE_ID}_${SLURM_JOB_ID:-manual}}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export INTERNVLA_ATTN_IMPLEMENTATION="${INTERNVLA_ATTN_IMPLEMENTATION:-sdpa}"
export INTERNNAV_HABITAT_CONFIG_PATH="${SCENE_CONFIG}"
export INTERNNAV_OUTPUT_PATH="${INTERNNAV_OUTPUT_PATH:-./logs/habitat/${RUN_NAME}}"
export INTERNNAV_MAX_STEPS_PER_EPISODE="${INTERNNAV_MAX_STEPS_PER_EPISODE:-500}"
export INTERNNAV_VIS_DEBUG="${INTERNNAV_VIS_DEBUG:-false}"

mkdir -p logs/slurm "${INTERNNAV_OUTPUT_PATH}"

echo "[job] started at $(date)"
echo "[job] node=${SLURMD_NODENAME:-unknown}"
echo "[job] job_id=${SLURM_JOB_ID:-unknown}"
echo "[job] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[job] config=${CONFIG}"
echo "[job] scene_id=${SCENE_ID}"
echo "[job] scene_config=${SCENE_CONFIG}"
echo "[job] output_path=${INTERNNAV_OUTPUT_PATH}"
echo "[job] attn=${INTERNVLA_ATTN_IMPLEMENTATION}"
echo "[job] max_steps_per_episode=${INTERNNAV_MAX_STEPS_PER_EPISODE}"
echo "[job] vis_debug=${INTERNNAV_VIS_DEBUG}"

nvidia-smi || true

python -u scripts/eval/eval.py --config "${CONFIG}"

echo "[job] finished at $(date)"
echo "[job] result=${INTERNNAV_OUTPUT_PATH}/result.json"
echo "[job] progress=${INTERNNAV_OUTPUT_PATH}/progress.json"
echo "[job] videos=${INTERNNAV_OUTPUT_PATH}/vis_0/${SCENE_ID}"
