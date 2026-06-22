#!/usr/bin/env bash
#SBATCH --job-name=internvla_n1_single_scene
#SBATCH --partition=rtx4090
#SBATCH --gres=gpu:rtx4090d:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=/srv/shared/home/ycl/workspace/InternNav/logs/slurm/%x_%j.out
#SBATCH --error=/srv/shared/home/ycl/workspace/InternNav/logs/slurm/%x_%j.err

set -euo pipefail

cd /srv/shared/home/ycl/workspace/InternNav

source /srv/shared/home/ycl/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

echo "[job] started at $(date)"
echo "[job] node=${SLURMD_NODENAME:-unknown}"
echo "[job] job_id=${SLURM_JOB_ID:-unknown}"
echo "[job] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[job] config=scripts/eval/configs/h1_internvla_n1_async_cfg.py"

nvidia-smi || true

python -u scripts/eval/eval.py \
  --config scripts/eval/configs/h1_internvla_n1_async_cfg.py

echo "[job] finished at $(date)"
