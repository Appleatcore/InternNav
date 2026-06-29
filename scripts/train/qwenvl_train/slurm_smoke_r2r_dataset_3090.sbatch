#!/usr/bin/env bash
#SBATCH --job-name=internvla_r2r_dataset_smoke
#SBATCH --partition=rtx3090
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/srv/shared/home/ycl/workspace/InternNav/logs/slurm/%x_%j.out
#SBATCH --error=/srv/shared/home/ycl/workspace/InternNav/logs/slurm/%x_%j.err

set -euo pipefail

cd /srv/shared/home/ycl/workspace/InternNav

source /srv/shared/home/ycl/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-env_isaaclab}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "[job] started at $(date)"
echo "[job] node=${SLURMD_NODENAME:-unknown}"
echo "[job] job_id=${SLURM_JOB_ID:-manual}"
echo "[job] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[job] conda_env=${CONDA_DEFAULT_ENV:-unknown}"
nvidia-smi || true

python -u scripts/dataset_converters/prepare_internvla_traj_data.py \
    --src-dir "${SRC_DIR:-traj_data/vln_pe/r2r}" \
    --extract-root "${EXTRACT_ROOT:-traj_data/vln_pe/r2r_extracted}" \
    --flat-root "${FLAT_ROOT:-traj_data/vln_pe_r2r}" \
    --manifest "${MANIFEST:-traj_data/vln_pe/r2r_units.txt}" \
    --limit-scenes "${LIMIT_SCENES:-1}"

python -u scripts/train/qwenvl_train/smoke_nav_pixel_goal_dataset.py \
    --model-name-or-path "${MODEL_NAME_OR_PATH:-checkpoints/InternVLA-N1-DualVLN}" \
    --vln-dataset-use "${VLN_DATASET_USE:-vln_pe_r2r_offline%10}" \
    --num-samples "${NUM_SAMPLES:-2}"

echo "[job] finished at $(date)"
