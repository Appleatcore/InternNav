#!/usr/bin/env bash
#SBATCH --job-name=internvla_diag_3090_multigpu
#SBATCH --partition=rtx3090
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/srv/shared/home/ycl/workspace/InternNav/logs/slurm/%x_%j.out
#SBATCH --error=/srv/shared/home/ycl/workspace/InternNav/logs/slurm/%x_%j.err

set -eo pipefail

cd /srv/shared/home/ycl/workspace/InternNav

source /srv/shared/home/ycl/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-env_isaaclab}"
set -u

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export WANDB_DISABLED=true

find_cuda_home() {
    local candidate
    for candidate in \
        "${CUDA_HOME:-}" \
        "${CUDA_PATH:-}" \
        "${CONDA_PREFIX:-}" \
        "${CONDA_PREFIX:-}/targets/x86_64-linux" \
        /usr/local/cuda \
        /usr/local/cuda-12.8 \
        /usr/local/cuda-12 \
        /usr/local/cuda-13.0; do
        if [[ -n "${candidate}" && -x "${candidate}/bin/nvcc" ]]; then
            echo "${candidate}"
            return 0
        fi
    done
    return 1
}

if CUDA_HOME_FOUND="$(find_cuda_home)"; then
    export CUDA_HOME="${CUDA_HOME_FOUND}"
    export CUDA_PATH="${CUDA_HOME_FOUND}"
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-1}}"
MASTER_PORT="${MASTER_PORT:-$((RANDOM % 101 + 23001))}"
DIAG_MODE="${DIAG_MODE:-all}"
MODEL_DIR="${MODEL_DIR:-checkpoints/InternVLA-N1-DualVLN}"
REWARD_PATH="${REWARD_PATH:-logs/offline_grpo/r2r_offline_grpo_vln_pe_full_1gpu/rewards.jsonl}"

LOCAL_CACHE_ROOT="${LOCAL_CACHE_ROOT:-/tmp/ycl_internnav_diag_${SLURM_JOB_ID:-manual}}"
if [[ "${USE_LOCAL_CACHE:-1}" == "1" ]]; then
    export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${LOCAL_CACHE_ROOT}/triton}"
    export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${LOCAL_CACHE_ROOT}/torch_extensions}"
    export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-${LOCAL_CACHE_ROOT}/pycache}"
    mkdir -p "${TRITON_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}" "${PYTHONPYCACHEPREFIX}"
fi

echo "[diag] started_at=$(date)"
echo "[diag] node=${SLURMD_NODENAME:-unknown}"
echo "[diag] job_id=${SLURM_JOB_ID:-manual}"
echo "[diag] mode=${DIAG_MODE}"
echo "[diag] nproc_per_node=${NPROC_PER_NODE}"
echo "[diag] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[diag] CUDA_HOME=${CUDA_HOME:-unset}"
echo "[diag] TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-unset}"
echo "[diag] TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-unset}"
echo "[diag] PYTHONPYCACHEPREFIX=${PYTHONPYCACHEPREFIX:-unset}"
echo "[diag] model_dir=${MODEL_DIR}"
echo "[diag] reward_path=${REWARD_PATH}"
nvidia-smi || true
which nvcc || true
nvcc -V || true

torchrun --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${MASTER_PORT}" \
    scripts/train/qwenvl_train/diagnose_3090_multigpu.py \
    --mode "${DIAG_MODE}" \
    --model-dir "${MODEL_DIR}" \
    --reward-path "${REWARD_PATH}"

echo "[diag] finished_at=$(date)"
