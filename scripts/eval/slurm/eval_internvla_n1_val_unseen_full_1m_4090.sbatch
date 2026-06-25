#!/usr/bin/env bash
#SBATCH --job-name=internvla_n1_val_unseen_full_3m
#SBATCH --partition=rtx3090
#SBATCH --gres=gpu:rtx3090:4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --output=/srv/shared/home/ycl/workspace/InternNav/logs/slurm/%x_%j.out
#SBATCH --error=/srv/shared/home/ycl/workspace/InternNav/logs/slurm/%x_%j.err

set -euo pipefail

cd /srv/shared/home/ycl/workspace/InternNav

source /srv/shared/home/ycl/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

CONFIG="scripts/eval/configs/h1_internvla_n1_val_unseen_full_1m_cfg.py"
TASK_NAME="test_n1_val_unseen_full_3m"
PROGRESS_LOG="logs/${TASK_NAME}/progress/${TASK_NAME}.log"
export CONFIG

export INTERNNAV_SKIP_PATH_KEYS_FILE="${INTERNNAV_SKIP_PATH_KEYS_FILE:-logs/${TASK_NAME}/skip_path_keys.txt}"
export INTERNNAV_STUCK_TIMEOUT_SECONDS="${INTERNNAV_STUCK_TIMEOUT_SECONDS:-7200}"
export INTERNNAV_WATCHDOG_POLL_SECONDS="${INTERNNAV_WATCHDOG_POLL_SECONDS:-60}"
export INTERNNAV_MAX_WATCHDOG_RESTARTS="${INTERNNAV_MAX_WATCHDOG_RESTARTS:-20}"

mkdir -p "$(dirname "${INTERNNAV_SKIP_PATH_KEYS_FILE}")"
touch "${INTERNNAV_SKIP_PATH_KEYS_FILE}"

echo "[job] started at $(date)"
echo "[job] node=${SLURMD_NODENAME:-unknown}"
echo "[job] job_id=${SLURM_JOB_ID:-unknown}"
echo "[job] ntasks=${SLURM_NTASKS:-1}"
echo "[job] cpus_per_task=${SLURM_CPUS_PER_TASK:-8}"
echo "[job] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[job] config=${CONFIG}"
echo "[job] output_path=logs/${TASK_NAME}"
echo "[job] save_video=false"
echo "[job] estimated_time=14-18h"
echo "[job] skip_file=${INTERNNAV_SKIP_PATH_KEYS_FILE}"
echo "[job] stuck_timeout_seconds=${INTERNNAV_STUCK_TIMEOUT_SECONDS}"

nvidia-smi || true

restart_count=0
while true; do
    run_start=$(date +%s)
    watchdog_killed=0

    srun --ntasks="${SLURM_NTASKS:-1}" --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
        bash -c 'export CUDA_VISIBLE_DEVICES="${SLURM_LOCALID:-0}"; echo "[rank ${SLURM_PROCID:-0}] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"; python -u scripts/eval/eval.py --config "${CONFIG}"' &
    eval_pid=$!

    while kill -0 "${eval_pid}" 2>/dev/null; do
        sleep "${INTERNNAV_WATCHDOG_POLL_SECONDS}"
        kill -0 "${eval_pid}" 2>/dev/null || break
        [[ -f "${PROGRESS_LOG}" ]] || continue

        progress_mtime=$(stat -c %Y "${PROGRESS_LOG}")
        if (( progress_mtime < run_start )); then
            continue
        fi

        now=$(date +%s)
        progress_age=$((now - progress_mtime))
        if (( progress_age < INTERNNAV_STUCK_TIMEOUT_SECONDS )); then
            continue
        fi

        stuck_key=$(python - "${PROGRESS_LOG}" <<'PY'
import re
import sys

path = sys.argv[1]
starts = []
finishes = set()
with open(path, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        m = re.search(r'start sampling trajectory_id:\s*(\S+)', line)
        if m:
            starts.append(m.group(1))
        m = re.search(r'finish:\s*\[trajectory_id:([^\]]+)\]', line)
        if m:
            finishes.add(m.group(1).strip())

for key in reversed(starts):
    if key not in finishes:
        print(key)
        break
PY
)

        echo "[watchdog] progress log stale for ${progress_age}s at $(date)"
        if [[ -n "${stuck_key}" ]]; then
            if ! grep -qxF "${stuck_key}" "${INTERNNAV_SKIP_PATH_KEYS_FILE}"; then
                echo "${stuck_key}" >> "${INTERNNAV_SKIP_PATH_KEYS_FILE}"
            fi
            echo "[watchdog] added stuck key to skip file: ${stuck_key}"
        else
            echo "[watchdog] no unfinished trajectory found; killing eval without appending skip key"
        fi

        kill -TERM "${eval_pid}" 2>/dev/null || true
        sleep 30
        kill -KILL "${eval_pid}" 2>/dev/null || true
        wait "${eval_pid}" || true
        watchdog_killed=1
        break
    done

    if (( watchdog_killed == 0 )); then
        if wait "${eval_pid}"; then
            eval_status=0
        else
            eval_status=$?
        fi
        if (( eval_status == 0 )); then
            break
        fi
        echo "[job] eval exited with status ${eval_status}"
        exit "${eval_status}"
    fi

    restart_count=$((restart_count + 1))
    if (( restart_count > INTERNNAV_MAX_WATCHDOG_RESTARTS )); then
        echo "[watchdog] exceeded max restarts: ${INTERNNAV_MAX_WATCHDOG_RESTARTS}"
        exit 124
    fi
    echo "[watchdog] restarting eval after stuck sample; restart_count=${restart_count}"
done

echo "[job] finished at $(date)"
