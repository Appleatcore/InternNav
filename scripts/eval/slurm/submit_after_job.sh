#!/usr/bin/env bash
set -u

if [ "$#" -ne 3 ]; then
  echo "usage: $0 <job_id_to_wait> <sbatch_script> <log_path>" >&2
  exit 2
fi

WAIT_JOB="$1"
SBATCH_SCRIPT="$2"
LOG_PATH="$3"

cd /srv/shared/home/ycl/workspace/InternNav || exit 1
mkdir -p "$(dirname "$LOG_PATH")"

{
  echo "[defer] started at $(date '+%F %T %Z'); waiting for job ${WAIT_JOB}"

  while squeue -h -j "${WAIT_JOB}" | grep -q .; do
    sleep 300
  done

  echo "[defer] job ${WAIT_JOB} left queue at $(date '+%F %T %Z'); submitting ${SBATCH_SCRIPT}"

  for attempt in $(seq 1 12); do
    echo "[defer] sbatch attempt ${attempt} at $(date '+%F %T %Z')"
    if sbatch "${SBATCH_SCRIPT}"; then
      echo "[defer] submitted ${SBATCH_SCRIPT} at $(date '+%F %T %Z')"
      exit 0
    fi
    sleep 300
  done

  echo "[defer] failed to submit ${SBATCH_SCRIPT} after 12 attempts at $(date '+%F %T %Z')"
  exit 1
} >> "${LOG_PATH}" 2>&1
