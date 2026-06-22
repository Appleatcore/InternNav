#!/usr/bin/env bash
set -u

cd /srv/shared/home/ycl/workspace/InternNav || exit 1

LOG="logs/slurm/submit_habitat_after_4611.log"
JOB_TO_WAIT="${1:-4611}"
HABITAT_SCRIPT="scripts/eval/slurm/eval_habitat_dual_system_val_unseen_full_novideo_4090.sbatch"

mkdir -p "$(dirname "$LOG")"

{
  echo "[defer] started at $(date '+%F %T %Z'); waiting for job ${JOB_TO_WAIT}"

  while squeue -h -j "${JOB_TO_WAIT}" | grep -q .; do
    sleep 300
  done

  echo "[defer] job ${JOB_TO_WAIT} left queue at $(date '+%F %T %Z'); submitting Habitat script"

  for attempt in $(seq 1 12); do
    echo "[defer] sbatch attempt ${attempt} at $(date '+%F %T %Z')"
    if sbatch "${HABITAT_SCRIPT}"; then
      echo "[defer] Habitat sbatch submitted at $(date '+%F %T %Z')"
      exit 0
    fi
    sleep 300
  done

  echo "[defer] failed to submit Habitat after 12 attempts at $(date '+%F %T %Z')"
  exit 1
} >> "${LOG}" 2>&1
