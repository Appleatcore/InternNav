# VLN-PE External Skip Keys

## Change Log

### 2026-06-25

- Added support for external VLN-PE path-key skip files.
- Kept the existing hard-coded `skip_path_key_list` for known bad samples.
- Updated the episode loader and result logger to use the combined skip set, so skipped samples are excluded from both evaluation and final metric denominators.
- Added a Slurm watchdog to `scripts/eval/slurm/eval_internvla_n1_val_unseen_full_1m_4090.sbatch`.

## Why

Some InternUtopia/Isaac VLN-PE samples can hang inside simulator stepping. When that happens, episode-level checks such as `max_step`, `stuck`, or `fall` may not run because the process is blocked before an observation is returned.

The external skip file lets a Slurm watchdog append a stuck `path_key`, restart the evaluation, and continue without editing source code.

## Files Read

The evaluator now skips the union of:

- hard-coded keys in `skip_path_key_list`
- keys from the file pointed to by `INTERNNAV_SKIP_PATH_KEYS_FILE`
- keys from `logs/skip_path_keys.txt`

The environment variable path is intended for task-specific Slurm jobs. The global `logs/skip_path_keys.txt` file is a fallback that works even if the Slurm script does not set the variable.

## File Format

Each key can be written on its own line:

```text
4152_1021
6926_1748
```

Comments and comma-separated values are also accepted:

```text
# stuck in QUCTc6BB5sX on 2026-06-24
4152_1021

6926_1748, 5167_1329
```

## Slurm Usage

A task-specific job can set:

```bash
export INTERNNAV_SKIP_PATH_KEYS_FILE="${INTERNNAV_SKIP_PATH_KEYS_FILE:-logs/${TASK_NAME}/skip_path_keys.txt}"
mkdir -p "$(dirname "${INTERNNAV_SKIP_PATH_KEYS_FILE}")"
touch "${INTERNNAV_SKIP_PATH_KEYS_FILE}"
```

If a watchdog detects that a sample has been stuck for more than two hours, it can append the current key:

```bash
echo "4152_1021" >> "${INTERNNAV_SKIP_PATH_KEYS_FILE}"
```

Then the job should be restarted. On the next evaluator initialization, `4152_1021` will be excluded from the episode loader and from metric aggregation.

## Full Val-Unseen Watchdog

The full VLN-PE val-unseen Slurm script now sets a task-specific skip file and runs `scripts/eval/eval.py` under a watchdog:

```bash
scripts/eval/slurm/eval_internvla_n1_val_unseen_full_1m_4090.sbatch
```

The script requests four 3090 GPUs by default and launches one evaluator rank per GPU with `srun`. The dataset loader slices episodes with `rank::world_size`, so ranks evaluate disjoint path keys and write into the same resumable result store.

Default watchdog behavior:

- `INTERNNAV_SKIP_PATH_KEYS_FILE=logs/test_n1_val_unseen_full_3m/skip_path_keys.txt`
- `INTERNNAV_STUCK_TIMEOUT_SECONDS=7200`
- `INTERNNAV_WATCHDOG_POLL_SECONDS=60`
- `INTERNNAV_MAX_WATCHDOG_RESTARTS=20`

If `logs/test_n1_val_unseen_full_3m/progress/test_n1_val_unseen_full_3m.log` has not been updated for more than the timeout, the script:

1. Parses the last started but unfinished `trajectory_id`.
2. Appends it to the skip file if it is not already present.
3. Terminates the blocked evaluator process.
4. Restarts the evaluator so the new skip key takes effect.

The timeout can be overridden at submission time:

```bash
INTERNNAV_STUCK_TIMEOUT_SECONDS=5400 sbatch scripts/eval/slurm/eval_internvla_n1_val_unseen_full_1m_4090.sbatch
```

The GPU count can be overridden by Slurm options if more 3090 GPUs are available on one node:

```bash
sbatch --gres=gpu:rtx3090:4 --ntasks=4 --ntasks-per-node=4 --cpus-per-task=8 \
  scripts/eval/slurm/eval_internvla_n1_val_unseen_full_1m_4090.sbatch
```

## Current Limitation

The external skip file is read when the evaluator starts. A running Python process will not dynamically drop the current episode just because the file changed; the Slurm watchdog handles this by killing and restarting the evaluator.
