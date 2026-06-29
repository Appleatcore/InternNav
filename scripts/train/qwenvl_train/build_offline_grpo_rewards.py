#!/usr/bin/env python
"""Build offline GRPO-style rewards for trajectory candidates.

Input JSONL format:
  {
    "sample_id": "scene__episode__frame",
    "candidate_id": 0,
    "candidate": [[x, y], ...],
    "gt": [[x, y], ...],
    "goal": [x, y]   # optional
  }

Output JSONL adds reward components and group-relative advantage.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def as_xy(points) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"trajectory must have shape [T, >=2], got {arr.shape}")
    return arr[:, :2]


def align_traj(candidate: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    length = min(len(candidate), len(gt))
    if length == 0:
        raise ValueError("empty candidate or gt trajectory")
    return candidate[:length], gt[:length]


def trajectory_smoothness(candidate: np.ndarray) -> float:
    if len(candidate) < 3:
        return 0.0
    velocity = np.diff(candidate, axis=0)
    accel = np.diff(velocity, axis=0)
    return float(np.linalg.norm(accel, axis=1).mean())


def score_record(record: dict, smoothness_weight: float, goal_weight: float) -> dict:
    candidate, gt = align_traj(as_xy(record["candidate"]), as_xy(record["gt"]))
    distances = np.linalg.norm(candidate - gt, axis=1)
    ade = float(distances.mean())
    fde = float(distances[-1])
    smoothness = trajectory_smoothness(candidate)

    goal_error = fde
    if "goal" in record and record["goal"] is not None:
        goal = np.asarray(record["goal"], dtype=np.float32)[:2]
        goal_error = float(np.linalg.norm(candidate[-1] - goal))

    reward = -ade - fde - smoothness_weight * smoothness - goal_weight * goal_error
    scored = dict(record)
    scored.update(
        {
            "ade": ade,
            "fde": fde,
            "smoothness": smoothness,
            "goal_error": goal_error,
            "reward": float(reward),
        }
    )
    return scored


def add_group_advantages(records: list[dict], eps: float) -> list[dict]:
    grouped = defaultdict(list)
    for record in records:
        grouped[record["sample_id"]].append(record)

    output = []
    for sample_id, group in grouped.items():
        rewards = np.asarray([item["reward"] for item in group], dtype=np.float32)
        mean = float(rewards.mean())
        std = float(rewards.std())
        for item in group:
            advantage = (item["reward"] - mean) / max(std, eps)
            weight = max(0.0, 1.0 + advantage)
            item = dict(item)
            item["group_reward_mean"] = mean
            item["group_reward_std"] = std
            item["advantage"] = float(advantage)
            item["loss_weight"] = float(weight)
            output.append(item)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoothness-weight", type=float, default=0.1)
    parser.add_argument("--goal-weight", type=float, default=0.5)
    parser.add_argument("--eps", type=float, default=1e-6)
    args = parser.parse_args()

    records = []
    with args.input.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(score_record(json.loads(line), args.smoothness_weight, args.goal_weight))

    records = add_group_advantages(records, args.eps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[summary] wrote {len(records)} scored candidates to {args.output}")


if __name__ == "__main__":
    main()
