#!/usr/bin/env python
"""Smoke test NavPixelGoalDataset without loading the full model."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import transformers
from torchvision.transforms import v2
from transformers import AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))


def require_module(name: str) -> None:
    if importlib.util.find_spec(name) is None:
        raise RuntimeError(f"missing Python dependency: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", default="checkpoints/InternVLA-N1-DualVLN")
    parser.add_argument("--vln-dataset-use", default="r2r_125cm_0_30%1")
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--find-sample-id", default=None)
    parser.add_argument("--num-history", type=int, default=8)
    parser.add_argument("--sample-step", type=int, default=4)
    parser.add_argument("--num-future-steps", type=int, default=4)
    parser.add_argument("--predict-step-num", type=int, default=32)
    parser.add_argument("--resize-h", type=int, default=384)
    parser.add_argument("--resize-w", type=int, default=384)
    parser.add_argument("--offline-reward-path", default=None)
    parser.add_argument("--offline-reward-reduce", default="mean", choices=("mean", "max", "min"))
    args = parser.parse_args()

    for module_name in ("pyarrow",):
        require_module(module_name)

    from internnav.dataset.internvla_n1_lerobot_dataset import (  # noqa: PLC0415
        DataCollatorForSupervisedDataset,
        make_traj_sample_id,
        NavPixelGoalDataset,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=8192,
        padding_side="right",
        use_fast=False,
    )
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    data_args = SimpleNamespace(
        vln_dataset_use=args.vln_dataset_use,
        video_max_total_pixels=1664 * 28 * 28,
        video_min_total_pixels=256 * 28 * 28,
        model_type="internvla-n1",
        sample_step=args.sample_step,
        predict_step_num=args.predict_step_num,
        pixel_goal_only=True,
        offline_reward_path=args.offline_reward_path,
        offline_reward_reduce=args.offline_reward_reduce,
        num_future_steps=args.num_future_steps,
        num_history=args.num_history,
        image_processor=processor.image_processor,
        transform_train=v2.Resize((args.resize_h, args.resize_w)),
    )

    dataset = NavPixelGoalDataset(tokenizer=tokenizer, data_args=data_args)
    print(f"[dataset] length={len(dataset)}")
    if len(dataset) == 0:
        raise RuntimeError(
            "dataset is empty; check extracted data layout, vln_dataset_use, and whether parquet files contain "
            "pose.<setting>, goal.<setting>, and relative_goal_frame_id.<setting> columns"
        )

    samples = []
    start_idx = args.sample_index
    if args.find_sample_id is not None:
        start_idx = None
        for idx, sample in enumerate(dataset.list_data_dict):
            ep_id, data_path, video, height, pitch_1, pitch_2, instruction, frame_range, action, pose = sample
            sample_id = make_traj_sample_id(video, ep_id, frame_range[0], frame_range[1])
            if sample_id == args.find_sample_id:
                start_idx = idx
                break
        if start_idx is None:
            raise RuntimeError(f"sample_id not found: {args.find_sample_id}")

    for idx in range(start_idx, min(start_idx + args.num_samples, len(dataset))):
        item = dataset[idx]
        samples.append(item)
        print(
            "[sample]",
            idx,
            "input_ids",
            tuple(item["input_ids"].shape),
            "pixel_values",
            tuple(item["pixel_values"].shape),
            "traj_images",
            tuple(item["traj_images"].shape),
            "traj_depths",
            tuple(item["traj_depths"].shape),
            "traj_poses",
            tuple(item["traj_poses"].shape),
            "sample_id",
            item.get("sample_id"),
        )
        if "traj_reward_weights" in item:
            print("[sample]", idx, "traj_reward_weights", item["traj_reward_weights"].tolist())

    batch = DataCollatorForSupervisedDataset(tokenizer=tokenizer)(samples)
    print("[batch] keys=", sorted(batch.keys()))
    for key in (
        "input_ids",
        "pixel_values",
        "traj_images",
        "traj_depths",
        "traj_poses",
        "traj_reward_weights",
        "video_frame_num",
    ):
        value = batch.get(key)
        if isinstance(value, torch.Tensor):
            print(f"[batch] {key}={tuple(value.shape)}")


if __name__ == "__main__":
    main()
