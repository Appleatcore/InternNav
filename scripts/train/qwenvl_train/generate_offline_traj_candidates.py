#!/usr/bin/env python
"""Generate System1 trajectory candidates for offline GRPO-style rewards.

The output JSONL is the input format expected by build_offline_grpo_rewards.py.
This script intentionally processes one dataset item at a time because
InternVLAN1.generate_traj currently assumes batch-size 1 in the async System1
image branch used by the DualVLN checkpoint.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
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


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def move_tensor(value: torch.Tensor | None, device: torch.device, dtype: torch.dtype | None = None):
    if value is None:
        return None
    if dtype is not None and value.is_floating_point():
        return value.to(device=device, dtype=dtype)
    return value.to(device=device)


def find_dataset_index(dataset, sample_id: str) -> int:
    from internnav.dataset.internvla_n1_lerobot_dataset import make_traj_sample_id  # noqa: PLC0415

    for idx, sample in enumerate(dataset.list_data_dict):
        ep_id, _data_path, video, _height, _pitch_1, _pitch_2, _instruction, frame_range, _action, _pose = sample
        current_id = make_traj_sample_id(video, ep_id, frame_range[0], frame_range[1])
        if current_id == sample_id:
            return idx
    raise RuntimeError(f"sample_id not found: {sample_id}")


def build_dataset(args: argparse.Namespace):
    from internnav.dataset.internvla_n1_lerobot_dataset import NavPixelGoalDataset  # noqa: PLC0415

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.model_max_length,
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
        offline_reward_path=None,
        offline_reward_reduce="mean",
        num_future_steps=args.num_future_steps,
        num_history=args.num_history,
        image_processor=processor.image_processor,
        transform_train=v2.Resize((args.resize_h, args.resize_w)),
    )
    return NavPixelGoalDataset(tokenizer=tokenizer, data_args=data_args)


def disable_deepspeed_auto_import() -> None:
    """Avoid importing deepspeed from Transformers during inference-only loading.

    On the plain rtx3090 nodes deepspeed can fail at import time when CUDA_HOME is
    unavailable. Candidate generation does not use deepspeed, so treating it as
    unavailable here keeps model loading independent of that training dependency.
    """

    try:
        import transformers.integrations.deepspeed as hf_deepspeed
    except Exception:
        return
    hf_deepspeed.is_deepspeed_available = lambda: False


def load_model(args: argparse.Namespace, device: torch.device):
    disable_deepspeed_auto_import()

    from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM  # noqa: PLC0415

    torch_dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32
    model = InternVLAN1ForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
        attn_implementation=args.attn_implementation,
        device_map={"": device} if device.type == "cuda" else None,
    )
    if device.type != "cuda":
        model.to(device)
    model.eval()
    return model, torch_dtype


def generate_for_item(model, item: dict, args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> list[dict]:
    sample_id = item["sample_id"]
    input_ids = item["input_ids"].to(device)
    pixel_values = move_tensor(item.get("pixel_values"), device, dtype)
    image_grid_thw = move_tensor(item.get("image_grid_thw"), device)
    traj_images = move_tensor(item["traj_images"], device, dtype)
    traj_depths = move_tensor(item.get("traj_depths"), device, dtype)
    traj_poses = item["traj_poses"]

    if traj_images.ndim != 4:
        raise RuntimeError(f"expected traj_images [F,H,W,C], got {tuple(traj_images.shape)}")
    if traj_poses.ndim != 3:
        raise RuntimeError(f"expected traj_poses [F,T,3], got {tuple(traj_poses.shape)}")

    max_context_frames = min(args.context_frames, traj_images.shape[0])
    records: list[dict] = []

    with torch.inference_mode():
        traj_latents = model.generate_latents(input_ids, pixel_values, image_grid_thw)
        for frame_idx in range(max_context_frames):
            frame_latents = traj_latents
            current_rgb = traj_images[frame_idx : frame_idx + 1]
            pixel_goal_rgb = traj_images[0:1]
            images_dp = torch.stack([pixel_goal_rgb, current_rgb], dim=1)

            depths_dp = None
            if traj_depths is not None:
                current_depth = traj_depths[frame_idx : frame_idx + 1].unsqueeze(-1)
                pixel_goal_depth = traj_depths[0:1].unsqueeze(-1)
                depths_dp = torch.stack([pixel_goal_depth, current_depth], dim=1)

            candidates = model.generate_traj(
                traj_latents=frame_latents,
                images_dp=images_dp,
                depths_dp=depths_dp,
                predict_step_nums=args.predict_step_num,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                num_sample_trajs=args.num_candidates,
            )

            candidates = candidates.detach().float().cpu()
            gt = traj_poses[frame_idx].float().cpu()
            goal = gt[-1, :2]
            for candidate_id, candidate in enumerate(candidates):
                records.append(
                    {
                        "sample_id": sample_id,
                        "candidate_id": candidate_id,
                        "frame_index": frame_idx,
                        "candidate": candidate.tolist(),
                        "gt": gt.tolist(),
                        "goal": goal.tolist(),
                    }
                )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", default="checkpoints/InternVLA-N1-DualVLN")
    parser.add_argument("--vln-dataset-use", default="vln_pe_r2r_offline%10")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--find-sample-id", default=None)
    parser.add_argument("--context-frames", type=int, default=1)
    parser.add_argument("--num-candidates", type=int, default=8)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--attn-implementation", default=os.environ.get("INTERNVLA_ATTN_IMPLEMENTATION", "sdpa"))
    parser.add_argument("--num-history", type=int, default=8)
    parser.add_argument("--sample-step", type=int, default=4)
    parser.add_argument("--num-future-steps", type=int, default=4)
    parser.add_argument("--predict-step-num", type=int, default=32)
    parser.add_argument("--resize-h", type=int, default=384)
    parser.add_argument("--resize-w", type=int, default=384)
    parser.add_argument("--model-max-length", type=int, default=8192)
    args = parser.parse_args()

    for module_name in ("pyarrow",):
        require_module(module_name)

    device = resolve_device(args.device)
    if device.type == "cpu" and (args.bf16 or args.fp16):
        args.bf16 = False
        args.fp16 = False

    dataset = build_dataset(args)
    if len(dataset) == 0:
        raise RuntimeError("dataset is empty; run the loader smoke test first and check the extracted data layout")

    start_idx = find_dataset_index(dataset, args.find_sample_id) if args.find_sample_id else args.sample_index
    end_idx = min(start_idx + args.num_samples, len(dataset))
    if start_idx >= end_idx:
        raise RuntimeError(f"empty sample range: start={start_idx}, end={end_idx}, dataset_len={len(dataset)}")

    model, dtype = load_model(args, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    total_records = 0
    with args.output.open("w", encoding="utf-8") as f:
        for idx in range(start_idx, end_idx):
            item = dataset[idx]
            records = generate_for_item(model, item, args, device, dtype)
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            total_records += len(records)
            print(f"[sample] idx={idx} sample_id={item['sample_id']} candidates={len(records)}", flush=True)

    print(f"[summary] wrote {total_records} candidates to {args.output}")


if __name__ == "__main__":
    main()
