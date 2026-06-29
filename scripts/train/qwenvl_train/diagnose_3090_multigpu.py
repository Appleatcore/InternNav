#!/usr/bin/env python
"""Small diagnostics for isolating RTX3090 multi-GPU smoke failures.

The script is intentionally independent of the training entrypoint. Run it with
torchrun so every rank exercises the same import, filesystem, and NCCL paths.
"""

from __future__ import annotations

import argparse
import importlib
import json
import mmap
import os
import pathlib
import socket
import sys
import time
from typing import Iterable


def rank_info() -> dict:
    return {
        "host": socket.gethostname(),
        "rank": int(os.environ.get("RANK", "0")),
        "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }


def log(event: str, **payload) -> None:
    record = {"event": event, **rank_info(), **payload}
    print(json.dumps(record, sort_keys=True), flush=True)


def import_module(name: str) -> None:
    start = time.time()
    module = importlib.import_module(name)
    log(
        "import_ok",
        module=name,
        version=getattr(module, "__version__", None),
        seconds=round(time.time() - start, 3),
    )


def mode_imports(args: argparse.Namespace) -> None:
    modules = [
        "numpy",
        "sympy",
        "torch",
        "torch._dynamo",
        "deepspeed",
    ]
    if args.extra_import:
        modules.extend(args.extra_import)
    for module in modules:
        import_module(module)


def mode_nccl(args: argparse.Namespace) -> None:
    import torch
    import torch.distributed as dist

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=args.backend, init_method="env://")
    tensor = torch.ones(1, device="cuda") * (dist.get_rank() + 1)
    dist.all_reduce(tensor)
    expected = dist.get_world_size() * (dist.get_world_size() + 1) / 2
    if float(tensor.item()) != expected:
        raise RuntimeError(f"unexpected all_reduce result {tensor.item()} != {expected}")
    dist.barrier()
    log("nccl_ok", backend=args.backend, value=float(tensor.item()))
    dist.destroy_process_group()


def read_head_tail(path: pathlib.Path, bytes_per_side: int = 1024 * 1024) -> int:
    size = path.stat().st_size
    with path.open("rb") as handle:
        handle.read(min(size, bytes_per_side))
        if size > bytes_per_side:
            handle.seek(max(0, size - bytes_per_side))
            handle.read(min(size, bytes_per_side))
    return size


def mmap_probe(path: pathlib.Path, bytes_to_touch: int = 1024 * 1024) -> int:
    size = path.stat().st_size
    if size == 0:
        return size
    with path.open("rb") as handle:
        with mmap.mmap(handle.fileno(), length=0, access=mmap.ACCESS_READ) as mapped:
            mapped[: min(size, bytes_to_touch)]
            if size > bytes_to_touch:
                mapped[max(0, size - bytes_to_touch) :]
    return size


def compile_python_source(path: pathlib.Path) -> None:
    source = path.read_text(encoding="utf-8")
    compile(source, str(path), "exec")


def iter_existing(paths: Iterable[str]) -> Iterable[pathlib.Path]:
    for raw in paths:
        path = pathlib.Path(raw)
        if path.exists():
            yield path
        else:
            log("path_missing", path=str(path))


def default_python_probe_paths() -> list[str]:
    import deepspeed
    import sympy
    import torch

    base_paths = [
        pathlib.Path(deepspeed.__file__).resolve(),
        pathlib.Path(torch.__file__).resolve(),
        pathlib.Path(sympy.__file__).resolve(),
    ]
    site = pathlib.Path(torch.__file__).resolve().parent
    base_paths.extend(
        [
            site / "_dynamo" / "__init__.py",
            site / "_dynamo" / "convert_frame.py",
            site / "onnx" / "symbolic_caffe2.py",
        ]
    )
    return [str(path) for path in base_paths]


def default_checkpoint_probe_paths(model_dir: str, reward_path: str | None) -> list[str]:
    paths = [
        os.path.join(model_dir, "config.json"),
        os.path.join(model_dir, "model.safetensors.index.json"),
    ]
    paths.extend(sorted(str(path) for path in pathlib.Path(model_dir).glob("model-*.safetensors")))
    if reward_path:
        paths.append(reward_path)
    return paths


def mode_io(args: argparse.Namespace) -> None:
    mode_imports(argparse.Namespace(extra_import=[]))
    probe_paths = default_python_probe_paths()
    probe_paths.extend(default_checkpoint_probe_paths(args.model_dir, args.reward_path))
    if args.path:
        probe_paths.extend(args.path)

    for path in iter_existing(probe_paths):
        start = time.time()
        size = read_head_tail(path)
        log("read_ok", path=str(path), size=size, seconds=round(time.time() - start, 3))
        if path.suffix == ".py":
            start = time.time()
            compile_python_source(path)
            log("compile_ok", path=str(path), seconds=round(time.time() - start, 3))
        start = time.time()
        mmap_probe(path)
        log("mmap_ok", path=str(path), seconds=round(time.time() - start, 3))

    try:
        from safetensors import safe_open
    except Exception as exc:  # pragma: no cover - diagnostic only
        log("safetensors_unavailable", error=repr(exc))
        return

    for path in pathlib.Path(args.model_dir).glob("model-*.safetensors"):
        start = time.time()
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            sample = keys[0] if keys else None
            if sample:
                _ = handle.get_tensor(sample)
        log(
            "safetensors_ok",
            path=str(path),
            tensor_count=len(keys),
            sample=sample,
            seconds=round(time.time() - start, 3),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("imports", "nccl", "io", "all"), default="all")
    parser.add_argument("--backend", default="nccl")
    parser.add_argument("--model-dir", default="checkpoints/InternVLA-N1-DualVLN")
    parser.add_argument(
        "--reward-path",
        default="logs/offline_grpo/r2r_offline_grpo_vln_pe_full_1gpu/rewards.jsonl",
    )
    parser.add_argument("--path", action="append", default=[])
    parser.add_argument("--extra-import", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log(
        "start",
        mode=args.mode,
        python=sys.executable,
        cwd=os.getcwd(),
        cuda_home=os.environ.get("CUDA_HOME"),
        triton_cache=os.environ.get("TRITON_CACHE_DIR"),
        torch_extensions=os.environ.get("TORCH_EXTENSIONS_DIR"),
        pycache_prefix=os.environ.get("PYTHONPYCACHEPREFIX"),
    )
    if args.mode in ("imports", "all"):
        mode_imports(args)
    if args.mode in ("io", "all"):
        mode_io(args)
    if args.mode in ("nccl", "all"):
        mode_nccl(args)
    log("done", mode=args.mode)


if __name__ == "__main__":
    main()
