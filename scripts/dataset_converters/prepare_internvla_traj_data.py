#!/usr/bin/env python
"""Prepare InternData-N1 trajectory archives for InternVLA-N1 training.

The mini InternData-N1 tarballs are nested as:
  scene_id/group_id/{data,meta,videos}

The current InternVLA-N1 loader expects each trainable unit to directly expose
{data,meta,videos}. This script extracts scene archives, expands nested video
archives, and creates a flat directory of symlinks:
  traj_data/r2r/scene_id__group_id -> extracted/scene_id/group_id
"""

from __future__ import annotations

import argparse
import os
import tarfile
from pathlib import Path


def is_lerobot_unit(path: Path) -> bool:
    return (path / "meta" / "episodes.jsonl").is_file() and (path / "data").is_dir() and (path / "videos").is_dir()


def extract_scene_archives(src_dir: Path, extract_root: Path, limit_scenes: int | None) -> list[Path]:
    tar_paths = sorted(src_dir.glob("*.tar.gz"))
    if limit_scenes is not None:
        tar_paths = tar_paths[:limit_scenes]

    extract_root.mkdir(parents=True, exist_ok=True)
    for tar_path in tar_paths:
        scene_dir = extract_root / tar_path.name.removesuffix(".tar.gz")
        if scene_dir.exists():
            print(f"[skip] extracted scene exists: {scene_dir}")
            continue
        print(f"[extract] {tar_path} -> {extract_root}")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(extract_root)
    return [extract_root / p.name.removesuffix(".tar.gz") for p in tar_paths]


def extract_nested_video_archives(unit_dir: Path) -> int:
    count = 0
    for video_tar in unit_dir.glob("videos/chunk-*/*.tar.gz"):
        output_dir = video_tar.parent / video_tar.name.removesuffix(".tar.gz")
        if output_dir.exists():
            continue
        print(f"[extract-video] {video_tar}")
        with tarfile.open(video_tar, "r:gz") as tar:
            tar.extractall(video_tar.parent)
        count += 1
    return count


def collect_units(extract_root: Path) -> list[Path]:
    units = []
    for path in sorted(extract_root.rglob("*")):
        if path.is_dir() and is_lerobot_unit(path):
            units.append(path)
    return units


def link_units(units: list[Path], flat_root: Path) -> None:
    flat_root.mkdir(parents=True, exist_ok=True)
    for unit in units:
        rel = unit.relative_to(unit.parents[1]) if len(unit.parents) >= 2 else unit.name
        link_name = "__".join(unit.parts[-2:])
        link_path = flat_root / link_name
        target = unit.resolve()
        if link_path.is_symlink() and link_path.resolve() == target:
            continue
        if link_path.exists() or link_path.is_symlink():
            raise FileExistsError(f"{link_path} already exists and does not point to {target}")
        print(f"[link] {link_path} -> {target} ({rel})")
        os.symlink(target, link_path)


def write_manifest(units: list[Path], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        for unit in units:
            f.write(str(unit.resolve()) + "\n")
    print(f"[manifest] wrote {len(units)} units to {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dir", type=Path, default=Path("traj_data/vln_ce/r2r"))
    parser.add_argument("--extract-root", type=Path, default=Path("traj_data/vln_ce/r2r_extracted"))
    parser.add_argument("--flat-root", type=Path, default=Path("traj_data/r2r"))
    parser.add_argument("--manifest", type=Path, default=Path("traj_data/vln_ce/r2r_units.txt"))
    parser.add_argument("--limit-scenes", type=int, default=None)
    parser.add_argument("--no-extract-videos", action="store_true")
    parser.add_argument("--no-link", action="store_true")
    args = parser.parse_args()

    if not args.src_dir.is_dir():
        raise FileNotFoundError(f"source directory not found: {args.src_dir}")

    extract_scene_archives(args.src_dir, args.extract_root, args.limit_scenes)
    units = collect_units(args.extract_root)
    if not units:
        raise RuntimeError(f"no LeRobot trajectory units found under {args.extract_root}")

    if not args.no_extract_videos:
        video_count = 0
        for unit in units:
            video_count += extract_nested_video_archives(unit)
        print(f"[summary] extracted {video_count} nested video archives")

    if not args.no_link:
        link_units(units, args.flat_root)

    write_manifest(units, args.manifest)
    print(f"[summary] units={len(units)} flat_root={args.flat_root}")


if __name__ == "__main__":
    main()
