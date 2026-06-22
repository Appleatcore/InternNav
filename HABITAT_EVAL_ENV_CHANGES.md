# Habitat Eval Environment Changes

Date: 2026-06-16

Target command:

```bash
python scripts/eval/eval.py --config scripts/eval/configs/habitat_dual_system_cfg.py
```

Target stack:

- Model: InternVLA-N1-DualVLN
- Dataset: VLN-CE/R2R
- Scene data: MP3D-CE
- Simulator: Habitat Sim
- Conda environment: `internvla_habitat_eval`

## Code Changes

### `internnav/habitat_extensions/vln/habitat_vln_evaluator.py`

Changed Habitat model loading so `flash_attn` is optional.

Previous behavior:

- Always used `attn_implementation="flash_attention_2"`.
- The evaluator failed at model load time if `flash_attn` was not installed.

New behavior:

- If `INTERNVLA_ATTN_IMPLEMENTATION` is set, use that value.
- Else, use `flash_attention_2` only when `flash_attn` is importable.
- Else, fall back to PyTorch `sdpa`.

This allows an initial smoke/eval run without installing `flash_attn`.

Added optional `save_all_videos` support for Habitat VLN evaluation.

Previous behavior:

- `save_video=True` only wrote the top-down/RGB video when `success == 1.0`.

New behavior:

- Existing configs keep the same default behavior.
- If `eval_settings["save_all_videos"] = True`, videos are saved for every episode, including failed episodes.

This is used by the single-scene Slurm eval config.

## Single-Scene Video Slurm Run

Added single-scene Habitat eval files:

```text
scripts/eval/configs/vln_r2r_val_unseen_pLe4wQe7qrG.yaml
scripts/eval/configs/habitat_dual_system_single_scene_video_cfg.py
scripts/eval/slurm/eval_habitat_dual_system_single_scene_video_4090.sbatch
```

Default scene:

```text
pLe4wQe7qrG
```

This scene has 18 `val_unseen` R2R episodes.

The Slurm script writes each run to:

```text
logs/habitat/habitat_dual_single_scene_pLe4wQe7qrG_<job_id>
```

It enables:

```text
save_video=True
save_all_videos=True
INTERNVLA_ATTN_IMPLEMENTATION=sdpa
```

## Medium-Scene Habitat Video Run

Added a medium-sized Habitat val_unseen scene config and Slurm runner:

```text
scripts/eval/configs/vln_r2r_val_unseen_X7HyMhZNoso.yaml
scripts/eval/slurm/eval_habitat_dual_system_medium_scene_video_4090.sbatch
```

Default scene:

```text
X7HyMhZNoso
```

This scene has 141 `val_unseen` R2R episodes. The Slurm script writes each run to:

```text
logs/habitat/habitat_dual_medium_scene_X7HyMhZNoso_<job_id>
```

The shared single-scene eval config now accepts `INTERNNAV_HABITAT_CONFIG_PATH`, so the Slurm script can switch scenes without duplicating model/eval settings.

## Environment Notes

Created a dedicated conda environment:

```bash
conda create -n internvla_habitat_eval python=3.9 -y
```

Installed the official Habitat route from the InternNav docs:

```bash
conda install -n internvla_habitat_eval habitat-sim==0.2.4 withbullet headless -c conda-forge -c aihabitat -y
conda run -n internvla_habitat_eval python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

Installed Habitat-Lab / Habitat-Baselines v0.2.4 from:

```text
/srv/shared/home/ycl/workspace/habitat-lab-v0.2.4
```

Installed InternNav core with:

```bash
conda run -n internvla_habitat_eval python -m pip install -e .
```

Installed Habitat/InternVLA dependencies except `flash_attn`.

`depth_camera_filtering` was copied from the existing `internvla_habitat39` environment because direct GitHub clone timed out on the current node.

`diffusion_policy` was exposed through the same local `.pth` path used by `internvla_habitat39`:

```text
/srv/shared/home/ycl/workspace/InternNav/internnav/model/encoder
```

Extracted local MP3D-CE data:

```text
data/mp3d_ce.tar.gz -> data/scene_data/mp3d_ce
```

The extracted scene directory contains 90 `.glb` scene files and is about 21G.

## FlashAttention Policy

`flash_attn` was intentionally not installed during this setup.

Default behavior is now:

```text
flash_attn installed -> flash_attention_2
flash_attn missing   -> sdpa
```

To force one mode:

```bash
INTERNVLA_ATTN_IMPLEMENTATION=sdpa python scripts/eval/eval.py --config scripts/eval/configs/habitat_dual_system_cfg.py
INTERNVLA_ATTN_IMPLEMENTATION=flash_attention_2 python scripts/eval/eval.py --config scripts/eval/configs/habitat_dual_system_cfg.py
```

## Verification

Import check passed in `internvla_habitat_eval`:

```text
torch 2.6.0+cu124
habitat_sim 0.2.4
habitat 0.2.4
habitat_baselines 0.2.4
transformers 4.51.0
flash_attn_found False
attn sdpa
```

Slurm GPU/Habitat check passed on one RTX 4090 D:

```text
cuda_available True
device NVIDIA GeForce RTX 4090 D
R2R split val_unseen
episodes 1839
first_scene data/scene_data/mp3d_ce/mp3d/zsNo4HB9uLZ/zsNo4HB9uLZ.glb
```

Model load check passed on one RTX 4090 D with `attn_implementation='sdpa'`:

```text
processor_loaded Qwen2_5_VLProcessor
model_loaded InternVLAN1ForCausalLM
param_device cuda:0
gpu_mem_gb 15.62
```
