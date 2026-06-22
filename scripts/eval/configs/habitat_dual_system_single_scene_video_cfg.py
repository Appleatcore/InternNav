import os

from internnav.configs.agent import AgentCfg
from internnav.configs.evaluator import EnvCfg, EvalCfg

OUTPUT_PATH = os.environ.get(
    "INTERNNAV_OUTPUT_PATH",
    "./logs/habitat/habitat_dual_system_single_scene_video_pLe4wQe7qrG",
)
HABITAT_CONFIG_PATH = os.environ.get(
    "INTERNNAV_HABITAT_CONFIG_PATH",
    "scripts/eval/configs/vln_r2r_val_unseen_pLe4wQe7qrG.yaml",
)
MAX_STEPS_PER_EPISODE = int(os.environ.get("INTERNNAV_MAX_STEPS_PER_EPISODE", "500"))
VIS_DEBUG = os.environ.get("INTERNNAV_VIS_DEBUG", "false").lower() in {"1", "true", "yes"}

eval_cfg = EvalCfg(
    agent=AgentCfg(
        model_name='internvla_n1',
        model_settings={
            "mode": "dual_system",
            "model_path": "checkpoints/InternVLA-N1-DualVLN",
            "num_history": 8,
            "resize_w": 384,
            "resize_h": 384,
            "max_new_tokens": 1024,
            "vis_debug": VIS_DEBUG,
            "vis_debug_path": os.path.join(OUTPUT_PATH, "vis_debug"),
        },
    ),
    env=EnvCfg(
        env_type='habitat',
        env_settings={
            'config_path': HABITAT_CONFIG_PATH,
        },
    ),
    eval_type='habitat_vln',
    eval_settings={
        "output_path": OUTPUT_PATH,
        "save_video": True,
        "save_all_videos": True,
        "epoch": 0,
        "max_steps_per_episode": MAX_STEPS_PER_EPISODE,
        "port": os.environ.get("INTERNNAV_DIST_PORT", "2334"),
        "dist_url": "env://",
    },
)
