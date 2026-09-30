#!/usr/bin/env python3
"""Play the package-owned pretrained transition policy."""

import argparse
import importlib.metadata
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="SA-16DOF-FuXian-Trot-Bound-Transition-Play-v0")
parser.add_argument("--checkpoint", type=Path, default=None)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_steps", type=int, default=-1)
parser.add_argument("--video", action="store_true")
parser.add_argument("--video_length", type=int, default=1000)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.video:
    args.enable_cameras = True
simulation_app = AppLauncher(args).app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import trot_bound_switch  # noqa: F401
from trot_bound_switch.diagnostics import assert_package_business_dependencies
from trot_bound_switch.paths import PRETRAINED_POLICY_PATH
from trot_bound_switch.tasks.agents import TransitionPPORunnerCfg
from trot_bound_switch.tasks.transition_env_cfg import TransitionPlayEnvCfg


def main() -> None:
    checkpoint = (args.checkpoint or PRETRAINED_POLICY_PATH).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    assert_package_business_dependencies(checkpoint)

    env_cfg = TransitionPlayEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    if args.device is not None:
        env_cfg.sim.device = args.device
    agent_cfg = TransitionPPORunnerCfg()
    if args.device is not None:
        agent_cfg.device = args.device
    agent_cfg = handle_deprecated_rsl_rl_cfg(
        agent_cfg, importlib.metadata.version("rsl-rl-lib")
    )

    render_mode = "rgb_array" if args.video else None
    gym_env = gym.make(args.task, cfg=env_cfg, render_mode=render_mode)
    if args.video:
        video_dir = Path.cwd() / "videos" / "pretrained"
        gym_env = gym.wrappers.RecordVideo(
            gym_env,
            video_folder=str(video_dir),
            step_trigger=lambda step: step == 0,
            video_length=args.video_length,
            disable_logger=True,
        )
        print(f"[INFO] Recording video to: {video_dir}")
    env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    print(f"[PLAY] Loading checkpoint: {checkpoint}", flush=True)
    runner.load(str(checkpoint))
    print("[PLAY] Checkpoint loaded", flush=True)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    obs = env.get_observations()

    step = 0
    wall_start = time.perf_counter()
    with torch.inference_mode():
        while simulation_app.is_running() and (args.num_steps < 0 or step < args.num_steps):
            obs, _, _, _ = env.step(policy(obs))
            step += 1
    wall_seconds = time.perf_counter() - wall_start
    simulated_seconds = step * env.unwrapped.step_dt
    real_time_factor = simulated_seconds / wall_seconds if wall_seconds > 0.0 else float("inf")
    print(
        f"[PLAY] Completed {step} steps ({simulated_seconds:.2f} simulated seconds), RTF={real_time_factor:.3f}",
        flush=True,
    )
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
