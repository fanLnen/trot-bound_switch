#!/usr/bin/env python3
"""Train a package-owned SA16 Trot or Bound joint-action expert from random initialization."""

from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--gait", choices=("trot", "bound"), required=True)
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--run_name", default="from_scratch")
parser.add_argument("--output_dir", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import trot_bound_switch  # noqa: F401
from trot_bound_switch.tasks import BOUND_EXPERT_TASK_ID, TROT_EXPERT_TASK_ID
from trot_bound_switch.tasks.agents import BoundExpertPPORunnerCfg, TrotExpertPPORunnerCfg
from trot_bound_switch.tasks.expert_env_cfg import BoundExpertEnvCfg, TrotExpertEnvCfg


def main() -> None:
    if args.gait == "trot":
        task_id = TROT_EXPERT_TASK_ID
        env_cfg = TrotExpertEnvCfg()
        agent_cfg = TrotExpertPPORunnerCfg()
    else:
        task_id = BOUND_EXPERT_TASK_ID
        env_cfg = BoundExpertEnvCfg()
        agent_cfg = BoundExpertPPORunnerCfg()

    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    agent_cfg.seed = args.seed
    agent_cfg.resume = False
    agent_cfg.run_name = args.run_name
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations
    if args.device is not None:
        env_cfg.sim.device = args.device
        agent_cfg.device = args.device

    agent_cfg = handle_deprecated_rsl_rl_cfg(
        agent_cfg, importlib.metadata.version("rsl-rl-lib")
    )
    if agent_cfg.resume:
        raise RuntimeError("Expert training entry point is fresh-only and must never resume.")

    log_dir = args.output_dir.expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=False)
    print(f"[INFO] Fresh {args.gait} expert logs: {log_dir}")

    gym_env = gym.make(task_id, cfg=env_cfg)
    env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device)
    dump_yaml(str(log_dir / "params" / "env.yaml"), env_cfg)
    dump_yaml(str(log_dir / "params" / "agent.yaml"), agent_cfg)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    print(f"[RESULT] expert_log_dir={log_dir}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
