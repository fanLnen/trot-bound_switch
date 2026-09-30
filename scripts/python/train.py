#!/usr/bin/env python3
"""Fresh PPO training entry point for the package-owned transition task."""

import argparse
import os
from datetime import datetime
import importlib.metadata
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="SA-16DOF-FuXian-Trot-Bound-Transition-v0")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--max_iterations", type=int, default=1351)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--run_name", default="forced_transition_formal_4096x1351_reproduction")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import gymnasium as gym
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.storage import RolloutStorage

from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import trot_bound_switch  # noqa: F401
from trot_bound_switch.diagnostics import assert_package_business_dependencies
from trot_bound_switch.paths import PRETRAINED_POLICY_PATH
from trot_bound_switch.tasks.agents import TransitionPPORunnerCfg
from trot_bound_switch.tasks.transition_env_cfg import TransitionEnvCfg


def _install_runner_init_trace() -> None:
    """Print runner-construction boundaries when explicitly requested."""

    def wrap_init(cls: type, label: str) -> None:
        original_init = cls.__init__

        def traced_init(self, *init_args, **init_kwargs):
            print(f"[TRACE] {label} init begin", flush=True)
            original_init(self, *init_args, **init_kwargs)
            print(f"[TRACE] {label} init end", flush=True)

        cls.__init__ = traced_init

    wrap_init(RolloutStorage, "RolloutStorage")
    wrap_init(PPO, "PPO")
    wrap_init(torch.optim.Adam, "Adam")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    assert_package_business_dependencies(PRETRAINED_POLICY_PATH)

    env_cfg = TransitionEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    if args.device is not None:
        env_cfg.sim.device = args.device

    agent_cfg = TransitionPPORunnerCfg()
    agent_cfg.seed = args.seed
    agent_cfg.max_iterations = args.max_iterations
    agent_cfg.run_name = args.run_name
    agent_cfg.resume = False
    if args.device is not None:
        agent_cfg.device = args.device
    agent_cfg = handle_deprecated_rsl_rl_cfg(
        agent_cfg, importlib.metadata.version("rsl-rl-lib")
    )
    if agent_cfg.resume:
        raise RuntimeError("train.py is fresh-only and must never resume a checkpoint")

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = repo_root / "logs" / "rsl_rl" / agent_cfg.experiment_name / f"{stamp}_{args.run_name}"
    log_dir.mkdir(parents=True, exist_ok=False)
    print(f"[INFO] Fresh training logs: {log_dir}")

    gym_env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
    if os.environ.get("TBS_TRACE_RUNNER_INIT") == "1":
        _install_runner_init_trace()
    print("[TRAIN] Creating RSL-RL runner...", flush=True)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device)
    print("[TRAIN] RSL-RL runner created", flush=True)
    dump_yaml(str(log_dir / "params" / "env.yaml"), env_cfg)
    dump_yaml(str(log_dir / "params" / "agent.yaml"), agent_cfg)
    print(f"[TRAIN] Starting {agent_cfg.max_iterations} learning iteration(s)", flush=True)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    print("[TRAIN] Learning completed", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
