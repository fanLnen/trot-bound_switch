#!/usr/bin/env python3
"""Validate package provenance, task registration, assets, and checkpoints."""

import argparse
import re

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--instantiate", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import gymnasium as gym
import torch

import trot_bound_switch  # noqa: F401
from trot_bound_switch.diagnostics import assert_package_business_dependencies
from trot_bound_switch.paths import (
    BOUND_DECODER_PATH,
    PRETRAINED_POLICY_PATH,
    TROT_DECODER_PATH,
    URDF_PATH,
)
from trot_bound_switch.tasks import PLAY_TASK_ID, TASK_ID
from trot_bound_switch.tasks.transition_env_cfg import TransitionPlayEnvCfg


def main() -> None:
    assert_package_business_dependencies(PRETRAINED_POLICY_PATH)
    for task_id in (TASK_ID, PLAY_TASK_ID):
        spec = gym.spec(task_id)
        print(f"[SMOKE] registered task: {spec.id} -> {spec.entry_point}")

    policy = torch.load(PRETRAINED_POLICY_PATH, map_location="cpu", weights_only=False)
    if tuple(policy["actor_state_dict"]["mlp.0.weight"].shape) != (512, 62):
        raise RuntimeError("Unexpected policy observation shape")
    if tuple(policy["actor_state_dict"]["mlp.6.weight"].shape) != (16, 128):
        raise RuntimeError("Unexpected policy action shape")
    for expected, path in (("trot", TROT_DECODER_PATH), ("bound", BOUND_DECODER_PATH)):
        decoder = torch.load(path, map_location="cpu", weights_only=False)
        if decoder.get("gait") != expected:
            raise RuntimeError(f"Unexpected decoder gait in {path}")
        print(f"[SMOKE] loaded {expected} decoder: {path}")
    print(f"[SMOKE] loaded policy iter={policy.get('iter')}: {PRETRAINED_POLICY_PATH}")

    urdf = URDF_PATH.read_text(encoding="utf-8")
    refs = sorted(set(re.findall(r'package://([^<"]+)', urdf)))
    resource_root = URDF_PATH.parents[2]
    missing = [ref for ref in refs if not (resource_root / ref).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing URDF mesh resources: {missing}")
    print(f"[SMOKE] URDF mesh closure: {len(refs)} files")

    if args.instantiate:
        cfg = TransitionPlayEnvCfg()
        cfg.scene.num_envs = 2
        env = gym.make(PLAY_TASK_ID, cfg=cfg)
        env.reset()
        env.close()
        print("[SMOKE] environment instantiated and reset")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
