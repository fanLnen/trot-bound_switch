"""Stage-one rewards for the frozen-decoder latent trot task."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg


def linear_velocity_tracking(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Table S3 horizontal velocity tracking reward."""
    asset: Articulation = env.scene[asset_cfg.name]
    command_xy = env.command_manager.get_command(command_name)[:, :2]
    velocity_error = command_xy - asset.data.root_lin_vel_b[:, :2]
    command_x = command_xy[:, 0]

    error_l2_squared = torch.sum(torch.square(velocity_error), dim=1)
    moving = command_x > 0.1
    safe_command_x = torch.where(moving, command_x, torch.ones_like(command_x))
    normalized_error_l2_squared = error_l2_squared / torch.square(safe_command_x)
    tracking_error = torch.where(moving, normalized_error_l2_squared, error_l2_squared)
    return torch.exp(-tracking_error / 0.25)


def angular_velocity_tracking(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Table S3 yaw-rate tracking reward."""
    asset: Articulation = env.scene[asset_cfg.name]
    command_yaw = env.command_manager.get_command(command_name)[:, 2]
    yaw_error_l2_squared = torch.square(command_yaw - asset.data.root_ang_vel_b[:, 2])
    return torch.exp(-yaw_error_l2_squared / 0.25)


class LatentActionSmoothness(ManagerTermBase):
    """Table S3 first- and second-order smoothness over the 16D latent action."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._two_steps_ago = torch.zeros(
            (env.num_envs, env.action_manager.total_action_dim), device=env.device
        )

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._two_steps_ago[env_ids] = 0.0

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        current = env.action_manager.action
        previous = env.action_manager.prev_action
        first_difference = current - previous
        second_difference = current - 2.0 * previous + self._two_steps_ago
        reward = torch.sum(torch.square(first_difference), dim=1)
        reward += 0.15 * torch.sum(torch.square(second_difference), dim=1)
        self._two_steps_ago.copy_(previous)
        return reward


def motor_electrical_torque_cost(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    phase_resistance: float = 0.57,
    torque_constant: float = 0.72,
) -> torch.Tensor:
    """Table S3 torque regularizer, extended from 12 to all 16 selected joints."""
    asset: Articulation = env.scene[asset_cfg.name]
    torque_l2_squared = torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)
    return (phase_resistance / torque_constant**2) * torque_l2_squared
