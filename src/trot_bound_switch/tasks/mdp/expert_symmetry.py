from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["mirror_symmetry_data_augmentation"]


def _switch_sa16_left_right(x: torch.Tensor) -> torch.Tensor:
    """Left-right mirror transform for SA16 16-DOF three-segment legs.

    This version matches the actual Isaac Lab joint/action order printed by the robot:

        0  FL_hip_joint
        1  FR_hip_joint
        2  RL_hip_joint
        3  RR_hip_joint

        4  FL_thigh_joint
        5  FR_thigh_joint
        6  RL_thigh_joint
        7  RR_thigh_joint

        8  FL_calf_joint
        9  FR_calf_joint
        10 RL_calf_joint
        11 RR_calf_joint

        12 FL_ankle_joint
        13 FR_ankle_joint
        14 RL_ankle_joint
        15 RR_ankle_joint

    Left-right mirror rule:
        FL <-> FR
        RL <-> RR

    Sign rule under sagittal-plane reflection:
        hip   -> opposite sign
        thigh -> same sign
        calf  -> same sign
        ankle -> same sign
    """
    if x.shape[-1] != 16:
        raise ValueError(
            f"Expected 16 dimensions for SA16 mirror symmetry, but got {x.shape[-1]}."
        )

    y = x.clone()

    # ------------------------------------------------------------------
    # Hip joints: left-right swap + sign flip
    # ------------------------------------------------------------------
    y[:, 0] = -x[:, 1]   # FL_hip <- -FR_hip
    y[:, 1] = -x[:, 0]   # FR_hip <- -FL_hip
    y[:, 2] = -x[:, 3]   # RL_hip <- -RR_hip
    y[:, 3] = -x[:, 2]   # RR_hip <- -RL_hip

    # ------------------------------------------------------------------
    # Thigh joints: left-right swap, same sign
    # ------------------------------------------------------------------
    y[:, 4] = x[:, 5]    # FL_thigh <- FR_thigh
    y[:, 5] = x[:, 4]    # FR_thigh <- FL_thigh
    y[:, 6] = x[:, 7]    # RL_thigh <- RR_thigh
    y[:, 7] = x[:, 6]    # RR_thigh <- RL_thigh

    # ------------------------------------------------------------------
    # Calf joints: left-right swap, same sign
    # ------------------------------------------------------------------
    y[:, 8] = x[:, 9]    # FL_calf <- FR_calf
    y[:, 9] = x[:, 8]    # FR_calf <- FL_calf
    y[:, 10] = x[:, 11]  # RL_calf <- RR_calf
    y[:, 11] = x[:, 10]  # RR_calf <- RL_calf

    # ------------------------------------------------------------------
    # Ankle joints: left-right swap, same sign
    # ------------------------------------------------------------------
    y[:, 12] = x[:, 13]  # FL_ankle <- FR_ankle
    y[:, 13] = x[:, 12]  # FR_ankle <- FL_ankle
    y[:, 14] = x[:, 15]  # RL_ankle <- RR_ankle
    y[:, 15] = x[:, 14]  # RR_ankle <- RL_ankle

    return y


def _transform_actions_left_right(actions: torch.Tensor) -> torch.Tensor:
    """Apply left-right mirror transformation to 16-DOF actions."""
    if actions.shape[-1] != 16:
        raise ValueError(
            f"Expected 16 action dimensions for SA16 mirror symmetry, but got {actions.shape[-1]}."
        )

    actions = actions.clone()
    actions[:] = _switch_sa16_left_right(actions[:])
    return actions


def _transform_policy_obs_left_right(env: ManagerBasedRLEnv, obs: torch.Tensor) -> torch.Tensor:
    """Apply left-right symmetry transformation to policy observations.

    Supported layouts:

    Case A, obs_dim = 57:
        base_ang_vel       3
        projected_gravity  3
        velocity_commands  3
        joint_pos          16
        joint_vel          16
        actions            16

    Case B, obs_dim = 60:
        base_lin_vel       3
        base_ang_vel       3
        projected_gravity  3
        velocity_commands  3
        joint_pos          16
        joint_vel          16
        actions            16

    Case C, obs_dim = 73:
        base_ang_vel       3
        projected_gravity  3
        velocity_commands  3
        joint_pos          16
        joint_vel          16
        joint_effort       16
        actions            16

    Case D, obs_dim = 76:
        base_lin_vel       3
        base_ang_vel       3
        projected_gravity  3
        velocity_commands  3
        joint_pos          16
        joint_vel          16
        joint_effort       16
        actions            16
    """
    obs = obs.clone()
    device = obs.device
    dtype = obs.dtype
    obs_dim = obs.shape[-1]

    # Reflection across sagittal plane: y -> -y.
    # Polar vector example:
    #   linear velocity: [vx, vy, vz] -> [vx, -vy, vz]
    #   gravity projection: [gx, gy, gz] -> [gx, -gy, gz]
    polar_sign = torch.tensor([1.0, -1.0, 1.0], device=device, dtype=dtype)

    # Angular vector under reflection:
    #   angular velocity: [wx, wy, wz] -> [-wx, wy, -wz]
    angular_sign = torch.tensor([-1.0, 1.0, -1.0], device=device, dtype=dtype)

    # Command:
    #   [vx, vy, wz] -> [vx, -vy, -wz]
    command_sign = torch.tensor([1.0, -1.0, -1.0], device=device, dtype=dtype)

    if obs_dim == 57:
        # 0:3   base_ang_vel
        # 3:6   projected_gravity
        # 6:9   velocity_commands
        # 9:25  joint_pos
        # 25:41 joint_vel
        # 41:57 actions
        obs[:, 0:3] = obs[:, 0:3] * angular_sign
        obs[:, 3:6] = obs[:, 3:6] * polar_sign
        obs[:, 6:9] = obs[:, 6:9] * command_sign

        obs[:, 9:25] = _switch_sa16_left_right(obs[:, 9:25])
        obs[:, 25:41] = _switch_sa16_left_right(obs[:, 25:41])
        obs[:, 41:57] = _switch_sa16_left_right(obs[:, 41:57])

    elif obs_dim == 60:
        # 0:3   base_lin_vel
        # 3:6   base_ang_vel
        # 6:9   projected_gravity
        # 9:12  velocity_commands
        # 12:28 joint_pos
        # 28:44 joint_vel
        # 44:60 actions
        obs[:, 0:3] = obs[:, 0:3] * polar_sign
        obs[:, 3:6] = obs[:, 3:6] * angular_sign
        obs[:, 6:9] = obs[:, 6:9] * polar_sign
        obs[:, 9:12] = obs[:, 9:12] * command_sign

        obs[:, 12:28] = _switch_sa16_left_right(obs[:, 12:28])
        obs[:, 28:44] = _switch_sa16_left_right(obs[:, 28:44])
        obs[:, 44:60] = _switch_sa16_left_right(obs[:, 44:60])

    elif obs_dim == 73:
        # 0:3   base_ang_vel
        # 3:6   projected_gravity
        # 6:9   velocity_commands
        # 9:25  joint_pos
        # 25:41 joint_vel
        # 41:57 joint_effort
        # 57:73 actions
        obs[:, 0:3] = obs[:, 0:3] * angular_sign
        obs[:, 3:6] = obs[:, 3:6] * polar_sign
        obs[:, 6:9] = obs[:, 6:9] * command_sign

        obs[:, 9:25] = _switch_sa16_left_right(obs[:, 9:25])
        obs[:, 25:41] = _switch_sa16_left_right(obs[:, 25:41])
        obs[:, 41:57] = _switch_sa16_left_right(obs[:, 41:57])
        obs[:, 57:73] = _switch_sa16_left_right(obs[:, 57:73])

    elif obs_dim == 76:
        # 0:3   base_lin_vel
        # 3:6   base_ang_vel
        # 6:9   projected_gravity
        # 9:12  velocity_commands
        # 12:28 joint_pos
        # 28:44 joint_vel
        # 44:60 joint_effort
        # 60:76 actions
        obs[:, 0:3] = obs[:, 0:3] * polar_sign
        obs[:, 3:6] = obs[:, 3:6] * angular_sign
        obs[:, 6:9] = obs[:, 6:9] * polar_sign
        obs[:, 9:12] = obs[:, 9:12] * command_sign

        obs[:, 12:28] = _switch_sa16_left_right(obs[:, 12:28])
        obs[:, 28:44] = _switch_sa16_left_right(obs[:, 28:44])
        obs[:, 44:60] = _switch_sa16_left_right(obs[:, 44:60])
        obs[:, 60:76] = _switch_sa16_left_right(obs[:, 60:76])

    else:
        raise ValueError(
            f"Unsupported policy observation dimension for SA16 mirror symmetry: {obs_dim}. "
            "Please check your PolicyCfg observation order."
        )

    return obs


@torch.no_grad()
def mirror_symmetry_data_augmentation(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """Augment observations and actions by left-right mirror symmetry.

    It creates two versions:
        1. original
        2. left-right mirrored

    The PPO symmetry loss then encourages:
        pi(M(s)) ≈ M(pi(s))

    Important:
        This file assumes the actual SA16 joint/action order is:

            FL_hip, FR_hip, RL_hip, RR_hip,
            FL_thigh, FR_thigh, RL_thigh, RR_thigh,
            FL_calf, FR_calf, RL_calf, RR_calf,
            FL_ankle, FR_ankle, RL_ankle, RR_ankle

        If the robot joint_names change, _switch_sa16_left_right() must be updated.
    """

    # ------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------
    if obs is not None:
        batch_size = obs.batch_size[0]

        # Two transformations: original and left-right mirror.
        obs_aug = obs.repeat(2)

        # Original policy observation.
        obs_aug["policy"][:batch_size] = obs["policy"][:]

        # Mirrored policy observation.
        obs_aug["policy"][batch_size:] = _transform_policy_obs_left_right(
            env.unwrapped,
            obs["policy"],
        )
    else:
        obs_aug = None

    # ------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------
    if actions is not None:
        batch_size = actions.shape[0]

        # Two transformations: original and left-right mirror.
        actions_aug = torch.zeros(
            batch_size * 2,
            actions.shape[1],
            device=actions.device,
            dtype=actions.dtype,
        )

        # Original actions.
        actions_aug[:batch_size] = actions[:]

        # Mirrored actions.
        actions_aug[batch_size:] = _transform_actions_left_right(actions)
    else:
        actions_aug = None

    return obs_aug, actions_aug