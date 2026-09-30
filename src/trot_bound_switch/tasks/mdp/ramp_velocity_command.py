from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass


class RampVelocityCommand(CommandTerm):
    """Velocity command with a constant acceleration ramp.

    The command is:
        vx(t) = start_vx + sign(target_vx - start_vx) * acceleration * t

    It is clipped once it reaches target_vx.

    Output command format:
        [vx, vy, wz]

    This command is mainly intended for PLAY/evaluation, not for normal randomized training.
    """

    cfg: RampVelocityCommandCfg

    def __init__(self, cfg: RampVelocityCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self._command = torch.zeros(self.num_envs, 3, device=self.device)

        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _resample_command(self, env_ids: torch.Tensor):
        # Nothing random to resample.
        # The ramp is generated from episode time.
        self._command[env_ids, 0] = self.cfg.start_vx
        self._command[env_ids, 1] = self.cfg.vy
        self._command[env_ids, 2] = self.cfg.wz

    def _update_command(self):
        # episode_length_buf resets automatically when the env resets.
        # Therefore the ramp also restarts after reset.
        t = self._env.episode_length_buf.to(device=self.device, dtype=torch.float32) * self._env.step_dt

        start_vx = float(self.cfg.start_vx)
        target_vx = float(self.cfg.target_vx)
        accel = abs(float(self.cfg.acceleration))

        if target_vx >= start_vx:
            vx = start_vx + accel * t
            vx = torch.clamp(vx, max=target_vx)
        else:
            vx = start_vx - accel * t
            vx = torch.clamp(vx, min=target_vx)

        self._command[:, 0] = vx
        self._command[:, 1] = float(self.cfg.vy)
        self._command[:, 2] = float(self.cfg.wz)

    def _update_metrics(self):
        robot = self._env.scene[self.cfg.asset_name]

        base_lin_vel_b = robot.data.root_lin_vel_b
        base_ang_vel_b = robot.data.root_ang_vel_b

        self.metrics["error_vel_xy"] = torch.norm(
            self._command[:, 0:2] - base_lin_vel_b[:, 0:2],
            dim=1,
        )
        self.metrics["error_vel_yaw"] = torch.abs(
            self._command[:, 2] - base_ang_vel_b[:, 2]
        )


@configclass
class RampVelocityCommandCfg(CommandTermCfg):
    """Configuration for RampVelocityCommand."""

    class_type: type = RampVelocityCommand

    asset_name: str = "robot"

    # ramp parameters
    start_vx: float = 0.0
    target_vx: float = 3.0
    acceleration: float = 0.5

    # fixed lateral and yaw commands
    vy: float = 0.0
    wz: float = 0.0

    # keep command term active without random resampling
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

    # optional
    debug_vis: bool = False