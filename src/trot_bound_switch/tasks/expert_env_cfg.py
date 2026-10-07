"""Package-owned Trot/Bound expert training and collection environments."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from trot_bound_switch.assets.robot import SA16_CFG
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm

from .mdp.expert_bound_reward import adaptive_energy_total as adaptive_energy_total_reward
from .mdp.expert_curriculum import liangan_velocity_command_curriculum
from .mdp.expert_trot_reward import adaptive_energy_total as adaptive_energy_total_trot_reward

@configclass
class SA16FuXianBoundEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = SA16_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        # scale down the terrains because the robot is small
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        #commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.1, 1)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)


        self.curriculum.liangan_velocity_levels = CurrTerm(
            func=liangan_velocity_command_curriculum,
            params={
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot"),
                "speed_levels": [
                    # (0.1, 1.0),
                    # (0.1, 1.5),
                    # (0.1, 2.0),
                    # (0.1, 2.5),
                    # (0.1, 3.0),
                    # (0.1, 3.5),
                    # (0.1, 4.0),
                    # (0.1, 4.5),
                    # (0.1, 5.0),
                    # (0.1, 6.0),
                    # (0.1, 7.0),

                    # (0.1, 1.0),
                    # (0.1, 1.5),
                    # (0.1, 2.0),
                    # (0.1, 2.5),
                    # (0.1, 3.0),
                     (0.1, 3.5),
                    # (0.1, 4.0),
                    # (0.1, 4.5),
                    # (0.1, 5.0),
                    # (0.1, 6.0),
                    # (0.1, 7.0),
                ],
                "error_threshold_ratio": 0.15,
                "timeout_threshold": 0.96,
                "min_iterations": 150,
                
                "stats_window_iterations": 100,

                # 新增：高速区升档条件
                "high_speed_ratio": 0.8,
                "high_error_threshold_ratio": 0.20,
                "min_high_samples": 2048,

                # 新增：高速段偏置采样
                # 当前 (0.1, 5.0)，上一档 (0.1, 4.5) 时，
                # 大约 30% 从 (4.5, 5.0) 采样
                "high_sample_prob": 0.30,

                "print_interval_iterations": 50,
                "num_steps_per_env": 24,
            },
        )

        # reduce action scale
        self.actions.joint_pos.scale = 0.25

        # event
        self.events.push_robot = None
        self.events.add_base_mass.params["mass_distribution_params"] = (0.0, 0.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base_link"
        self.events.base_external_force_torque = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (0.0, 0.0), "y": (0.0,0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.base_com = None

        # rewards
        # -- paper-style adaptive energy regularization
        self.rewards.adaptive_energy_total = RewTerm(
            func=adaptive_energy_total_reward,
            weight=1.0,
            params={
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=[
                        # ".*_hip",
                        # ".*_thigh",
                        # ".*_upper",
                        # ".*_calf",
                        # ".*_lower",
                        # ".*_ankle",
                        # ".*_foot_rod",
                        ".*_hip_Link",
                        ".*_thigh_Link",
                        ".*_calf_Link",
                        ".*_ankle_Link",
                    ],
                ),
                "front_feet_sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=("FL_foot_Link", "FR_foot_Link"),
                    preserve_order=True,
                ),
                "rear_feet_sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=("RL_foot_Link", "RR_foot_Link"),
                    preserve_order=True,
                ),
                "front_feet_asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=("FL_foot_Link", "FR_foot_Link"),
                    preserve_order=True,
                ),
                "rear_feet_asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=("RL_foot_Link", "RR_foot_Link"),
                    preserve_order=True,
                ),
                # motion reward
                # corresponds to your previous std = sqrt(0.25)
                "sigma_lin": 0.25,
                "sigma_ang": 0.25,
                "ang_reward_weight": 1.5,
                "lin_reward_weight": 2.5,

                # energy reward
                "alpha_en": 1.25,
                # Enable strict front-pair/rear-pair alternating contact from
                # the beginning of training. No command-speed gate is used.
                "bound_reward_weight": 0.3,
                "bound_contact_threshold": 1.0,
                "sigma_en_lin": 1000.0,#高速采用了100取代1000
                "sigma_en_ang": 500.0,#高速采用了50取代500
                "clip_lin": 0.2,
                "clip_ang": 0.2,

                # auxiliary penalties
                "aux_orientation_weight": 0.8,
                "aux_action_rate_weight": 0.01,
                "aux_joint_limit_weight": 0.1,
                "aux_contact_weight": 0.5,
                "aux_front_feet_slip_weight": 0.05,
                "aux_rear_feet_slip_weight": 0.05,
                "aux_hip_abduction_weight": 0.8,
                "aux_pair_force_balance_weight": 0.05,
                "pair_force_balance_min_cmd_vx": 1.0,
                "pair_force_balance_ramp_width": 1.0,
                "pair_force_balance_cmd_vy_sigma": 0.10,
                "pair_force_balance_cmd_wz_sigma": 0.15,
                "pair_force_balance_contact_threshold": 10.0,
                "aux_pair_touchdown_frequency_weight": 0.0,
                "pair_touchdown_frequency_min_cmd_vx": 1.0,
                "pair_touchdown_frequency_ramp_width": 1.0,
                "pair_touchdown_frequency_cmd_vy_sigma": 0.10,
                "pair_touchdown_frequency_cmd_wz_sigma": 0.15,
                # safety parameters
                "contact_threshold": 1.0,
                "slip_contact_threshold": 1.0,
            },
        )

        # ------------------------------------------------------------
        # Disable original additive rewards to avoid double counting.
        # These terms are already included in adaptive_energy_total,
        # or should be removed for paper-style reproduction.
        # ------------------------------------------------------------
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot."
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            # ".*_hip",
            # ".*_thigh",
            # ".*_upper",
            # ".*_calf",
            # ".*_lower",
            # ".*_ankle",
            # ".*_foot_rod",
            ".*_hip_Link",
            ".*_thigh_Link",
            ".*_calf_Link",
            ".*_ankle_Link",
        ]

        

        if hasattr(self.rewards, "track_lin_vel_xy_exp"):
            self.rewards.track_lin_vel_xy_exp = None

        if hasattr(self.rewards, "track_ang_vel_z_exp"):
            self.rewards.track_ang_vel_z_exp= None

        if hasattr(self.rewards, "lin_vel_z_l2"):
            self.rewards.lin_vel_z_l2= None

        if hasattr(self.rewards, "ang_vel_xy_l2"):
            self.rewards.ang_vel_xy_l2 = None

        if hasattr(self.rewards, "dof_torques_l2"):
            self.rewards.dof_torques_l2 = None

        if hasattr(self.rewards, "joint_torques_l2"):
            self.rewards.joint_torques_l2 = None

        if hasattr(self.rewards, "dof_acc_l2"):
            self.rewards.dof_acc_l2= None

        if hasattr(self.rewards, "joint_acc_l2"):
            self.rewards.joint_acc_l2 = None

        if hasattr(self.rewards, "action_rate_l2"):
            self.rewards.action_rate_l2= None

        if hasattr(self.rewards, "feet_air_time"):
            self.rewards.feet_air_time = None

        if hasattr(self.rewards, "flat_orientation_l2"):
            self.rewards.flat_orientation_l2= None

        if hasattr(self.rewards, "dof_pos_limits"):
            self.rewards.dof_pos_limits= None

        if hasattr(self.rewards, "joint_pos_limits"):
            self.rewards.joint_pos_limits= None

        if hasattr(self.rewards, "undesired_contacts"):
            self.rewards.undesired_contacts = None
        self._apply_ground_cfg()

        #terminations
        self.terminations.base_contact.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces",
            body_names="base_link",
        )

    def _apply_ground_cfg(self) -> None:
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None


class SpeedDependentYawVelocityCommand(UniformVelocityCommand):
    """Sample yaw commands whose feasible limit decreases with forward speed."""

    cfg: "SpeedDependentYawVelocityCommandCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._turn_fraction = torch.zeros(self.num_envs, device=self.device)

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        count = len(env_ids)
        turn_fraction = torch.empty(count, device=self.device).uniform_(-1.0, 1.0)
        straight_mask = torch.rand(count, device=self.device) < self.cfg.straight_probability
        turn_fraction[straight_mask] = 0.0
        self._turn_fraction[env_ids] = turn_fraction

    def _update_command(self):
        vx_abs = torch.abs(self.vel_command_b[:, 0])
        speed_for_limit = torch.clamp(vx_abs, min=self.cfg.speed_floor)
        yaw_limit = torch.clamp(
            self.cfg.lateral_acceleration_limit / speed_for_limit,
            max=self.cfg.max_yaw_rate,
        )
        current_iteration = self._env.common_step_counter / max(self.cfg.num_steps_per_iteration, 1)
        ramp_progress = min(current_iteration / max(self.cfg.ramp_iterations, 1), 1.0)
        turn_scale = self.cfg.initial_turn_scale + (1.0 - self.cfg.initial_turn_scale) * ramp_progress
        self.vel_command_b[:, 2] = self._turn_fraction * yaw_limit * turn_scale
        super()._update_command()


@configclass
class SpeedDependentYawVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = SpeedDependentYawVelocityCommand
    straight_probability: float = 0.25
    max_yaw_rate: float = 0.8
    lateral_acceleration_limit: float = 0.8
    speed_floor: float = 1.0
    initial_turn_scale: float = 0.25
    ramp_iterations: int = 300
    num_steps_per_iteration: int = 24


def _replace_bound_reward_with_trot(cfg) -> None:
    reward_cfg = cfg.rewards.adaptive_energy_total
    params = dict(reward_cfg.params)
    params.pop("bound_reward_weight")
    params.pop("bound_contact_threshold")
    params["trot_reward_weight"] = 1.0
    params.update(
        trot_sensor_cfg=SceneEntityCfg(
            "contact_forces",
            body_names=("FL_foot_Link", "FR_foot_Link", "RL_foot_Link", "RR_foot_Link"),
            preserve_order=True,
        ),
        trot_contact_threshold=5.0,
        trot_pattern_weight=0.2,
        trot_alternation_weight=0.8,
    )
    reward_cfg.func = adaptive_energy_total_trot_reward
    reward_cfg.params = params


@configclass
class TrotExpertEnvCfg(SA16FuXianBoundEnvCfg):
    """Fresh Trot expert recipe using the final turning-task configuration."""

    def __post_init__(self):
        super().__post_init__()
        _replace_bound_reward_with_trot(self)
        self.commands.base_velocity = SpeedDependentYawVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(10.0, 10.0),
            rel_standing_envs=0.02,
            heading_command=False,
            debug_vis=True,
            straight_probability=0.25,
            max_yaw_rate=0.8,
            lateral_acceleration_limit=0.8,
            speed_floor=1.0,
            initial_turn_scale=0.25,
            ramp_iterations=300,
            num_steps_per_iteration=24,
            ranges=SpeedDependentYawVelocityCommandCfg.Ranges(
                lin_vel_x=(0.1, 1.0),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(-0.8, 0.8),
                heading=None,
            ),
        )
        self.curriculum.liangan_velocity_levels.params["speed_levels"] = [(0.1, 3.5)]


BoundExpertEnvCfg = SA16FuXianBoundEnvCfg


class CollectionSpeedDependentYawVelocityCommand(SpeedDependentYawVelocityCommand):
    cfg: "CollectionSpeedDependentYawVelocityCommandCfg"

    def _resample_command(self, env_ids: Sequence[int]):
        UniformVelocityCommand._resample_command(self, env_ids)
        count = len(env_ids)
        straight_mask = torch.rand(count, device=self.device) < self.cfg.straight_probability
        turn_magnitude = torch.empty(count, device=self.device).uniform_(self.cfg.min_turn_fraction, 1.0)
        turn_sign = torch.where(
            torch.rand(count, device=self.device) < 0.5,
            -torch.ones(count, device=self.device),
            torch.ones(count, device=self.device),
        )
        turn_fraction = turn_sign * turn_magnitude
        turn_fraction[straight_mask] = 0.0
        self._turn_fraction[env_ids] = turn_fraction


@configclass
class CollectionSpeedDependentYawVelocityCommandCfg(SpeedDependentYawVelocityCommandCfg):
    class_type: type = CollectionSpeedDependentYawVelocityCommand
    min_turn_fraction: float = 0.25


@configclass
class TrotCollectEnvCfg(TrotExpertEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity = CollectionSpeedDependentYawVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(10.0, 10.0),
            rel_standing_envs=0.0,
            heading_command=False,
            debug_vis=False,
            straight_probability=0.80,
            min_turn_fraction=0.25,
            max_yaw_rate=0.8,
            lateral_acceleration_limit=0.8,
            speed_floor=1.0,
            initial_turn_scale=1.0,
            ramp_iterations=1,
            num_steps_per_iteration=24,
            ranges=CollectionSpeedDependentYawVelocityCommandCfg.Ranges(
                lin_vel_x=(0.1, 3.5),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(-0.8, 0.8),
                heading=None,
            ),
        )
        self.curriculum.liangan_velocity_levels.params["speed_levels"] = [(0.1, 3.5)]


@configclass
class BoundCollectEnvCfg(BoundExpertEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity = UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(10.0, 10.0),
            rel_standing_envs=0.0,
            heading_command=False,
            debug_vis=False,
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(0.1, 3.5),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0.0, 0.0),
                heading=None,
            ),
        )
        self.curriculum.liangan_velocity_levels.params["speed_levels"] = [(0.1, 3.5)]
