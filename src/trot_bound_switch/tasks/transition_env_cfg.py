"""Package-owned SA16 forced Trot/Bound transition environment."""

from dataclasses import MISSING
import logging

import torch

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from trot_bound_switch.assets.robot import SA16_CFG
from trot_bound_switch.paths import BOUND_DECODER_PATH, TROT_DECODER_PATH

from .mdp import (
    LatentActionSmoothness,
    RampVelocityCommandCfg,
    TransitionLatentJointPositionAction,
    TransitionLatentJointPositionActionCfg,
    TransitionPitchCost,
    TransitionPitchRateCost,
    advance_forced_gait_scheduler,
    angular_velocity_tracking,
    gait_command,
    liangan_velocity_command_curriculum,
    linear_velocity_tracking,
    motor_electrical_torque_cost,
    transition_progress,
)

logger = logging.getLogger(__name__)


@configclass
class TransitionSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=(
                f"{ISAACLAB_NUCLEUS_DIR}/Materials/"
                "TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl"
            ),
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = MISSING
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=(
                f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/"
                "kloofendal_43d_clear_puresky_4k.hdr"
            ),
        ),
    )


@configclass
class CommandsCfg:
    base_velocity = base_mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=base_mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.1, 3.5),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=None,
        ),
    )


@configclass
class ActionsCfg:
    joint_pos = TransitionLatentJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        offset=0.0,
        preserve_order=False,
        use_default_offset=True,
        clip=None,
        debug_vis=False,
        trot_decoder_checkpoint=str(TROT_DECODER_PATH),
        bound_decoder_checkpoint=str(BOUND_DECODER_PATH),
        latent_clip=3.0,
        dwell_time_range_s=(3.0, 6.0),
        transition_window_steps=30,
        debug_interval=1000,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # Order is part of the model_1300 contract: 60 stock values + 2 transition values.
        base_lin_vel = ObsTerm(func=base_mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=base_mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=base_mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        velocity_commands = ObsTerm(
            func=base_mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        joint_pos = ObsTerm(func=base_mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=base_mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=base_mdp.last_action)
        gait_command = ObsTerm(func=gait_command, params={"action_name": "joint_pos"})
        transition_progress = ObsTerm(func=transition_progress, params={"action_name": "joint_pos"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    physics_material = EventTerm(
        func=base_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=base_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (0.0, 0.0),
            "operation": "add",
        },
    )
    reset_base = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=base_mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )
    advance_forced_gait_scheduler = EventTerm(
        func=advance_forced_gait_scheduler,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        is_global_time=True,
        params={"action_name": "joint_pos"},
    )


@configclass
class RewardsCfg:
    linear_velocity_tracking = RewTerm(
        func=linear_velocity_tracking,
        weight=1.5,
        params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot")},
    )
    angular_velocity_tracking = RewTerm(
        func=angular_velocity_tracking,
        weight=0.5,
        params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot")},
    )
    latent_action_smoothness = RewTerm(func=LatentActionSmoothness, weight=-0.01)
    joint_torque = RewTerm(
        func=motor_electrical_torque_cost,
        weight=-1.0e-7,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "phase_resistance": 0.57,
            "torque_constant": 0.72,
        },
    )
    joint_acceleration = RewTerm(
        func=base_mdp.joint_acc_l2,
        weight=-3.0e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    termination = RewTerm(func=base_mdp.is_terminated, weight=-1.5)
    transition_pitch = RewTerm(
        func=TransitionPitchCost, weight=-0.5, params={"action_name": "joint_pos"}
    )
    transition_pitch_rate = RewTerm(
        func=TransitionPitchRateCost,
        weight=-0.02,
        params={
            "action_name": "joint_pos",
            "command_name": "base_velocity",
            "transition_window_steps": 30,
            "recovery_error_threshold": 0.25,
            "recovery_streak_steps": 10,
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=base_mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"),
            "threshold": 1.0,
        },
    )


@configclass
class CurriculumCfg:
    velocity_levels = CurrTerm(
        func=liangan_velocity_command_curriculum,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "speed_levels": [(0.1, 3.5)],
            "error_threshold_ratio": 0.15,
            "timeout_threshold": 0.96,
            "min_iterations": 150,
            "stats_window_iterations": 100,
            "high_speed_ratio": 0.8,
            "high_error_threshold_ratio": 0.20,
            "min_high_samples": 2048,
            "high_sample_prob": 0.30,
            "print_interval_iterations": 50,
            "num_steps_per_env": 24,
        },
    )


@configclass
class TransitionEnvCfg(ManagerBasedRLEnvCfg):
    scene: TransitionSceneCfg = TransitionSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.scene.robot = SA16_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class TransitionPlayEnvCfg(TransitionEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.curriculum.velocity_levels = None
        self.commands.base_velocity = RampVelocityCommandCfg(
            asset_name="robot",
            start_vx=3.5,
            target_vx=3.5,
            acceleration=1.0,
            vy=0.0,
            wz=0.0,
        )
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (3.0, -3.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.3)


class TransitionEnv(ManagerBasedRLEnv):
    """Environment with transition diagnostics and strict policy contracts."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        policy_dim = self.observation_manager.group_obs_dim["policy"]
        policy_dim = policy_dim[0] if isinstance(policy_dim, tuple) else policy_dim
        if policy_dim != 62 or self.action_manager.total_action_dim != 16:
            raise RuntimeError(
                f"Expected policy/action dimensions 62/16, got "
                f"{policy_dim}/{self.action_manager.total_action_dim}"
            )
        logger.warning("Transition task runtime dimensions: policy_obs=%d action=16", policy_dim)

    def step(self, action):
        obs, reward, terminated, truncated, extras = super().step(action)
        action_term = self.action_manager.get_term("joint_pos")
        if not isinstance(action_term, TransitionLatentJointPositionAction):
            raise TypeError("Transition task requires TransitionLatentJointPositionAction")
        log = extras.setdefault("log", {})
        log.update(action_term.logging_metrics())
        pitch_cfg = self.reward_manager.get_term_cfg("transition_pitch")
        rate_cfg = self.reward_manager.get_term_cfg("transition_pitch_rate")
        log.update(pitch_cfg.func.logging_metrics(float(pitch_cfg.weight)))
        log.update(rate_cfg.func.logging_metrics(float(rate_cfg.weight)))
        self._add_step_diagnostics(log, action_term, rate_cfg.func, terminated, truncated)
        return obs, reward, terminated, truncated, extras

    def _add_step_diagnostics(self, log, action, diagnostics, terminated, truncated):
        robot = self.scene[action.cfg.asset_name]
        command = self.command_manager.get_command("base_velocity")
        velocity_error = command[:, :2] - robot.data.root_lin_vel_b[:, :2]
        xy_error = torch.linalg.vector_norm(velocity_error, dim=1)
        safe_x = torch.where(command[:, 0] > 0.1, command[:, 0], torch.ones_like(command[:, 0]))
        linear = torch.exp(-torch.sum(torch.square(velocity_error), dim=1) / torch.square(safe_x) / 0.25)
        angular = torch.exp(-torch.square(command[:, 2] - robot.data.root_ang_vel_b[:, 2]) / 0.25)
        valid = ~(terminated | truncated)
        masks = {
            "normal": ~diagnostics._step_transition_mask,
            "transition": diagnostics._step_transition_mask,
            "trot_to_bound": diagnostics._step_transition_mask & (diagnostics._step_direction == 1),
            "bound_to_trot": diagnostics._step_transition_mask & (diagnostics._step_direction == 2),
        }
        for name, mask in masks.items():
            selected = mask & valid
            for key, values in (
                ("linear_tracking", linear),
                ("angular_tracking", angular),
                ("xy_error", xy_error),
                ("pitch_abs", diagnostics._step_pitch_abs),
                ("pitch_rate_abs", diagnostics._step_pitch_rate_abs),
                ("joint_acceleration_rms", diagnostics._step_joint_acc_rms),
                ("decoded_action_jump_l2", action.step_decoded_action_jump_l2),
                ("q_des_jump_l2", action.step_q_des_jump_l2),
                ("decoded_target_jump_raw", action.step_decoded_target_jump_raw),
            ):
                log[f"TransitionStep/{name}_{key}"] = values[selected].mean().item() if torch.any(selected) else 0.0
        log["TransitionStep/base_contact_rate"] = terminated.float().mean().item()
        log["TransitionStep/timeout_rate"] = truncated.float().mean().item()
