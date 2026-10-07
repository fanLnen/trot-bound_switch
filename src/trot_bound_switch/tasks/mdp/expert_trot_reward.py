from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor


def trot_gait_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    previous_group: torch.Tensor,
    contact_threshold: float = 5.0,
    pattern_weight: float = 0.2,
    alternation_weight: float = 0.8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute phase-free trot structure and alternating-support rewards.

    The sensor body order must be ``[FL, FR, RL, RR]``. FL and RR share one
    support group, while FR and RL form the other. A full alternation bonus is
    emitted only when the strict support group changes from one diagonal to the
    other. Invalid contact patterns do not change the remembered group.
    """
    if pattern_weight < 0.0 or alternation_weight < 0.0:
        raise ValueError("Trot reward weights must be non-negative.")
    if abs(pattern_weight + alternation_weight - 1.0) > 1.0e-6:
        raise ValueError("Trot pattern and alternation weights must sum to 1.0.")

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    if forces.shape[1] != 4:
        raise RuntimeError(
            "Trot gait reward expects four feet ordered as [FL, FR, RL, RR]. "
            f"Got {forces.shape[1]} bodies."
        )
    force_norm = torch.linalg.norm(forces, dim=-1)
    contact = force_norm > contact_threshold
    contact_float = contact.to(dtype=force_norm.dtype)

    fl, fr, rl, rr = contact_float.unbind(dim=1)
    diagonal_sync = 1.0 - 0.5 * (torch.abs(fl - rr) + torch.abs(fr - rl))
    group_a = 0.5 * (fl + rr)
    group_b = 0.5 * (fr + rl)
    pattern_reward = diagonal_sync * torch.abs(group_a - group_b)

    fl_contact, fr_contact, rl_contact, rr_contact = contact.unbind(dim=1)
    strict_group_a = fl_contact & rr_contact & ~fr_contact & ~rl_contact
    strict_group_b = fr_contact & rl_contact & ~fl_contact & ~rr_contact
    current_group = torch.full_like(previous_group, -1)
    current_group[strict_group_a] = 0
    current_group[strict_group_b] = 1

    valid_group = current_group >= 0
    alternated = valid_group & (previous_group >= 0) & (current_group != previous_group)
    alternation_reward = alternated.to(dtype=pattern_reward.dtype)
    reward = pattern_weight * pattern_reward + alternation_weight * alternation_reward
    return reward, pattern_reward, alternation_reward, current_group


class adaptive_energy_total(ManagerTermBase):
    """Composite trot reward with episode-level TensorBoard component logging."""

    _LOG_COMPONENT_NAMES = (
        "Reward/total",
        "Reward/pre_aux_total",
        "Reward/aux_multiplier",
        "RewardRaw/linear_tracking",
        "RewardRaw/angular_tracking",
        "RewardRaw/energy",
        "RewardRaw/trot",
        "RewardRaw/trot_pattern",
        "RewardRaw/trot_alternation",
        "RewardContribution/linear_tracking",
        "RewardContribution/angular_tracking",
        "RewardContribution/energy",
        "RewardContribution/trot",
        "RewardContribution/pre_aux_total",
        "RewardContribution/final_total",
        "AuxRaw/orientation",
        "AuxRaw/action_rate",
        "AuxRaw/action_rate_mean_per_dof",
        "AuxRaw/action_rate_rms_per_dof",
        "AuxRaw/joint_limit",
        "AuxRaw/contact",
        "AuxRaw/front_feet_slip",
        "AuxRaw/rear_feet_slip",
        "AuxRaw/hip_abduction",
        "AuxRaw/yaw_tracking",
        "AuxRaw/pair_force_balance",
        "AuxRaw/pair_touchdown_frequency_hz",
        "AuxWeighted/orientation",
        "AuxWeighted/action_rate",
        "AuxWeighted/joint_limit",
        "AuxWeighted/contact",
        "AuxWeighted/front_feet_slip",
        "AuxWeighted/rear_feet_slip",
        "AuxWeighted/hip_abduction",
        "AuxWeighted/yaw_tracking",
        "AuxWeighted/pair_force_balance",
        "AuxWeighted/pair_touchdown_frequency",
        "Aux/total",
        "Aux/multiplier",
    )

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._component_sums = {
            name: torch.zeros(env.num_envs, device=env.device) for name in self._LOG_COMPONENT_NAMES
        }
        self._component_step_count = torch.zeros(env.num_envs, device=env.device)
        self._prev_front_stance = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self._prev_rear_stance = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self._has_stance_history = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self._last_trot_group = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.int8)

    def reset(self, env_ids: torch.Tensor):
        """Log episode means for reset environments, then clear their component buffers."""
        if len(env_ids) == 0:
            return

        counts = self._component_step_count[env_ids]
        valid = counts > 0
        if torch.any(valid):
            valid_env_ids = env_ids[valid]
            valid_counts = counts[valid]
            for name, component_sum in self._component_sums.items():
                episode_means = component_sum[valid_env_ids] / valid_counts
                self._env.extras["log"][name] = episode_means.mean()

        for component_sum in self._component_sums.values():
            component_sum[env_ids] = 0.0
        self._component_step_count[env_ids] = 0.0
        self._prev_front_stance[env_ids] = False
        self._prev_rear_stance[env_ids] = False
        self._has_stance_history[env_ids] = False
        self._last_trot_group[env_ids] = -1

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str = "base_velocity",
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
        front_feet_asset_cfg: SceneEntityCfg | None = None,
        rear_feet_asset_cfg: SceneEntityCfg | None = None,
        front_feet_sensor_cfg: SceneEntityCfg | None = None,
        rear_feet_sensor_cfg: SceneEntityCfg | None = None,
        trot_sensor_cfg: SceneEntityCfg | None = None,

        # motion reward
        sigma_lin: float = 0.25,
        sigma_ang: float = 0.25,
        ang_reward_weight: float = 0.5,
        lin_reward_weight: float = 1.5,

        # energy reward
        alpha_en: float = 0.5,
        sigma_en_lin: float = 1000.0,
        sigma_en_ang: float = 500.0,
        clip_lin: float = 0.2,
        clip_ang: float = 0.2,

        # phase-free trotting contact reward
        trot_reward_weight: float = 0.0,
        trot_contact_threshold: float = 5.0,
        trot_pattern_weight: float = 0.2,
        trot_alternation_weight: float = 0.8,

        # auxiliary penalties
        aux_orientation_weight: float = 0.8,
        aux_action_rate_weight: float = 0.01,
        aux_joint_limit_weight: float = 0.2,
        aux_contact_weight: float = 0.5,
        aux_front_feet_slip_weight: float = 0.0,
        aux_rear_feet_slip_weight: float = 0.0,

        # hip abduction/adduction penalty for straight running
        aux_hip_abduction_weight: float = 0.4,
        hip_abduction_min_cmd_vx: float = 0.0,
        hip_abduction_ramp_width: float = 0.0,
        hip_abduction_cmd_vy_sigma: float = 0.10,
        hip_abduction_cmd_wz_sigma: float = 0.15,
        hip_abduction_target: float = 0.0,

        aux_yaw_tracking_weight: float = 0.2,

        # straight-line pair force-balance penalty
        aux_pair_force_balance_weight: float = 0.0,
        pair_force_balance_min_cmd_vx: float = 1.0,
        pair_force_balance_ramp_width: float = 1.0,
        pair_force_balance_cmd_vy_sigma: float = 0.10,
        pair_force_balance_cmd_wz_sigma: float = 0.15,
        pair_force_balance_contact_threshold: float = 10.0,

        # low step-frequency penalty for straight-line bounding
        aux_pair_touchdown_frequency_weight: float = 0.0,
        pair_touchdown_frequency_min_cmd_vx: float = 1.0,
        pair_touchdown_frequency_ramp_width: float = 1.0,
        pair_touchdown_frequency_cmd_vy_sigma: float = 0.10,
        pair_touchdown_frequency_cmd_wz_sigma: float = 0.15,

        # safety parameters
        contact_threshold: float = 1.0,
        slip_contact_threshold: float = 1.0,
        eps: float = 1.0e-6,
    ) -> torch.Tensor:
        """Compute the composite adaptive-energy and phase-free trot reward.

        The final reward remains:
            R = (R_motion + w_trot * R_trot + alpha_en * R_energy) * exp(-R_aux)
        """

        asset: Articulation = env.scene[asset_cfg.name]
        command = env.command_manager.get_command(command_name)

        device = command.device
        dtype = command.dtype

        # ------------------------------------------------------------
        # 1. Motion reward
        # ------------------------------------------------------------
        base_lin_vel_b = asset.data.root_lin_vel_b
        base_ang_vel_b = asset.data.root_ang_vel_b

        lin_vel_error = torch.sum(
            torch.square(command[:, 0:2] - base_lin_vel_b[:, 0:2]),
            dim=1,
        )

        ang_vel_error = torch.square(command[:, 2] - base_ang_vel_b[:, 2])

        r_lin = torch.exp(-lin_vel_error / sigma_lin)
        r_ang = torch.exp(-ang_vel_error / sigma_ang)

        r_motion = lin_reward_weight * r_lin + ang_reward_weight * r_ang

        # ------------------------------------------------------------
        # 2. Contact sensor data
        # ------------------------------------------------------------
        contact_sensor = env.scene.sensors[sensor_cfg.name]
        net_contact_forces = contact_sensor.data.net_forces_w_history

        # ------------------------------------------------------------
        # 3. Phase-free trotting contact reward
        # ------------------------------------------------------------
        def feet_force_norms(feet_sensor_cfg: SceneEntityCfg | None) -> torch.Tensor | None:
            if feet_sensor_cfg is None:
                return None

            feet_contact_sensor: ContactSensor = env.scene.sensors[feet_sensor_cfg.name]
            forces = feet_contact_sensor.data.net_forces_w[:, feet_sensor_cfg.body_ids, :]
            return torch.norm(forces, dim=-1)

        front_force_norms = feet_force_norms(front_feet_sensor_cfg)
        rear_force_norms = feet_force_norms(rear_feet_sensor_cfg)
        front_contacts = (
            front_force_norms > trot_contact_threshold if front_force_norms is not None else None
        )
        rear_contacts = (
            rear_force_norms > trot_contact_threshold if rear_force_norms is not None else None
        )

        if front_contacts is None or rear_contacts is None:
            front_stance = torch.zeros(command.shape[0], device=device, dtype=torch.bool)
            front_flight = torch.zeros(command.shape[0], device=device, dtype=torch.bool)
            rear_stance = torch.zeros(command.shape[0], device=device, dtype=torch.bool)
            rear_flight = torch.zeros(command.shape[0], device=device, dtype=torch.bool)
        else:
            if front_contacts.shape[1] != 2 or rear_contacts.shape[1] != 2:
                raise RuntimeError(
                    "Trot reward expects exactly two front feet and two rear feet. "
                    f"Got front={front_contacts.shape[1]}, rear={rear_contacts.shape[1]}."
                )

            front_stance = torch.all(front_contacts, dim=1)
            front_flight = torch.all(~front_contacts, dim=1)
            rear_stance = torch.all(rear_contacts, dim=1)
            rear_flight = torch.all(~rear_contacts, dim=1)

        if trot_sensor_cfg is None:
            r_trot = torch.zeros(command.shape[0], device=device, dtype=dtype)
            trot_pattern_reward = torch.zeros_like(r_trot)
            trot_alternation_reward = torch.zeros_like(r_trot)
        else:
            r_trot, trot_pattern_reward, trot_alternation_reward, current_trot_group = trot_gait_reward(
                env=env,
                sensor_cfg=trot_sensor_cfg,
                previous_group=self._last_trot_group,
                contact_threshold=trot_contact_threshold,
                pattern_weight=trot_pattern_weight,
                alternation_weight=trot_alternation_weight,
            )
            valid_trot_group = current_trot_group >= 0
            with torch.no_grad():
                self._last_trot_group[valid_trot_group] = current_trot_group[valid_trot_group]

        # ------------------------------------------------------------
        # 4. Energy reward
        # ------------------------------------------------------------
        joint_vel = asset.data.joint_vel

        # applied_torque is preferred because it is the torque after actuator clipping.
        if hasattr(asset.data, "applied_torque"):
            torque = asset.data.applied_torque
        else:
            torque = asset.data.computed_torque

        joint_power = torch.sum(torch.abs(torque * joint_vel), dim=1)

        lin_speed = torch.clamp(torch.abs(base_lin_vel_b[:, 0]), min=clip_lin)
        yaw_speed = torch.clamp(torch.abs(base_ang_vel_b[:, 2]), min=clip_ang)

        energy_divider = sigma_en_lin * lin_speed + sigma_en_ang * yaw_speed + eps

        # Adaptive energy reward:
        #   lower power -> closer to 1
        #   higher power -> closer to 0
        r_energy = torch.exp(-joint_power / energy_divider)

        # ------------------------------------------------------------
        # 5. Auxiliary penalties
        #    R_aux must be non-negative. Larger means worse.
        # ------------------------------------------------------------

        # 5.1 trunk orientation penalty
        orientation_penalty = torch.sum(
            torch.square(asset.data.projected_gravity_b[:, 0:2]),
            dim=1,
        )

        # 5.2 action rate penalty
        action = env.action_manager.action

        if hasattr(env.action_manager, "prev_action"):
            prev_action = env.action_manager.prev_action
        elif hasattr(env.action_manager, "previous_action"):
            prev_action = env.action_manager.previous_action
        else:
            prev_action = torch.zeros_like(action)

        action_rate_penalty = torch.sum(
            torch.square(action - prev_action),
            dim=1,
        )
        action_rate_mean_per_dof = torch.mean(
            torch.square(action - prev_action),
            dim=1,
        )
        action_rate_rms_per_dof = torch.sqrt(action_rate_mean_per_dof + eps)

        # 5.3 joint position limit penalty
        joint_pos = asset.data.joint_pos
        soft_joint_limits = asset.data.soft_joint_pos_limits

        soft_lower = soft_joint_limits[:, :, 0]
        soft_upper = soft_joint_limits[:, :, 1]

        lower_violation = torch.clamp(soft_lower - joint_pos, min=0.0)
        upper_violation = torch.clamp(joint_pos - soft_upper, min=0.0)

        joint_limit_penalty = torch.sum(
            lower_violation + upper_violation,
            dim=1,
        )




        # ------------------------------------------------------------
        # 5.4 hip abduction/adduction penalty for straight running
        # ------------------------------------------------------------
        # Goal:
        #   During straight forward running, keep all legs close to their sagittal
        #   vertical planes by penalizing hip abduction/adduction deviation.
        #
        # SA16 joint order:
        #   0 FL_hip
        #   1 FR_hip
        #   2 RL_hip
        #   3 RR_hip
        #
        # If your joint order changes, update hip_joint_ids.
        hip_joint_ids = torch.tensor(
            [0, 1, 2, 3],
            device=device,
            dtype=torch.long,
        )

        hip_pos = joint_pos[:, hip_joint_ids]

        hip_abduction_raw_penalty = torch.mean(
            torch.square(hip_pos - hip_abduction_target),
            dim=1,
        )

        # Activate mainly for straight forward commands.
        hip_speed_gate = torch.clamp(
            (command[:, 0] - hip_abduction_min_cmd_vx) / max(hip_abduction_ramp_width, eps),
            min=0.0,
            max=1.0,
        )

        hip_lateral_gate = torch.exp(
            -torch.square(command[:, 1] / (hip_abduction_cmd_vy_sigma + eps))
        )

        hip_turning_gate = torch.exp(
            -torch.square(command[:, 2] / (hip_abduction_cmd_wz_sigma + eps))
        )

        hip_forward_gate = (command[:, 0] > 0.0).float()

        hip_abduction_gate = (
            hip_speed_gate
            * hip_lateral_gate
            * hip_turning_gate
            * hip_forward_gate
        )

        hip_abduction_penalty = hip_abduction_gate * hip_abduction_raw_penalty


        # 5.5 yaw velocity tracking penalty
        # Penalize mismatch between commanded yaw rate and actual yaw rate.
        yaw_tracking_penalty = ang_vel_error

        # 5.6 pair force-balance penalty for straight-line bounding
        def pair_force_imbalance(
            force_norms: torch.Tensor | None,
            bound_stance: torch.Tensor,
        ) -> torch.Tensor:
            if force_norms is None or force_norms.shape[1] < 2:
                return torch.zeros_like(orientation_penalty)
            contacts = force_norms > pair_force_balance_contact_threshold
            pair_stance = torch.all(contacts, dim=1) & bound_stance
            left_force = force_norms[:, 0]
            right_force = force_norms[:, 1]
            imbalance = torch.abs(left_force - right_force) / (left_force + right_force + eps)
            return pair_stance.to(dtype=dtype) * imbalance

        pair_force_speed_gate = torch.clamp(
            (command[:, 0] - pair_force_balance_min_cmd_vx) / max(pair_force_balance_ramp_width, eps),
            min=0.0,
            max=1.0,
        )
        pair_force_lateral_gate = torch.exp(
            -torch.square(command[:, 1] / (pair_force_balance_cmd_vy_sigma + eps))
        )
        pair_force_turning_gate = torch.exp(
            -torch.square(command[:, 2] / (pair_force_balance_cmd_wz_sigma + eps))
        )
        pair_force_forward_gate = (command[:, 0] > 0.0).to(dtype=dtype)
        pair_force_balance_gate = (
            pair_force_speed_gate
            * pair_force_lateral_gate
            * pair_force_turning_gate
            * pair_force_forward_gate
        )

        front_pair_force_imbalance = pair_force_imbalance(front_force_norms, front_stance & rear_flight)
        rear_pair_force_imbalance = pair_force_imbalance(rear_force_norms, rear_stance & front_flight)
        pair_force_balance_penalty = pair_force_balance_gate * (
            front_pair_force_imbalance + rear_pair_force_imbalance
        )

        # 5.7 pair touchdown-frequency penalty for lower-step-frequency bounding
        valid_stance_history = self._has_stance_history.to(dtype=torch.bool)
        front_pair_touchdown = front_stance & ~self._prev_front_stance & valid_stance_history
        rear_pair_touchdown = rear_stance & ~self._prev_rear_stance & valid_stance_history
        pair_touchdown_events = front_pair_touchdown.to(dtype=dtype) + rear_pair_touchdown.to(dtype=dtype)

        pair_touchdown_frequency_gate = torch.clamp(
            (command[:, 0] - pair_touchdown_frequency_min_cmd_vx)
            / max(pair_touchdown_frequency_ramp_width, eps),
            min=0.0,
            max=1.0,
        )
        pair_touchdown_frequency_gate = (
            pair_touchdown_frequency_gate
            * torch.exp(-torch.square(command[:, 1] / (pair_touchdown_frequency_cmd_vy_sigma + eps)))
            * torch.exp(-torch.square(command[:, 2] / (pair_touchdown_frequency_cmd_wz_sigma + eps)))
            * (command[:, 0] > 0.0).to(dtype=dtype)
        )
        pair_touchdown_frequency_penalty = (
            pair_touchdown_frequency_gate * pair_touchdown_events / max(env.step_dt, eps)
        )

        with torch.no_grad():
            self._prev_front_stance[:] = front_stance
            self._prev_rear_stance[:] = rear_stance
            self._has_stance_history[:] = True

        # 5.8 undesired contact penalty
        # This still uses sensor_cfg.body_ids.
        # Make sure sensor_cfg.body_ids points to undesired contact bodies,
        # such as base/thigh/calf, not normal foot bodies.
        contact_force_norm = torch.norm(
            net_contact_forces[:, :, sensor_cfg.body_ids],
            dim=-1,
        )

        max_contact_force = torch.max(contact_force_norm, dim=1)[0]

        contact_penalty = torch.sum(
            max_contact_force > contact_threshold,
            dim=1,
        ).float()

        # 5.9 foot slip penalties
        # Penalize planar foot velocity only while the corresponding foot is in contact.
        def feet_slip_penalty(
            feet_asset_cfg: SceneEntityCfg | None,
            feet_sensor_cfg: SceneEntityCfg | None,
        ) -> torch.Tensor:
            if feet_asset_cfg is None or feet_sensor_cfg is None:
                return torch.zeros_like(contact_penalty)

            feet_asset: Articulation = env.scene[feet_asset_cfg.name]
            feet_contact_sensor: ContactSensor = env.scene.sensors[feet_sensor_cfg.name]
            contacts = (
                feet_contact_sensor.data.net_forces_w_history[:, :, feet_sensor_cfg.body_ids, :]
                .norm(dim=-1)
                .max(dim=1)[0]
                > slip_contact_threshold
            )
            planar_speed = feet_asset.data.body_lin_vel_w[:, feet_asset_cfg.body_ids, :2].norm(dim=-1)
            return torch.sum(planar_speed * contacts, dim=1)

        front_feet_slip_penalty = feet_slip_penalty(front_feet_asset_cfg, front_feet_sensor_cfg)
        rear_feet_slip_penalty = feet_slip_penalty(rear_feet_asset_cfg, rear_feet_sensor_cfg)

        # ------------------------------------------------------------
        # Weighted auxiliary terms and r_aux
        # ------------------------------------------------------------
        aux_orientation_term = aux_orientation_weight * orientation_penalty
        aux_action_rate_term = aux_action_rate_weight * action_rate_penalty
        aux_joint_limit_term = aux_joint_limit_weight * joint_limit_penalty
        aux_contact_term = aux_contact_weight * contact_penalty
        aux_front_feet_slip_term = aux_front_feet_slip_weight * front_feet_slip_penalty
        aux_rear_feet_slip_term = aux_rear_feet_slip_weight * rear_feet_slip_penalty
        aux_hip_abduction_term = aux_hip_abduction_weight * hip_abduction_penalty
        aux_yaw_tracking_term = aux_yaw_tracking_weight * yaw_tracking_penalty
        aux_pair_force_balance_term = aux_pair_force_balance_weight * pair_force_balance_penalty
        aux_pair_touchdown_frequency_term = (
            aux_pair_touchdown_frequency_weight * pair_touchdown_frequency_penalty
        )

        r_aux = (
            aux_orientation_term
            + aux_action_rate_term
            + aux_joint_limit_term
            + aux_contact_term
            + aux_front_feet_slip_term
            + aux_rear_feet_slip_term
            + aux_hip_abduction_term
            + aux_yaw_tracking_term
            + aux_pair_force_balance_term
            + aux_pair_touchdown_frequency_term
        )

        # ------------------------------------------------------------
        # 6. Final reward
        # ------------------------------------------------------------

        linear_contribution = lin_reward_weight * r_lin
        angular_contribution = ang_reward_weight * r_ang
        energy_contribution = alpha_en * r_energy
        trot_contribution = trot_reward_weight * r_trot
        pre_aux_total = r_motion + trot_contribution + energy_contribution
        aux_multiplier = torch.exp(-r_aux)
        reward = pre_aux_total * aux_multiplier

        components = {
            "Reward/total": reward,
            "Reward/pre_aux_total": pre_aux_total,
            "Reward/aux_multiplier": aux_multiplier,
            "RewardRaw/linear_tracking": r_lin,
            "RewardRaw/angular_tracking": r_ang,
            "RewardRaw/energy": r_energy,
            "RewardRaw/trot": r_trot,
            "RewardRaw/trot_pattern": trot_pattern_reward,
            "RewardRaw/trot_alternation": trot_alternation_reward,
            "RewardContribution/linear_tracking": linear_contribution,
            "RewardContribution/angular_tracking": angular_contribution,
            "RewardContribution/energy": energy_contribution,
            "RewardContribution/trot": trot_contribution,
            "RewardContribution/pre_aux_total": pre_aux_total,
            "RewardContribution/final_total": reward,
            "AuxRaw/orientation": orientation_penalty,
            "AuxRaw/action_rate": action_rate_penalty,
            "AuxRaw/action_rate_mean_per_dof": action_rate_mean_per_dof,
            "AuxRaw/action_rate_rms_per_dof": action_rate_rms_per_dof,
            "AuxRaw/joint_limit": joint_limit_penalty,
            "AuxRaw/contact": contact_penalty,
            "AuxRaw/front_feet_slip": front_feet_slip_penalty,
            "AuxRaw/rear_feet_slip": rear_feet_slip_penalty,
            "AuxRaw/hip_abduction": hip_abduction_penalty,
            "AuxRaw/yaw_tracking": yaw_tracking_penalty,
            "AuxRaw/pair_force_balance": pair_force_balance_penalty,
            "AuxRaw/pair_touchdown_frequency_hz": pair_touchdown_frequency_penalty,
            "AuxWeighted/orientation": aux_orientation_term,
            "AuxWeighted/action_rate": aux_action_rate_term,
            "AuxWeighted/joint_limit": aux_joint_limit_term,
            "AuxWeighted/contact": aux_contact_term,
            "AuxWeighted/front_feet_slip": aux_front_feet_slip_term,
            "AuxWeighted/rear_feet_slip": aux_rear_feet_slip_term,
            "AuxWeighted/hip_abduction": aux_hip_abduction_term,
            "AuxWeighted/yaw_tracking": aux_yaw_tracking_term,
            "AuxWeighted/pair_force_balance": aux_pair_force_balance_term,
            "AuxWeighted/pair_touchdown_frequency": aux_pair_touchdown_frequency_term,
            "Aux/total": r_aux,
            "Aux/multiplier": aux_multiplier,
        }
        with torch.no_grad():
            for name, value in components.items():
                self._component_sums[name] += value.detach()
            self._component_step_count += 1.0

        return reward
