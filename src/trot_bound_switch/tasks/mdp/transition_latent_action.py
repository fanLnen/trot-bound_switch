"""Forced Trot/Bound transition scheduler and frozen-decoder action term."""

from __future__ import annotations

import logging

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass

from .latent_joint_action import ActionDecoder

logger = logging.getLogger(__name__)


class TransitionLatentJointPositionAction(JointPositionAction):
    """Route a 16D latent through a command-selected frozen Trot or Bound decoder."""

    cfg: TransitionLatentJointPositionActionCfg
    LATENT_DIM = 16
    JOINT_DIM = 16
    TROT = 0
    BOUND = 1
    TROT_TO_BOUND = 1
    BOUND_TO_TROT = 2

    def __init__(self, cfg: TransitionLatentJointPositionActionCfg, env):
        super().__init__(cfg, env)
        if self._num_joints != self.JOINT_DIM:
            raise RuntimeError(f"Transition action requires 16 joints, resolved {self._num_joints}")
        if not isinstance(self._scale, float) or self._scale != 0.25:
            raise RuntimeError(f"Transition decoder scale must be exactly 0.25, got {self._scale!r}")
        if cfg.clip is not None:
            raise RuntimeError("Joint target clipping is not supported by the transition latent action term")
        if cfg.transition_window_steps != 30:
            raise RuntimeError(f"Transition window must be 30 control steps, got {cfg.transition_window_steps}")

        self._trot_decoder = self._load_decoder(cfg.trot_decoder_checkpoint, "trot")
        self._bound_decoder = self._load_decoder(cfg.bound_decoder_checkpoint, "bound")

        shape = (self.num_envs, self.JOINT_DIM)
        self._processed_actions = torch.zeros(shape, device=self.device)
        self._clipped_latent = torch.zeros(shape, device=self.device)
        self._decoded_action = torch.zeros(shape, device=self.device)
        self._previous_decoded_action = torch.zeros(shape, device=self.device)
        self._previous_joint_target = torch.zeros(shape, device=self.device)
        self._decoded_action_jump_l2 = torch.zeros(self.num_envs, device=self.device)
        self._q_des_jump_l2 = torch.zeros(self.num_envs, device=self.device)
        self._decoded_target_jump_raw = torch.zeros(self.num_envs, device=self.device)

        self._gait = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._dwell_elapsed_s = torch.zeros(self.num_envs, device=self.device)
        self._dwell_duration_s = torch.zeros(self.num_envs, device=self.device)
        self._transition_progress = torch.zeros(self.num_envs, device=self.device)
        self._transition_age_steps = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._last_switch_direction = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._transition_id = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Step snapshots survive resets so post-step diagnostics still describe the action just simulated.
        self._step_gait = self._gait.clone()
        self._step_transition_progress = self._transition_progress.clone()
        self._step_switch_direction = self._last_switch_direction.clone()
        self._step_transition_id = self._transition_id.clone()
        self._step_raw_latent = torch.zeros(shape, device=self.device)
        self._step_clipped_latent = torch.zeros(shape, device=self.device)
        self._step_decoded_action = torch.zeros(shape, device=self.device)
        self._step_joint_target = torch.zeros(shape, device=self.device)
        self._step_decoded_action_jump_l2 = torch.zeros(self.num_envs, device=self.device)
        self._step_q_des_jump_l2 = torch.zeros(self.num_envs, device=self.device)
        self._step_decoded_target_jump_raw = torch.zeros(self.num_envs, device=self.device)

        self._gait_sample_counts = torch.zeros(2, dtype=torch.long, device=self.device)
        self._switch_counts = torch.zeros(2, dtype=torch.long, device=self.device)
        self._process_count = 0

        self.reset()
        resolved_joint_ids = list(range(self._num_joints)) if isinstance(self._joint_ids, slice) else self._joint_ids
        logger.warning(
            "TransitionLatentJointPositionAction joints=%s IDs=%s action_dim=%d dwell=[%.1f, %.1f]s "
            "transition_window=%d steps",
            self._joint_names,
            resolved_joint_ids,
            self.action_dim,
            cfg.dwell_time_range_s[0],
            cfg.dwell_time_range_s[1],
            cfg.transition_window_steps,
        )

    @property
    def raw_latent(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def clipped_latent(self) -> torch.Tensor:
        return self._clipped_latent

    @property
    def decoded_action(self) -> torch.Tensor:
        return self._decoded_action

    @property
    def processed_joint_target(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def gait(self) -> torch.Tensor:
        return self._gait

    @property
    def gait_command_observation(self) -> torch.Tensor:
        return self._gait.float().mul(2.0).sub(1.0).unsqueeze(-1)

    @property
    def transition_progress_observation(self) -> torch.Tensor:
        return self._transition_progress.unsqueeze(-1)

    @property
    def transition_progress(self) -> torch.Tensor:
        return self._transition_progress

    @property
    def transition_age_steps(self) -> torch.Tensor:
        return self._transition_age_steps

    @property
    def last_switch_direction(self) -> torch.Tensor:
        return self._last_switch_direction

    @property
    def transition_id(self) -> torch.Tensor:
        return self._transition_id

    @property
    def step_gait(self) -> torch.Tensor:
        return self._step_gait

    @property
    def step_transition_progress(self) -> torch.Tensor:
        return self._step_transition_progress

    @property
    def step_switch_direction(self) -> torch.Tensor:
        return self._step_switch_direction

    @property
    def step_transition_id(self) -> torch.Tensor:
        return self._step_transition_id

    @property
    def step_raw_latent(self) -> torch.Tensor:
        return self._step_raw_latent

    @property
    def step_clipped_latent(self) -> torch.Tensor:
        return self._step_clipped_latent

    @property
    def step_decoded_action(self) -> torch.Tensor:
        return self._step_decoded_action

    @property
    def step_joint_target(self) -> torch.Tensor:
        return self._step_joint_target

    @property
    def step_decoded_action_jump_l2(self) -> torch.Tensor:
        return self._step_decoded_action_jump_l2

    @property
    def step_q_des_jump_l2(self) -> torch.Tensor:
        return self._step_q_des_jump_l2

    @property
    def step_decoded_target_jump_raw(self) -> torch.Tensor:
        return self._step_decoded_target_jump_raw

    def process_actions(self, actions: torch.Tensor) -> None:
        actions = actions.to(self.device)
        if actions.shape != self._raw_actions.shape:
            raise RuntimeError(f"Expected transition latent shape {tuple(self._raw_actions.shape)}, got {tuple(actions.shape)}")
        if not torch.isfinite(actions).all():
            self._raise_non_finite("policy latent", actions)

        self._raw_actions.copy_(actions)
        self._clipped_latent.copy_(torch.clamp(actions, -self.cfg.latent_clip, self.cfg.latent_clip))
        self._decoded_action.zero_()
        trot_mask = self._gait == self.TROT
        bound_mask = ~trot_mask
        with torch.inference_mode():
            if torch.any(trot_mask):
                self._decoded_action[trot_mask] = self._trot_decoder(self._clipped_latent[trot_mask])
            if torch.any(bound_mask):
                self._decoded_action[bound_mask] = self._bound_decoder(self._clipped_latent[bound_mask])
        if not torch.isfinite(self._decoded_action).all():
            self._raise_non_finite("decoded action", self._decoded_action)

        self._processed_actions.copy_(self._decoded_action * self._scale + self._offset)
        if not torch.isfinite(self._processed_actions).all():
            self._raise_non_finite("processed joint target", self._processed_actions)

        decoded_delta = self._decoded_action - self._previous_decoded_action
        q_des_delta = self._processed_actions - self._previous_joint_target
        self._decoded_action_jump_l2.copy_(torch.linalg.vector_norm(decoded_delta, dim=1))
        self._q_des_jump_l2.copy_(torch.linalg.vector_norm(q_des_delta, dim=1))
        self._decoded_target_jump_raw.copy_(torch.mean(torch.square(q_des_delta), dim=1))

        self._step_gait.copy_(self._gait)
        self._step_transition_progress.copy_(self._transition_progress)
        self._step_switch_direction.copy_(self._last_switch_direction)
        self._step_transition_id.copy_(self._transition_id)
        self._step_raw_latent.copy_(actions)
        self._step_clipped_latent.copy_(self._clipped_latent)
        self._step_decoded_action.copy_(self._decoded_action)
        self._step_joint_target.copy_(self._processed_actions)
        self._step_decoded_action_jump_l2.copy_(self._decoded_action_jump_l2)
        self._step_q_des_jump_l2.copy_(self._q_des_jump_l2)
        self._step_decoded_target_jump_raw.copy_(self._decoded_target_jump_raw)
        self._previous_decoded_action.copy_(self._decoded_action)
        self._previous_joint_target.copy_(self._processed_actions)

        self._gait_sample_counts[0] += torch.count_nonzero(trot_mask)
        self._gait_sample_counts[1] += torch.count_nonzero(bound_mask)
        self._process_count += 1
        if self.cfg.debug_interval > 0 and self._process_count % self.cfg.debug_interval == 0:
            self._print_debug_stats()

    def advance_scheduler(self, env_ids=None) -> None:
        if env_ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        else:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if ids.numel() == 0:
            return

        # ManagerBasedRLEnv invokes interval events after resetting done environments.
        # Do not consume the first dwell step for those freshly reset environments.
        if hasattr(self._env, "reset_buf"):
            ids = ids[~self._env.reset_buf[ids]]
        if ids.numel() == 0:
            return

        active_transition = self._transition_age_steps[ids] >= 0
        active_ids = ids[active_transition]
        if active_ids.numel() > 0:
            self._transition_age_steps[active_ids] += 1
            age = self._transition_age_steps[active_ids]
            self._transition_progress[active_ids] = torch.clamp(
                1.0 - age.float() / float(self.cfg.transition_window_steps), min=0.0
            )
            completed = age >= self.cfg.transition_window_steps
            if torch.any(completed):
                completed_ids = active_ids[completed]
                self._transition_age_steps[completed_ids] = -1
                self._transition_progress[completed_ids] = 0.0

        self._dwell_elapsed_s[ids] += self._env.step_dt
        switch_ids = ids[self._dwell_elapsed_s[ids] >= self._dwell_duration_s[ids]]
        if switch_ids.numel() == 0:
            return

        previous = self._gait[switch_ids].clone()
        self._gait[switch_ids] = 1 - previous
        direction = torch.where(
            previous == self.TROT,
            torch.full_like(previous, self.TROT_TO_BOUND),
            torch.full_like(previous, self.BOUND_TO_TROT),
        )
        self._last_switch_direction[switch_ids] = direction
        self._transition_id[switch_ids] += 1
        self._transition_age_steps[switch_ids] = 0
        self._transition_progress[switch_ids] = 1.0
        self._dwell_elapsed_s[switch_ids] = 0.0
        self._sample_dwell_duration(switch_ids)
        self._switch_counts[0] += torch.count_nonzero(direction == self.TROT_TO_BOUND)
        self._switch_counts[1] += torch.count_nonzero(direction == self.BOUND_TO_TROT)

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
            env_ids = slice(None)
        else:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if ids.numel() == 0:
            return

        self._clipped_latent[env_ids] = 0.0
        self._decoded_action[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._previous_decoded_action[env_ids] = 0.0
        self._previous_joint_target[env_ids] = 0.0
        self._decoded_action_jump_l2[env_ids] = 0.0
        self._q_des_jump_l2[env_ids] = 0.0
        self._decoded_target_jump_raw[env_ids] = 0.0
        self._gait[ids] = torch.randint(0, 2, (ids.numel(),), device=self.device)
        self._dwell_elapsed_s[ids] = 0.0
        self._sample_dwell_duration(ids)
        self._transition_progress[ids] = 0.0
        self._transition_age_steps[ids] = -1
        self._last_switch_direction[ids] = 0
        self._transition_id[ids] = 0

    def logging_metrics(self) -> dict[str, float]:
        samples = max(int(self._gait_sample_counts.sum().item()), 1)
        switches = self._switch_counts.sum()
        raw = self._step_raw_latent
        clipped = self._step_clipped_latent
        return {
            "Transition/trot_fraction": self._gait_sample_counts[0].item() / samples,
            "Transition/bound_fraction": self._gait_sample_counts[1].item() / samples,
            "Transition/trot_to_bound_switch_count": float(self._switch_counts[0].item()),
            "Transition/bound_to_trot_switch_count": float(self._switch_counts[1].item()),
            "Transition/total_switch_count": float(switches.item()),
            "Transition/active_window_fraction": (self._step_transition_progress > 0.0).float().mean().item(),
            "Transition/latent_clip_fraction": (raw != clipped).float().mean().item(),
            "Transition/decoded_action_jump_l2_mean": self._step_decoded_action_jump_l2.mean().item(),
            "Transition/q_des_jump_l2_mean": self._step_q_des_jump_l2.mean().item(),
            "Transition/decoded_target_jump_raw_mean": self._step_decoded_target_jump_raw.mean().item(),
        }

    def _sample_dwell_duration(self, env_ids: torch.Tensor) -> None:
        lower, upper = self.cfg.dwell_time_range_s
        self._dwell_duration_s[env_ids] = torch.empty(env_ids.numel(), device=self.device).uniform_(lower, upper)

    def _load_decoder(self, checkpoint_path: str, expected_gait: str) -> ActionDecoder:
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        gait = str(ckpt.get("gait", "")).lower()
        if gait != expected_gait:
            raise RuntimeError(f"Decoder gait mismatch: expected {expected_gait!r}, got {gait!r}")
        decoder_config = ckpt.get("decoder_config", {})
        dims = (
            int(decoder_config.get("latent_dim", -1)),
            int(decoder_config.get("hidden_dim", -1)),
            int(decoder_config.get("action_dim", -1)),
        )
        if dims != (self.LATENT_DIM, 64, self.JOINT_DIM):
            raise RuntimeError(f"Unexpected {expected_gait} decoder dimensions: {dims}")
        if ckpt.get("action_mapping") != "q_des = q_default + 0.25 * decoded_action":
            raise RuntimeError(f"Unexpected {expected_gait} action mapping: {ckpt.get('action_mapping')!r}")
        actuator = ckpt.get("actuator", {})
        if float(actuator.get("Kp", -1.0)) != 40.0 or float(actuator.get("Kd", -1.0)) != 1.0:
            raise RuntimeError(f"Unexpected {expected_gait} actuator metadata: {actuator!r}")

        decoder = ActionDecoder(latent_dim=dims[0], hidden_dim=dims[1], action_dim=dims[2]).to(self.device)
        decoder.load_state_dict(ckpt["decoder_state_dict"], strict=True)
        decoder.eval()
        for parameter in decoder.parameters():
            parameter.requires_grad_(False)
        logger.warning("Loaded frozen %s transition decoder from %s", gait, checkpoint_path)
        return decoder

    def _print_debug_stats(self) -> None:
        logger.warning(
            "Transition action step=%d gait[trot=%.4f bound=%.4f] progress_mean=%.4f "
            "latent[min=%.4f max=%.4f std=%.4f clip=%.6f] decoded[min=%.4f max=%.4f std=%.4f] "
            "q_des[min=%.4f max=%.4f]",
            self._process_count,
            (self._step_gait == self.TROT).float().mean().item(),
            (self._step_gait == self.BOUND).float().mean().item(),
            self._step_transition_progress.mean().item(),
            self._step_clipped_latent.min().item(),
            self._step_clipped_latent.max().item(),
            self._step_clipped_latent.std(unbiased=False).item(),
            (self._step_raw_latent != self._step_clipped_latent).float().mean().item(),
            self._step_decoded_action.min().item(),
            self._step_decoded_action.max().item(),
            self._step_decoded_action.std(unbiased=False).item(),
            self._step_joint_target.min().item(),
            self._step_joint_target.max().item(),
        )

    @staticmethod
    def _raise_non_finite(name: str, value: torch.Tensor) -> None:
        raise RuntimeError(f"Non-finite {name}: bad_count={torch.count_nonzero(~torch.isfinite(value)).item()}")


def gait_command(env: ManagerBasedRLEnv, action_name: str = "joint_pos") -> torch.Tensor:
    action_term = env.action_manager.get_term(action_name)
    if not isinstance(action_term, TransitionLatentJointPositionAction):
        raise TypeError(f"Action term {action_name!r} is not TransitionLatentJointPositionAction")
    return action_term.gait_command_observation


def transition_progress(env: ManagerBasedRLEnv, action_name: str = "joint_pos") -> torch.Tensor:
    action_term = env.action_manager.get_term(action_name)
    if not isinstance(action_term, TransitionLatentJointPositionAction):
        raise TypeError(f"Action term {action_name!r} is not TransitionLatentJointPositionAction")
    return action_term.transition_progress_observation


def advance_forced_gait_scheduler(env: ManagerBasedRLEnv, env_ids, action_name: str = "joint_pos") -> None:
    action_term = env.action_manager.get_term(action_name)
    if not isinstance(action_term, TransitionLatentJointPositionAction):
        raise TypeError(f"Action term {action_name!r} is not TransitionLatentJointPositionAction")
    action_term.advance_scheduler(env_ids)


@configclass
class TransitionLatentJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = TransitionLatentJointPositionAction
    trot_decoder_checkpoint: str = ""
    bound_decoder_checkpoint: str = ""
    latent_clip: float = 3.0
    dwell_time_range_s: tuple[float, float] = (3.0, 6.0)
    transition_window_steps: int = 30
    debug_interval: int = 1000
