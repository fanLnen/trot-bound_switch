"""Transition-only rewards and diagnostics for forced Trot/Bound switching."""

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, RewardTermCfg
from isaaclab.utils import math as math_utils

from .transition_latent_action import TransitionLatentJointPositionAction


class TransitionPitchCost(ManagerTermBase):
    """Penalize squared base pitch only during the 0.6 s transition window."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._raw = torch.zeros(env.num_envs, device=env.device)
        self._step_raw = torch.zeros_like(self._raw)

    def reset(self, env_ids=None):
        self._raw[slice(None) if env_ids is None else env_ids] = 0.0

    def __call__(self, env, action_name):
        action = _get_action_term(env, action_name)
        robot = env.scene[action.cfg.asset_name]
        _, pitch, _ = math_utils.euler_xyz_from_quat(robot.data.root_quat_w)
        self._raw.copy_(torch.square(pitch) * (action.step_transition_progress > 0.0))
        self._step_raw.copy_(self._raw)
        return self._raw

    def logging_metrics(self, weight):
        return _distribution_metrics("TransitionReward/pitch", self._step_raw, weight)


class TransitionPitchRateCost(ManagerTermBase):
    """Penalize pitch rate and own all forced-transition outcome accounting."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._window_steps = int(cfg.params["transition_window_steps"])
        self._recovery_steps = int(cfg.params["recovery_streak_steps"])
        self._recovery_threshold = float(cfg.params["recovery_error_threshold"])
        if self._window_steps != 30 or self._recovery_steps != 10:
            raise ValueError("Transition window/recovery streak must be 30/10 control steps")
        n, device = env.num_envs, env.device
        self._raw = torch.zeros(n, device=device)
        self._step_raw = torch.zeros_like(self._raw)
        self._step_pitch_abs = torch.zeros_like(self._raw)
        self._step_pitch_rate_abs = torch.zeros_like(self._raw)
        self._step_xy_error = torch.zeros_like(self._raw)
        self._step_joint_acc_rms = torch.zeros_like(self._raw)
        self._step_transition_mask = torch.zeros(n, dtype=torch.bool, device=device)
        self._step_direction = torch.zeros(n, dtype=torch.long, device=device)
        self._step_terminated = torch.zeros(n, dtype=torch.bool, device=device)
        self._step_timed_out = torch.zeros(n, dtype=torch.bool, device=device)
        self._seen_id = torch.zeros(n, dtype=torch.long, device=device)
        self._window_active = torch.zeros(n, dtype=torch.bool, device=device)
        self._recovery_active = torch.zeros(n, dtype=torch.bool, device=device)
        self._direction = torch.zeros(n, dtype=torch.long, device=device)
        self._good_streak = torch.zeros(n, dtype=torch.long, device=device)
        self._recovery_age = torch.zeros(n, dtype=torch.long, device=device)
        self._switch_count = torch.zeros(2, dtype=torch.long, device=device)
        self._survival_success = torch.zeros_like(self._switch_count)
        self._survival_failure = torch.zeros_like(self._switch_count)
        self._survival_censored = torch.zeros_like(self._switch_count)
        self._recovered = torch.zeros_like(self._switch_count)
        self._failed_recovery = torch.zeros_like(self._switch_count)
        self._not_recovered_before_switch = torch.zeros_like(self._switch_count)
        self._recovery_censored_timeout = torch.zeros_like(self._switch_count)
        self._recovery_time_sum = torch.zeros(2, dtype=torch.float64, device=device)
        self._steps_since_switch = torch.full((n,), -1, dtype=torch.long, device=device)
        self._failure_time_bin_counts = torch.zeros((2, 4), dtype=torch.long, device=device)
        self._failure_without_prior_switch = torch.zeros((), dtype=torch.long, device=device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._raw[ids] = 0.0
        self._window_active[ids] = False
        self._recovery_active[ids] = False
        self._direction[ids] = 0
        self._good_streak[ids] = 0
        self._recovery_age[ids] = 0
        self._seen_id[ids] = 0
        self._steps_since_switch[ids] = -1

    def __call__(self, env, action_name, command_name, transition_window_steps,
                 recovery_error_threshold, recovery_streak_steps):
        del transition_window_steps, recovery_error_threshold, recovery_streak_steps
        action = _get_action_term(env, action_name)
        robot = env.scene[action.cfg.asset_name]
        _, pitch, _ = math_utils.euler_xyz_from_quat(robot.data.root_quat_w)
        pitch_rate = robot.data.root_ang_vel_b[:, 1]
        command = env.command_manager.get_command(command_name)
        xy_error = torch.linalg.vector_norm(command[:, :2] - robot.data.root_lin_vel_b[:, :2], dim=1)
        joint_acc_rms = torch.sqrt(torch.mean(torch.square(robot.data.joint_acc), dim=1))
        mask = action.step_transition_progress > 0.0
        self._raw.copy_(torch.square(pitch_rate) * mask)
        self._step_raw.copy_(self._raw)
        self._step_pitch_abs.copy_(torch.abs(pitch))
        self._step_pitch_rate_abs.copy_(torch.abs(pitch_rate))
        self._step_xy_error.copy_(xy_error)
        self._step_joint_acc_rms.copy_(joint_acc_rms)
        self._step_transition_mask.copy_(mask)
        self._step_direction.copy_(action.step_switch_direction)
        self._step_terminated.copy_(env.reset_terminated)
        self._step_timed_out.copy_(env.reset_time_outs)
        self._update_outcomes(env, action, xy_error)
        return self._raw

    def _update_outcomes(self, env, action, xy_error):
        transition_id = action.step_transition_id
        new_switch = (transition_id > 0) & (transition_id != self._seen_id)
        if torch.any(new_switch):
            interrupted = new_switch & self._recovery_active
            self._increment(self._not_recovered_before_switch, interrupted)
            self._recovery_active[interrupted] = False
            direction = action.step_switch_direction[new_switch]
            if torch.any((direction < 1) | (direction > 2)):
                raise RuntimeError("Invalid forced-transition direction")
            self._seen_id[new_switch] = transition_id[new_switch]
            self._direction[new_switch] = direction
            self._window_active[new_switch] = True
            self._recovery_active[new_switch] = True
            self._good_streak[new_switch] = 0
            self._recovery_age[new_switch] = 0
            self._steps_since_switch[new_switch] = 0
            self._increment(self._switch_count, new_switch)

        active = self._recovery_active.clone()
        good = active & (xy_error < self._recovery_threshold)
        self._good_streak[good] += 1
        self._good_streak[active & ~good] = 0
        recovered = active & (self._good_streak >= self._recovery_steps)
        if torch.any(recovered):
            first_good_step = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
            first_good_step[recovered] = self._recovery_age[recovered] - self._recovery_steps + 2
            recovery_time = first_good_step.float() * env.step_dt
            self._increment(self._recovered, recovered)
            self._add(self._recovery_time_sum, recovered, recovery_time)
            self._recovery_active[recovered] = False

        terminated, timed_out = env.reset_terminated, env.reset_time_outs
        self._record_failure_time_since_switch(terminated, env.step_dt)
        failed = self._recovery_active & terminated
        censored = self._recovery_active & timed_out & ~terminated
        self._increment(self._failed_recovery, failed)
        self._increment(self._recovery_censored_timeout, censored)
        self._recovery_active[failed | censored] = False
        self._recovery_age[self._recovery_active] += 1

        active_window = self._window_active.clone()
        full = action.transition_age_steps + 1 >= self._window_steps
        survival_failure = active_window & terminated
        survival_success = active_window & ~terminated & full
        survival_censored = active_window & timed_out & ~full & ~terminated
        self._increment(self._survival_failure, survival_failure)
        self._increment(self._survival_success, survival_success)
        self._increment(self._survival_censored, survival_censored)
        self._window_active[survival_failure | survival_success | survival_censored] = False

        active_switch_age = self._steps_since_switch >= 0
        self._steps_since_switch[active_switch_age] += 1

    def _record_failure_time_since_switch(self, terminated, step_dt):
        """Bin non-timeout failures by elapsed time from the latest gait switch."""
        has_prior_switch = self._steps_since_switch >= 0
        self._failure_without_prior_switch += torch.count_nonzero(terminated & ~has_prior_switch)
        valid = terminated & has_prior_switch
        if not torch.any(valid):
            return

        elapsed = (self._steps_since_switch + 1).float() * step_dt
        bins = (
            elapsed < 0.6,
            (elapsed >= 0.6) & (elapsed < 1.5),
            (elapsed >= 1.5) & (elapsed < 3.0),
            elapsed >= 3.0,
        )
        for direction in (1, 2):
            direction_mask = valid & (self._direction == direction)
            for bin_index, bin_mask in enumerate(bins):
                self._failure_time_bin_counts[direction - 1, bin_index] += torch.count_nonzero(
                    direction_mask & bin_mask
                )

    def _increment(self, target, mask):
        for direction in (1, 2):
            target[direction - 1] += torch.count_nonzero(mask & (self._direction == direction))

    def _add(self, target, mask, values):
        for direction in (1, 2):
            selected = mask & (self._direction == direction)
            if torch.any(selected):
                target[direction - 1] += values[selected].sum(dtype=target.dtype)

    def logging_metrics(self, weight):
        metrics = _distribution_metrics("TransitionReward/pitch_rate", self._step_raw, weight)
        for i, name in enumerate(("trot_to_bound", "bound_to_trot")):
            survival_n = self._survival_success[i] + self._survival_failure[i]
            recovery_n = (self._recovered[i] + self._failed_recovery[i]
                          + self._not_recovered_before_switch[i] + self._recovery_censored_timeout[i])
            metrics[f"Transition/{name}_switch_count"] = float(self._switch_count[i])
            metrics[f"Transition/{name}_survival_rate"] = _ratio(self._survival_success[i], survival_n)
            metrics[f"Transition/{name}_survival_censored_count"] = float(self._survival_censored[i])
            metrics[f"Transition/{name}_recovered_count"] = float(self._recovered[i])
            metrics[f"Transition/{name}_failed_recovery_count"] = float(self._failed_recovery[i])
            metrics[f"Transition/{name}_not_recovered_before_next_switch_count"] = float(self._not_recovered_before_switch[i])
            metrics[f"Transition/{name}_recovery_censored_timeout_count"] = float(self._recovery_censored_timeout[i])
            metrics[f"Transition/{name}_recovery_censored_timeout_rate"] = _ratio(self._recovery_censored_timeout[i], recovery_n)
            metrics[f"Transition/{name}_time_to_recover_velocity_mean_s"] = _ratio(self._recovery_time_sum[i], self._recovered[i])
        labels = ("0_0p6_s", "0p6_1p5_s", "1p5_3p0_s", "ge_3p0_s")
        overall = self._failure_time_bin_counts.sum(dim=0)
        overall_total = overall.sum()
        for index, label in enumerate(labels):
            metrics[f"TransitionFailure/overall_{label}_count"] = float(overall[index])
            metrics[f"TransitionFailure/overall_{label}_fraction"] = _ratio(overall[index], overall_total)
        for direction_index, name in enumerate(("trot_to_bound", "bound_to_trot")):
            counts = self._failure_time_bin_counts[direction_index]
            total = counts.sum()
            for index, label in enumerate(labels):
                metrics[f"TransitionFailure/{name}_{label}_count"] = float(counts[index])
                metrics[f"TransitionFailure/{name}_{label}_fraction"] = _ratio(counts[index], total)
        metrics["TransitionFailure/no_prior_switch_count"] = float(self._failure_without_prior_switch)
        return metrics

    def failure_timing_summary(self):
        """Return cumulative failure bins for deterministic evaluation output."""
        labels = ("0.0_to_0.6_s", "0.6_to_1.5_s", "1.5_to_3.0_s", "3.0_s_or_more")

        def summarize(counts):
            total = int(counts.sum().item())
            return {
                "total_with_prior_switch": total,
                "bins": {
                    label: {
                        "count": int(count.item()),
                        "fraction": float(count.item() / total) if total else None,
                    }
                    for label, count in zip(labels, counts)
                },
            }

        return {
            "overall": summarize(self._failure_time_bin_counts.sum(dim=0)),
            "trot_to_bound": summarize(self._failure_time_bin_counts[0]),
            "bound_to_trot": summarize(self._failure_time_bin_counts[1]),
            "no_prior_switch_count": int(self._failure_without_prior_switch.item()),
        }


def _get_action_term(env, action_name):
    action = env.action_manager.get_term(action_name)
    if not isinstance(action, TransitionLatentJointPositionAction):
        raise TypeError(f"Action term {action_name!r} is not TransitionLatentJointPositionAction")
    return action


def _distribution_metrics(prefix, raw, weight):
    weighted = raw * weight
    return {
        f"{prefix}_raw_mean": raw.mean().item(), f"{prefix}_raw_p95": torch.quantile(raw, 0.95).item(),
        f"{prefix}_raw_p99": torch.quantile(raw, 0.99).item(), f"{prefix}_raw_max": raw.max().item(),
        f"{prefix}_weighted_mean": weighted.mean().item(),
        f"{prefix}_weighted_p95": torch.quantile(weighted, 0.95).item(),
        f"{prefix}_weighted_p99": torch.quantile(weighted, 0.99).item(),
        f"{prefix}_weighted_min": weighted.min().item(),
    }


def _ratio(numerator, denominator):
    return float(numerator) / float(denominator) if float(denominator) > 0.0 else 0.0
