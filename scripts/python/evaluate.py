#!/usr/bin/env python3
"""Deterministically evaluate the forced Trot/Bound transition task."""

import argparse
import importlib.metadata
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="SA-16DOF-FuXian-Trot-Bound-Transition-v0")
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--num_steps", type=int, default=1000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--video", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.video:
    args.enable_cameras = True
simulation_app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import trot_bound_switch  # noqa: F401
from trot_bound_switch.diagnostics import assert_package_business_dependencies
from trot_bound_switch.tasks import TASK_ID
from trot_bound_switch.tasks.agents import TransitionPPORunnerCfg
from trot_bound_switch.tasks.mdp import TransitionLatentJointPositionAction
from trot_bound_switch.tasks.transition_env_cfg import TransitionEnvCfg

WEIGHTS = {
    "linear_velocity_tracking": 1.5,
    "angular_velocity_tracking": 0.5,
    "latent_action_smoothness": -0.01,
    "joint_torque": -1.0e-7,
    "joint_acceleration": -3.0e-7,
    "termination": -1.5,
    "transition_pitch": -0.5,
    "transition_pitch_rate": -0.02,
}
DIRECTIONS = ((1, "trot_to_bound"), (2, "bound_to_trot"))


class TelemetryOverlay(gym.Wrapper):
    """Draw forced-transition state on recorded rollout frames."""

    def __init__(self, env):
        super().__init__(env)
        try:
            self._font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        except OSError:
            self._font = ImageFont.load_default()

    def render(self):
        rendered = self.env.render()
        if rendered is None:
            return None
        if isinstance(rendered, list):
            return [self._draw(frame) for frame in rendered]
        return self._draw(rendered)

    def _draw(self, frame):
        base = self.unwrapped
        action = base.action_manager.get_term("joint_pos")
        gait = "Trot" if int(action.gait[0].item()) == action.TROT else "Bound"
        switch_age = "n/a"
        if int(action.transition_id[0].item()) > 0:
            switch_age = f"{float(action._dwell_elapsed_s[0].item()):.2f} s"
        command_vx = float(base.command_manager.get_command("base_velocity")[0, 0].item())
        actual_vx = float(base.scene[action.cfg.asset_name].data.root_lin_vel_b[0, 0].item())

        image = Image.fromarray(np.asarray(frame).astype(np.uint8)).convert("RGBA")
        panel = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(panel)
        draw.rectangle((16, 16, 440, 146), fill=(0, 0, 0, 178))
        lines = (
            f"Gait command: {gait}",
            f"Time since last switch: {switch_age}",
            f"Command vx: {command_vx:+.2f} m/s   Actual vx: {actual_vx:+.2f} m/s",
        )
        for index, line in enumerate(lines):
            draw.text((30, 27 + 38 * index), line, font=self._font, fill=(255, 255, 255, 255))
        return np.asarray(Image.alpha_composite(image, panel).convert("RGB"))


def stats(values):
    values = torch.as_tensor(values, dtype=torch.float32).reshape(-1).cpu()
    if values.numel() == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "p50": None,
                "p95": None, "p99": None, "max": None}
    q = torch.quantile(values, torch.tensor([0.50, 0.95, 0.99]))
    return {"count": values.numel(), "mean": values.mean().item(),
            "std": values.std(unbiased=False).item(), "min": values.min().item(),
            "p50": q[0].item(), "p95": q[1].item(), "p99": q[2].item(),
            "max": values.max().item()}


def cat(parts):
    return torch.cat(parts) if parts else torch.empty(0)


def main() -> None:
    checkpoint = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)
    assert_package_business_dependencies(checkpoint)
    if args.task != TASK_ID:
        raise ValueError(f"This repository evaluator only supports {TASK_ID}, got {args.task}")
    env_cfg = TransitionEnvCfg()
    agent_cfg = TransitionPPORunnerCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    if args.video:
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.eye = (3.0, -3.0, 2.0)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.3)
    if args.device is not None:
        env_cfg.sim.device = args.device
        agent_cfg.device = args.device
    agent_cfg = handle_deprecated_rsl_rl_cfg(
        agent_cfg, importlib.metadata.version("rsl-rl-lib")
    )
    gym_env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
    if args.video:
        gym_env = TelemetryOverlay(gym_env)
        gym_env = gym.wrappers.RecordVideo(
            gym_env, video_folder=str(output_dir / "video"), step_trigger=lambda step: step == 0,
            video_length=args.num_steps, disable_logger=True
        )
    env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
    base = env.unwrapped
    action_term = base.action_manager.get_term("joint_pos")
    if not isinstance(action_term, TransitionLatentJointPositionAction):
        raise TypeError(type(action_term))
    obs_dim = base.observation_manager.group_obs_dim["policy"][0]
    if obs_dim != 62 or base.action_manager.total_action_dim != 16:
        raise RuntimeError(f"Expected observation/action dimensions 62/16, got {obs_dim}/{base.action_manager.total_action_dim}")
    for name, expected in WEIGHTS.items():
        actual = float(base.reward_manager.get_term_cfg(name).weight)
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12):
            raise RuntimeError(f"Unexpected {name} weight: {actual}")

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=base.device)
    obs = env.get_observations()
    n, device = base.num_envs, base.device

    overall = {key: [] for key in (
        "linear_tracking", "angular_tracking", "xy_error", "yaw_error", "joint_acc_rms",
        "latent", "decoded_action", "q_des", "pitch_raw", "pitch_weighted",
        "pitch_rate_raw", "pitch_rate_weighted", "pitch_raw_transition",
        "pitch_weighted_transition", "pitch_rate_raw_transition", "pitch_rate_weighted_transition",
        "raw_latent", "latent_clipped", "selected_gait"
    )}
    rewards = {name: {"raw": [], "weighted": []} for name in WEIGHTS}
    phase = {name: {key: [] for key in ("linear_tracking", "angular_tracking", "xy_error",
             "joint_acc_rms", "decoded_jump_l2", "q_des_jump_l2")} for name in ("normal", "transition")}
    direction_steps = {name: {key: [] for key in ("xy_error", "pitch_abs", "pitch_rate_abs",
                       "joint_acc_rms", "decoded_jump_l2", "q_des_jump_l2")} for _, name in DIRECTIONS}
    transition_records = {name: {key: [] for key in ("pitch_peak", "pitch_rate_peak",
                          "xy_error_mean", "decoded_jump_peak", "q_des_jump_peak",
                          "joint_acc_rms_mean")} for _, name in DIRECTIONS}
    outcomes = {name: {key: 0 for key in ("switches", "survived", "survival_failed",
                "survival_censored", "recovered", "failed_recovery",
                "not_recovered_before_next_switch", "recovery_censored_timeout")} | {"recovery_times": []}
                for _, name in DIRECTIONS}

    seen_id = torch.zeros(n, dtype=torch.long, device=device)
    active_direction = torch.zeros(n, dtype=torch.long, device=device)
    window_active = torch.zeros(n, dtype=torch.bool, device=device)
    window_age = torch.zeros(n, dtype=torch.long, device=device)
    recovery_active = torch.zeros(n, dtype=torch.bool, device=device)
    recovery_age = torch.zeros(n, dtype=torch.long, device=device)
    good_streak = torch.zeros(n, dtype=torch.long, device=device)
    peaks = {key: torch.zeros(n, device=device) for key in
             ("pitch", "pitch_rate", "xy_sum", "decoded_jump", "q_des_jump", "joint_acc_sum")}
    episode_steps = torch.zeros(n, dtype=torch.long, device=device)
    completed_lengths, base_contacts, timeouts = [], [], []

    def increment(mask, key):
        for direction, name in DIRECTIONS:
            outcomes[name][key] += int(torch.count_nonzero(mask & (active_direction == direction)))

    def close_windows(mask):
        for direction, name in DIRECTIONS:
            selected = mask & (active_direction == direction)
            if torch.any(selected):
                count = window_age[selected].float() + 1.0
                transition_records[name]["pitch_peak"].append(peaks["pitch"][selected].cpu())
                transition_records[name]["pitch_rate_peak"].append(peaks["pitch_rate"][selected].cpu())
                transition_records[name]["xy_error_mean"].append((peaks["xy_sum"][selected] / count).cpu())
                transition_records[name]["decoded_jump_peak"].append(peaks["decoded_jump"][selected].cpu())
                transition_records[name]["q_des_jump_peak"].append(peaks["q_des_jump"][selected].cpu())
                transition_records[name]["joint_acc_rms_mean"].append((peaks["joint_acc_sum"][selected] / count).cpu())

    with torch.inference_mode():
        for step in range(args.num_steps):
            command = base.command_manager.get_command("base_velocity").detach().clone()
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            done = dones.bool()
            terminated = base.reset_terminated.detach().clone()
            timed_out = base.reset_time_outs.detach().clone()
            robot = base.scene["robot"]
            error = command[:, :2] - robot.data.root_lin_vel_b[:, :2]
            xy_error = torch.linalg.vector_norm(error, dim=1)
            safe_x = torch.where(command[:, 0] > 0.1, command[:, 0], torch.ones_like(command[:, 0]))
            linear = torch.exp(-(torch.sum(torch.square(error), dim=1) / torch.square(safe_x)) / 0.25)
            yaw_error = torch.abs(command[:, 2] - robot.data.root_ang_vel_b[:, 2])
            angular = torch.exp(-torch.square(yaw_error) / 0.25)
            joint_acc = torch.sqrt(torch.mean(torch.square(robot.data.joint_acc), dim=1))
            pitch_term = base.reward_manager.get_term_cfg("transition_pitch").func
            rate_term = base.reward_manager.get_term_cfg("transition_pitch_rate").func
            transition_mask = rate_term._step_transition_mask.detach().clone()
            direction = action_term.step_switch_direction.detach().clone()
            transition_id = action_term.step_transition_id.detach().clone()

            values = {"linear_tracking": linear, "angular_tracking": angular, "xy_error": xy_error,
                      "yaw_error": yaw_error, "joint_acc_rms": joint_acc,
                      "latent": action_term.step_clipped_latent.reshape(-1),
                      "decoded_action": action_term.step_decoded_action.reshape(-1),
                      "q_des": action_term.step_joint_target.reshape(-1),
                      "pitch_raw": pitch_term._step_raw,
                      "pitch_weighted": pitch_term._step_raw * WEIGHTS["transition_pitch"],
                      "pitch_rate_raw": rate_term._step_raw,
                      "pitch_rate_weighted": rate_term._step_raw * WEIGHTS["transition_pitch_rate"],
                      "pitch_raw_transition": pitch_term._step_raw[transition_mask],
                      "pitch_weighted_transition": pitch_term._step_raw[transition_mask] * WEIGHTS["transition_pitch"],
                      "pitch_rate_raw_transition": rate_term._step_raw[transition_mask],
                      "pitch_rate_weighted_transition": rate_term._step_raw[transition_mask] * WEIGHTS["transition_pitch_rate"],
                      "raw_latent": actions.reshape(-1),
                      "latent_clipped": (actions != action_term.step_clipped_latent).float().reshape(-1),
                      "selected_gait": action_term.step_gait.float()}
            for key, value in values.items():
                overall[key].append(value.cpu())
            for name, weight in WEIGHTS.items():
                index = base.reward_manager._term_names.index(name)
                weighted = base.reward_manager._step_reward[:, index]
                rewards[name]["weighted"].append(weighted.cpu())
                rewards[name]["raw"].append((weighted / weight).cpu())
            for name, mask in (("normal", ~transition_mask), ("transition", transition_mask)):
                selected = mask & ~done
                for key, value in (("linear_tracking", linear), ("angular_tracking", angular),
                                   ("xy_error", xy_error), ("joint_acc_rms", joint_acc),
                                   ("decoded_jump_l2", action_term.step_decoded_action_jump_l2),
                                   ("q_des_jump_l2", action_term.step_q_des_jump_l2)):
                    phase[name][key].append(value[selected].cpu())
            for direction_id, name in DIRECTIONS:
                selected = transition_mask & (direction == direction_id) & ~done
                for key, value in (("xy_error", xy_error), ("pitch_abs", rate_term._step_pitch_abs),
                                   ("pitch_rate_abs", rate_term._step_pitch_rate_abs),
                                   ("joint_acc_rms", joint_acc),
                                   ("decoded_jump_l2", action_term.step_decoded_action_jump_l2),
                                   ("q_des_jump_l2", action_term.step_q_des_jump_l2)):
                    direction_steps[name][key].append(value[selected].cpu())

            new_switch = (transition_id > 0) & (transition_id != seen_id)
            if torch.any(new_switch):
                interrupted = new_switch & recovery_active
                increment(interrupted, "not_recovered_before_next_switch")
                recovery_active[interrupted] = False
                seen_id[new_switch] = transition_id[new_switch]
                active_direction[new_switch] = direction[new_switch]
                window_active[new_switch] = True
                recovery_active[new_switch] = True
                window_age[new_switch] = 0
                recovery_age[new_switch] = 0
                good_streak[new_switch] = 0
                for value in peaks.values():
                    value[new_switch] = 0.0
                increment(new_switch, "switches")

            active_window = window_active.clone()
            peaks["pitch"] = torch.where(active_window, torch.maximum(peaks["pitch"], rate_term._step_pitch_abs), peaks["pitch"])
            peaks["pitch_rate"] = torch.where(active_window, torch.maximum(peaks["pitch_rate"], rate_term._step_pitch_rate_abs), peaks["pitch_rate"])
            peaks["decoded_jump"] = torch.where(active_window, torch.maximum(peaks["decoded_jump"], action_term.step_decoded_action_jump_l2), peaks["decoded_jump"])
            peaks["q_des_jump"] = torch.where(active_window, torch.maximum(peaks["q_des_jump"], action_term.step_q_des_jump_l2), peaks["q_des_jump"])
            peaks["xy_sum"][active_window] += xy_error[active_window]
            peaks["joint_acc_sum"][active_window] += joint_acc[active_window]
            survival_failed = active_window & terminated
            survival_complete = active_window & ~terminated & (window_age + 1 >= 30)
            survival_censored = active_window & timed_out & ~terminated & (window_age + 1 < 30)
            increment(survival_failed, "survival_failed")
            increment(survival_complete, "survived")
            increment(survival_censored, "survival_censored")
            close_windows(survival_failed | survival_complete | survival_censored)
            window_active[survival_failed | survival_complete | survival_censored] = False
            window_age[window_active] += 1

            active_recovery = recovery_active.clone()
            good = active_recovery & (xy_error < 0.25)
            good_streak[good] += 1
            good_streak[active_recovery & ~good] = 0
            recovered = active_recovery & (good_streak >= 10)
            for direction_id, name in DIRECTIONS:
                selected = recovered & (active_direction == direction_id)
                if torch.any(selected):
                    outcomes[name]["recovered"] += int(torch.count_nonzero(selected))
                    times = (recovery_age[selected] - 8).float() * base.step_dt
                    outcomes[name]["recovery_times"].append(times.cpu())
            recovery_active[recovered] = False
            failed_recovery = recovery_active & terminated
            recovery_censored = recovery_active & timed_out & ~terminated
            increment(failed_recovery, "failed_recovery")
            increment(recovery_censored, "recovery_censored_timeout")
            recovery_active[failed_recovery | recovery_censored] = False
            recovery_age[recovery_active] += 1

            episode_steps += 1
            done_ids = torch.nonzero(done).squeeze(-1)
            if done_ids.numel() > 0:
                completed_lengths.append(episode_steps[done_ids].cpu())
                base_contacts.append(terminated[done_ids].float().cpu())
                timeouts.append(timed_out[done_ids].float().cpu())
                episode_steps[done_ids] = 0
                seen_id[done_ids] = 0
                window_active[done_ids] = False
                recovery_active[done_ids] = False
            if (step + 1) % 100 == 0:
                print(f"[INFO] Transition evaluation {step + 1}/{args.num_steps}")

    # Keep finite-rollout censoring separate from physical episode timeout censoring.
    for direction_id, name in DIRECTIONS:
        outcomes[name]["survival_censored_evaluation_end"] = int(
            torch.count_nonzero(window_active & (active_direction == direction_id))
        )
        outcomes[name]["recovery_censored_evaluation_end"] = int(
            torch.count_nonzero(recovery_active & (active_direction == direction_id))
        )

    summary = {
        "metadata": {"task": args.task, "checkpoint": str(checkpoint), "num_envs": n,
                     "num_steps": args.num_steps, "seed": args.seed, "step_dt": base.step_dt,
                     "observation_dimension": obs_dim, "action_dimension": base.action_manager.total_action_dim,
                     "dwell_time_range_s": list(action_term.cfg.dwell_time_range_s),
                     "transition_window_steps": 30, "recovery_threshold_m_s": 0.25,
                     "recovery_consecutive_steps": 10, "weights": WEIGHTS},
        "overall": {key: stats(cat(value)) for key, value in overall.items()},
        "rewards": {name: {kind: stats(cat(parts)) for kind, parts in values.items()}
                    for name, values in rewards.items()},
        "phase": {name: {key: stats(cat(value)) for key, value in metrics.items()}
                  for name, metrics in phase.items()},
        "episodes": {"length_steps": stats(cat(completed_lengths)),
                     "base_contact_rate": cat(base_contacts).mean().item() if base_contacts else None,
                     "timeout_rate": cat(timeouts).mean().item() if timeouts else None},
        "time_from_last_gait_switch_to_failure": rate_term.failure_timing_summary(),
        "directions": {},
    }
    for _, name in DIRECTIONS:
        data = outcomes[name]
        observed = data["survived"] + data["survival_failed"]
        recovery_closed = (data["recovered"] + data["failed_recovery"]
                           + data["not_recovered_before_next_switch"] + data["recovery_censored_timeout"])
        summary["directions"][name] = {
            **{key: value for key, value in data.items() if key != "recovery_times"},
            "transition_survival_rate": data["survived"] / observed if observed else None,
            "survival_denominator_excludes_censored": observed,
            "recovery_censored_timeout_rate": data["recovery_censored_timeout"] / recovery_closed if recovery_closed else None,
            "time_to_recover_velocity_s": stats(cat(data["recovery_times"])),
            "transition_records": {key: stats(cat(value)) for key, value in transition_records[name].items()},
            "transition_steps": {key: stats(cat(value)) for key, value in direction_steps[name].items()},
        }
    output = output_dir / "transition_evaluation.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[INFO] Saved {output}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
