# Copyright (c) 2026
# Expert-policy rollout collector for Kang et al. APT-RL style representation learning.
#
# IMPORTANT:
# - The expert policy still uses its ORIGINAL observation space.
# - This script separately constructs the representation-learning state.
# - For a 16-DOF quadruped, the Kang-style state is 85-D per frame:
#   3 body lin vel
# + 3 body ang vel
# + 1 body height
# + 4 foot heights
# + 4 contact states
# + 3 velocity commands
# + 3 projected gravity
# + 16 joint positions
# + 16 joint velocities
# + 16 joint positions from 2 control steps ago
# + 16 joint velocities from 2 control steps ago
# = 85.
#
# The TVAE should later consume THREE CONSECUTIVE 85-D frames.
# This collector stores raw per-step frames plus env_id/episode_id/step_in_episode
# so a later dataset builder can make [s_t, s_{t+1}, s_{t+2}] without crossing reset.

"""Collect expert-policy state/torque data headlessly with many parallel Isaac Lab environments."""

import argparse
import importlib.metadata as metadata
import json
import re
from pathlib import Path

from packaging import version

from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Collect Kang-style representation-learning state/torque data from a trained RSL-RL expert."
)
parser.add_argument("--num_envs", type=int, default=512, help="Number of parallel environments.")
parser.add_argument("--seed", type=int, default=42, help="Environment seed.")
parser.add_argument("--checkpoint", type=Path, required=True, help="Trained expert checkpoint.")

parser.add_argument("--gait", choices=["trot", "bound"], required=True)
parser.add_argument(
    "--target_samples",
    type=int,
    default=2_000_000,
    help="Target number of valid per-frame samples to save.",
)
parser.add_argument(
    "--chunk_size",
    type=int,
    default=100_000,
    help="Approximate number of samples per .pt chunk.",
)
parser.add_argument(
    "--max_control_steps",
    type=int,
    default=100_000,
    help="Safety limit on collector control steps.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    required=True,
    help="Output directory for chunk files and metadata.",
)

# Scene entity names. Change only if your task uses different keys.
parser.add_argument("--robot_key", type=str, default="robot")
parser.add_argument("--contact_sensor_key", type=str, default="contact_forces")
parser.add_argument("--command_term", type=str, default="base_velocity")

# Feet. Exact ordered names are preferred when known.
parser.add_argument(
    "--foot_body_names",
    type=str,
    default=None,
    help="Comma-separated exact foot body names in desired order, e.g. FL_foot,FR_foot,RL_foot,RR_foot.",
)
parser.add_argument(
    "--foot_body_regex",
    type=str,
    default=r".*(foot|FOOT).*",
    help="Fallback regex used to auto-find exactly four foot bodies.",
)
parser.add_argument(
    "--contact_threshold",
    type=float,
    default=1.0,
    help="Contact force norm threshold [N] used to convert foot contact to 0/1.",
)

# The paper state contains q/dq from 2 control steps ago.
# We therefore discard the first two steps after every reset.
parser.add_argument(
    "--skip_steps_after_reset",
    type=int,
    default=2,
    help="Do not save the first N control steps after reset. Keep at 2 for Kang-style q/dq history.",
)

# These are checks, not command generators. The task should already generate
# the same commands the expert was trained on.
parser.add_argument("--expected_vx_min", type=float, default=0.1)
parser.add_argument("--expected_vx_max", type=float, default=3.5)
parser.add_argument(
    "--turn_yaw_threshold",
    type=float,
    default=0.05,
    help="abs(command yaw rate) above this value is counted as turning.",
)
parser.add_argument(
    "--expected_trot_turn_fraction",
    type=float,
    default=0.20,
    help="Expected turning fraction for trot; used only as a diagnostic.",
)

# Standard Isaac Lab launcher arguments, including --headless.
AppLauncher.add_app_launcher_args(parser)
# This collector is headless by default.
parser.set_defaults(headless=True)

args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# -----------------------------------------------------------------------------
# Isaac Lab / RSL-RL imports (after simulator launch)
# -----------------------------------------------------------------------------

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# Version compatibility helpers.
try:
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
except ImportError:
    def handle_deprecated_rsl_rl_cfg(cfg, _installed_version):
        return cfg

import trot_bound_switch  # noqa: F401
from trot_bound_switch.tasks import BOUND_COLLECT_TASK_ID, TROT_COLLECT_TASK_ID
from trot_bound_switch.tasks.agents import BoundExpertPPORunnerCfg, TrotExpertPPORunnerCfg
from trot_bound_switch.tasks.expert_env_cfg import BoundCollectEnvCfg, TrotCollectEnvCfg


installed_version = metadata.version("rsl-rl-lib")


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

class ChunkWriter:
    """GPU-side temporary buffer with chunked torch.save() output."""

    def __init__(self, output_dir: Path, chunk_size: int):
        self.output_dir = output_dir
        self.chunk_size = int(chunk_size)
        self.buffers: dict[str, list[torch.Tensor]] = {}
        self.buffer_count = 0
        self.chunk_index = 0
        self.total_written = 0

    def add(self, **items):
        if not items:
            return
        n = None
        for key, value in items.items():
            if value is None:
                continue
            if n is None:
                n = value.shape[0]
            elif value.shape[0] != n:
                raise RuntimeError(f"ChunkWriter size mismatch for '{key}': {value.shape[0]} vs {n}")
            self.buffers.setdefault(key, []).append(value.detach())
        if n is not None:
            self.buffer_count += n
        if self.buffer_count >= self.chunk_size:
            self.flush()

    def flush(self):
        if self.buffer_count == 0:
            return
        payload = {}
        for key, pieces in self.buffers.items():
            payload[key] = torch.cat(pieces, dim=0).cpu()

        out_file = self.output_dir / f"chunk_{self.chunk_index:05d}.pt"
        torch.save(payload, out_file)

        n = next(iter(payload.values())).shape[0]
        self.total_written += n
        print(
            f"[DATA] saved {out_file.name}: {n:,} samples "
            f"(total written {self.total_written:,})"
        )

        self.buffers.clear()
        self.buffer_count = 0
        self.chunk_index += 1


def _get_scene_entity(scene, key: str):
    try:
        return scene[key]
    except Exception as exc:
        raise KeyError(
            f"Could not find scene entity '{key}'. "
            f"Use --robot_key/--contact_sensor_key to match your task config."
        ) from exc


def resolve_feet(robot, contact_sensor, args):
    """Resolve the same four feet for kinematics and contact signals."""

    if args.foot_body_names:
        requested = [x.strip() for x in args.foot_body_names.split(",") if x.strip()]
        if len(requested) != 4:
            raise ValueError("--foot_body_names must contain exactly four comma-separated names.")
        patterns = [rf"^{re.escape(name)}$" for name in requested]
        body_ids, body_names = robot.find_bodies(patterns, preserve_order=True)
    else:
        body_ids, body_names = robot.find_bodies(args.foot_body_regex, preserve_order=False)

    if len(body_ids) != 4:
        raise RuntimeError(
            "Expected exactly 4 foot bodies, but found "
            f"{len(body_ids)}: {body_names}\n"
            f"Robot bodies are:\n{robot.body_names}\n"
            "Pass exact ordered names with --foot_body_names."
        )

    sensor_body_names = list(contact_sensor.body_names)
    sensor_ids = []
    for name in body_names:
        if name not in sensor_body_names:
            raise RuntimeError(
                f"Foot body '{name}' is not present in contact sensor '{args.contact_sensor_key}'.\n"
                f"Contact sensor bodies are: {sensor_body_names}"
            )
        sensor_ids.append(sensor_body_names.index(name))

    return body_ids, body_names, sensor_ids


def make_state_layout(num_joints: int):
    """Return exact slices for the Kang-style single-frame state."""
    layout = {}
    cursor = 0

    def add(name, width):
        nonlocal cursor
        layout[name] = [cursor, cursor + width]
        cursor += width

    add("base_lin_vel_b", 3)
    add("base_ang_vel_b", 3)
    add("body_height", 1)
    add("foot_height", 4)
    add("contact", 4)
    add("command_velocity", 3)
    add("projected_gravity_b", 3)
    add("joint_pos", num_joints)
    add("joint_vel", num_joints)
    add("joint_pos_t_minus_2", num_joints)
    add("joint_vel_t_minus_2", num_joints)

    return layout, cursor


def build_encoder_state(
    base_env,
    robot,
    contact_sensor,
    foot_body_ids,
    foot_sensor_ids,
    command_term: str,
    q_t_minus_2: torch.Tensor,
    dq_t_minus_2: torch.Tensor,
    contact_threshold: float,
):
    """
    Build one Kang-style state frame.

    IMPORTANT:
    q_t_minus_2/dq_t_minus_2 mean exactly TWO CONTROL STEPS AGO.
    They are NOT previous policy actions.
    """
    base_lin_vel = robot.data.root_lin_vel_b
    base_ang_vel = robot.data.root_ang_vel_b

    # Flat-ground height reference: local env origin z.
    # For the first strict-Kang dataset, use a flat environment.
    env_origin_z = base_env.scene.env_origins[:, 2]
    body_height = (robot.data.root_pos_w[:, 2] - env_origin_z).unsqueeze(-1)

    foot_z_w = robot.data.body_pos_w[:, foot_body_ids, 2]
    foot_height = foot_z_w - env_origin_z.unsqueeze(-1)

    # Contact information: 4 binary foot contacts.
    forces_w = contact_sensor.data.net_forces_w[:, foot_sensor_ids, :]
    contact = (torch.linalg.vector_norm(forces_w, dim=-1) > contact_threshold).to(dtype=base_lin_vel.dtype)

    command = base_env.command_manager.get_command(command_term).clone()
    if command.shape[-1] < 3:
        raise RuntimeError(
            f"Command term '{command_term}' has shape {tuple(command.shape)}; expected at least 3 dims [vx, vy, wz]."
        )
    command = command[:, :3]

    projected_gravity = robot.data.projected_gravity_b
    joint_pos = robot.data.joint_pos
    joint_vel = robot.data.joint_vel

    state = torch.cat(
        (
            base_lin_vel,          # 3
            base_ang_vel,          # 3
            body_height,           # 1
            foot_height,           # 4
            contact,               # 4
            command,               # 3
            projected_gravity,     # 3
            joint_pos,             # J
            joint_vel,             # J
            q_t_minus_2,           # J
            dq_t_minus_2,          # J
        ),
        dim=-1,
    )

    return state, command, contact, forces_w


def reset_policy_state(policy, policy_nn, dones):
    """Support multiple rsl_rl generations."""
    if version.parse(installed_version) >= version.parse("4.0.0"):
        if hasattr(policy, "reset"):
            policy.reset(dones)
    else:
        if policy_nn is not None and hasattr(policy_nn, "reset"):
            policy_nn.reset(dones)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    # -------------------------------------------------------------------------
    # Load the exact expert policy the same way as play.py
    # -------------------------------------------------------------------------

    if args_cli.gait == "trot":
        task_id = TROT_COLLECT_TASK_ID
        env_cfg = TrotCollectEnvCfg()
        agent_cfg = TrotExpertPPORunnerCfg()
    else:
        task_id = BOUND_COLLECT_TASK_ID
        env_cfg = BoundCollectEnvCfg()
        agent_cfg = BoundExpertPPORunnerCfg()

    env_cfg.scene.num_envs = args_cli.num_envs

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    agent_cfg.seed = args_cli.seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    resume_path = args_cli.checkpoint.expanduser().resolve()
    if not resume_path.is_file():
        raise FileNotFoundError(resume_path)
    print(f"[INFO] Loading checkpoint: {resume_path}")

    env_cfg.log_dir = str(resume_path.parent)
    env = gym.make(task_id, cfg=env_cfg)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    policy_nn = None
    if version.parse(installed_version) < version.parse("4.0.0"):
        if version.parse(installed_version) >= version.parse("2.3.0"):
            policy_nn = runner.alg.policy
        else:
            policy_nn = runner.alg.actor_critic

    # -------------------------------------------------------------------------
    # Resolve simulator data sources
    # -------------------------------------------------------------------------

    base_env = env.unwrapped

    if not hasattr(base_env, "scene"):
        raise RuntimeError("This collector expects an Isaac Lab environment with base_env.scene.")
    if not hasattr(base_env, "command_manager"):
        raise RuntimeError(
            "This collector expects a ManagerBased locomotion task with command_manager. "
            "If your task is DirectRLEnv, adapt command extraction."
        )

    robot = _get_scene_entity(base_env.scene, args_cli.robot_key)
    contact_sensor = _get_scene_entity(base_env.scene, args_cli.contact_sensor_key)

    if args_cli.command_term not in base_env.command_manager.active_terms:
        raise RuntimeError(
            f"Command term '{args_cli.command_term}' not found. "
            f"Active terms: {base_env.command_manager.active_terms}"
        )

    foot_body_ids, foot_body_names, foot_sensor_ids = resolve_feet(robot, contact_sensor, args_cli)

    num_envs = base_env.num_envs
    num_joints = robot.num_joints
    state_layout, state_dim = make_state_layout(num_joints)

    print(f"[INFO] num_envs      = {num_envs}")
    print(f"[INFO] num_joints    = {num_joints}")
    print(f"[INFO] feet          = {foot_body_names}")
    print(f"[INFO] state_dim     = {state_dim}")
    print(f"[INFO] TVAE input    = 3 x {state_dim}")

    if num_joints != 16:
        print(
            f"[WARNING] Robot has {num_joints} joints. "
            "For your SA16 experiment we expected 16."
        )
    if num_joints == 16 and state_dim != 85:
        raise RuntimeError(f"Internal layout error: 16-DOF robot must give 85-D state, got {state_dim}.")

    # -------------------------------------------------------------------------
    # Output directory and metadata
    # -------------------------------------------------------------------------

    output_dir = Path(args_cli.output_dir)

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_dict = {
        "task": task_id,
        "gait": args_cli.gait,
        "checkpoint": str(resume_path),
        "num_envs": int(num_envs),
        "num_joints": int(num_joints),
        "joint_names": list(robot.joint_names),
        "foot_body_names": list(foot_body_names),
        "contact_sensor_key": args_cli.contact_sensor_key,
        "command_term": args_cli.command_term,
        "contact_threshold_N": float(args_cli.contact_threshold),
        "single_frame_state_dim": int(state_dim),
        "encoder_sequence_length": 3,
        "encoder_input_shape": [3, int(state_dim)],
        "joint_history_offset_control_steps": 2,
        "state_layout": state_layout,
        "state_definition": [
            "base_lin_vel_b",
            "base_ang_vel_b",
            "body_height",
            "foot_height_4",
            "binary_contact_4",
            "command_velocity_vx_vy_wz",
            "projected_gravity_b",
            "joint_pos",
            "joint_vel",
            "joint_pos_t_minus_2",
            "joint_vel_t_minus_2",
        ],
        "torque_target": "robot.data.applied_torque sampled after env.step(action_t), paired with pre-step state_t",
        "extra_diagnostics": ["computed_torque", "policy_action", "contact_force_norm"],
        "height_reference": "world_z minus env_origin_z; intended for the first flat-terrain dataset",
        "sequence_rule": (
            "Build [s_t,s_t+1,s_t+2] only when env_id and episode_id match "
            "and step_in_episode is consecutive. Never cross reset."
        ),
        "normalization": "NONE during collection; normalize later during representation learning",
    }

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata_dict, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Output: {output_dir}")

    writer = ChunkWriter(output_dir=output_dir, chunk_size=args_cli.chunk_size)

    # -------------------------------------------------------------------------
    # Initial state/history
    # -------------------------------------------------------------------------

    obs = env.get_observations()

    q0 = robot.data.joint_pos.clone()
    dq0 = robot.data.joint_vel.clone()

    # At reset we do not yet possess true t-1/t-2 history. The first two steps
    # are therefore not saved. These initial values are placeholders only.
    q_prev1 = q0.clone()
    q_prev2 = q0.clone()
    dq_prev1 = dq0.clone()
    dq_prev2 = dq0.clone()

    episode_id = torch.zeros(num_envs, dtype=torch.long, device=base_env.device)
    step_in_episode = torch.zeros(num_envs, dtype=torch.long, device=base_env.device)
    env_ids_all = torch.arange(num_envs, dtype=torch.long, device=base_env.device)

    collected = 0
    control_step = 0

    # Diagnostics
    max_abs_applied_torque_seen = 0.0
    vx_seen_min = float("inf")
    vx_seen_max = float("-inf")
    command_sample_count = 0
    turning_sample_count = 0
    vx_hist = torch.zeros(7, dtype=torch.long)

    # -------------------------------------------------------------------------
    # Rollout collection
    # -------------------------------------------------------------------------

    try:
        while (
            simulation_app.is_running()
            and collected < args_cli.target_samples
            and control_step < args_cli.max_control_steps
        ):
            with torch.inference_mode():
                # -------------------------------------------------------------
                # s_t is read BEFORE applying action_t.
                # -------------------------------------------------------------
                state_t, command_t, contact_t, contact_forces_t = build_encoder_state(
                    base_env=base_env,
                    robot=robot,
                    contact_sensor=contact_sensor,
                    foot_body_ids=foot_body_ids,
                    foot_sensor_ids=foot_sensor_ids,
                    command_term=args_cli.command_term,
                    q_t_minus_2=q_prev2,
                    dq_t_minus_2=dq_prev2,
                    contact_threshold=args_cli.contact_threshold,
                )

                if state_t.shape != (num_envs, state_dim):
                    raise RuntimeError(
                        f"Bad encoder-state shape: got {tuple(state_t.shape)}, "
                        f"expected {(num_envs, state_dim)}"
                    )

                # Save current q/dq before simulation advances.
                q_t = robot.data.joint_pos.clone()
                dq_t = robot.data.joint_vel.clone()
                ep_t = episode_id.clone()
                step_t = step_in_episode.clone()

                # Expert policy uses its ORIGINAL observation.
                actions_t = policy(obs)

                # Apply action_t. Isaac Lab returns the next observation and dones.
                obs_next, _, dones, _ = env.step(actions_t)
                dones = dones.to(dtype=torch.bool)

                # Torque corresponding to the action that has just been applied.
                # applied_torque is AFTER actuator clipping for explicit actuators.
                applied_torque_t = robot.data.applied_torque.clone()
                computed_torque_t = robot.data.computed_torque.clone()

                max_abs_applied_torque_seen = max(
                    max_abs_applied_torque_seen,
                    float(applied_torque_t.abs().max().item()),
                )

                # First 2 control steps after reset are excluded because q_{t-2},
                # dq_{t-2} are not genuine history yet.
                valid = step_t >= args_cli.skip_steps_after_reset
                valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)

                if valid_idx.numel() > 0:
                    remaining = args_cli.target_samples - collected
                    if valid_idx.numel() > remaining:
                        valid_idx = valid_idx[:remaining]

                    selected_cmd = command_t[valid_idx]

                    # Dataset-wide command diagnostics.
                    vx = selected_cmd[:, 0]
                    vx_seen_min = min(vx_seen_min, float(vx.min().item()))
                    vx_seen_max = max(vx_seen_max, float(vx.max().item()))
                    command_sample_count += int(vx.numel())
                    turning_sample_count += int(
                        (selected_cmd[:, 2].abs() > args_cli.turn_yaw_threshold).sum().item()
                    )
                    vx_hist += torch.histc(
                        vx.float(),
                        bins=7,
                        min=args_cli.expected_vx_min,
                        max=args_cli.expected_vx_max,
                    ).to(dtype=torch.long).cpu()

                    force_norm = torch.linalg.vector_norm(contact_forces_t[valid_idx], dim=-1)

                    writer.add(
                        # Required representation-learning data
                        encoder_state=state_t[valid_idx].to(dtype=torch.float32),
                        applied_torque=applied_torque_t[valid_idx].to(dtype=torch.float32),

                        # Sequence/reset bookkeeping
                        env_id=env_ids_all[valid_idx].to(dtype=torch.int32),
                        episode_id=ep_t[valid_idx].to(dtype=torch.int64),
                        step_in_episode=step_t[valid_idx].to(dtype=torch.int32),
                        done_after_step=dones[valid_idx],

                        # Useful diagnostics / audit data
                        command=selected_cmd.to(dtype=torch.float32),
                        policy_action=actions_t[valid_idx].to(dtype=torch.float32),
                        computed_torque=computed_torque_t[valid_idx].to(dtype=torch.float32),
                        contact_force_norm=force_norm.to(dtype=torch.float32),
                    )

                    collected += int(valid_idx.numel())

                # -------------------------------------------------------------
                # Advance q/dq history for each environment.
                #
                # At the NEXT state (t+1), q_prev2 must equal q_{t-1}.
                # -------------------------------------------------------------
                q_prev2.copy_(q_prev1)
                dq_prev2.copy_(dq_prev1)
                q_prev1.copy_(q_t)
                dq_prev1.copy_(dq_t)

                # Episode bookkeeping. Isaac Lab auto-resets done environments.
                step_in_episode += 1

                done_ids = torch.nonzero(dones, as_tuple=False).squeeze(-1)
                if done_ids.numel() > 0:
                    episode_id[done_ids] += 1
                    step_in_episode[done_ids] = 0

                    # After auto-reset, do NOT let old-episode q/dq leak into the
                    # new episode's history. Initialize history with reset state;
                    # the first two steps are discarded anyway.
                    q_reset = robot.data.joint_pos[done_ids]
                    dq_reset = robot.data.joint_vel[done_ids]
                    q_prev1[done_ids] = q_reset
                    q_prev2[done_ids] = q_reset
                    dq_prev1[done_ids] = dq_reset
                    dq_prev2[done_ids] = dq_reset

                reset_policy_state(policy, policy_nn, dones)

                obs = obs_next
                control_step += 1

            if control_step % 100 == 0:
                print(
                    f"[COLLECT] control_step={control_step:,} "
                    f"samples={collected:,}/{args_cli.target_samples:,} "
                    f"max|tau_applied|={max_abs_applied_torque_seen:.3f}"
                )

    finally:
        writer.flush()

    # -------------------------------------------------------------------------
    # Final validation/report
    # -------------------------------------------------------------------------

    print("\n" + "=" * 78)
    print("[DONE] Expert-data collection finished")
    print(f"  gait                  : {args_cli.gait}")
    print(f"  samples               : {collected:,}")
    print(f"  control steps         : {control_step:,}")
    print(f"  output                : {output_dir}")
    print(f"  state shape per frame : ({state_dim},)")
    print(f"  future TVAE input     : (3, {state_dim})")
    print(f"  max |applied torque|  : {max_abs_applied_torque_seen:.6f} Nm")

    if command_sample_count > 0:
        turn_fraction = turning_sample_count / command_sample_count
        print(f"  command vx seen       : [{vx_seen_min:.4f}, {vx_seen_max:.4f}] m/s")
        print(f"  turning fraction      : {turn_fraction:.4%}")
        print(f"  vx 7-bin histogram    : {vx_hist.tolist()}")

        # Coverage checks.
        if vx_seen_min > args_cli.expected_vx_min + 0.15:
            print(
                "[WARNING] Lowest sampled vx is well above the intended lower range. "
                "Check your task command configuration."
            )
        if vx_seen_max < args_cli.expected_vx_max - 0.15:
            print(
                "[WARNING] Highest sampled vx is well below the intended upper range. "
                "Check your task command configuration."
            )

        if args_cli.gait == "bound":
            if turn_fraction > 0.01:
                print(
                    "[WARNING] Bound dataset contains noticeable turning commands. "
                    "You said Bound should be straight-only."
                )
        else:
            expected = args_cli.expected_trot_turn_fraction
            if abs(turn_fraction - expected) > 0.05:
                print(
                    f"[WARNING] Trot turning fraction is {turn_fraction:.3f}, "
                    f"not close to expected {expected:.3f}. "
                    "This script does NOT alter commands; fix/check the Trot task command sampler."
                )

    if max_abs_applied_torque_seen < 1.0e-6:
        raise RuntimeError(
            "robot.data.applied_torque stayed essentially zero. "
            "Do NOT train the torque decoder with this dataset. "
            "Your actuator may be implicit, or torque extraction must be adapted."
        )

    if collected < args_cli.target_samples:
        print(
            f"[WARNING] Stopped before target_samples: got {collected:,} / "
            f"{args_cli.target_samples:,}. Increase --max_control_steps if needed."
        )

    print("=" * 78)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
