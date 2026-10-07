#!/usr/bin/env python3
"""
Audit a dataset produced by collect_expert_data.py.

Example:
python scripts/python/representation/audit_expert_dataset.py \
  --dataset artifacts/full_pipeline/datasets/trot
"""

import argparse
import json
from collections import deque
from pathlib import Path

import torch


def pct(x, n):
    return 100.0 * x / max(n, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--turn_threshold", type=float, default=0.05)
    parser.add_argument("--expected_vx_min", type=float, default=0.1)
    parser.add_argument("--expected_vx_max", type=float, default=3.5)
    parser.add_argument("--history_atol", type=float, default=1e-6)
    args = parser.parse_args()

    root = Path(args.dataset).expanduser().resolve()
    meta_file = root / "metadata.json"
    chunks = sorted(root.glob("chunk_*.pt"))

    if not meta_file.exists():
        raise FileNotFoundError(f"Missing: {meta_file}")
    if not chunks:
        raise FileNotFoundError(f"No chunk_*.pt files found in {root}")

    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    layout = meta["state_layout"]
    state_dim_expected = int(meta["single_frame_state_dim"])
    num_joints = int(meta["num_joints"])

    print("=" * 90)
    print("DATASET AUDIT")
    print(f"dataset        : {root}")
    print(f"gait           : {meta.get('gait')}")
    print(f"checkpoint     : {meta.get('checkpoint')}")
    print(f"num_joints     : {num_joints}")
    print(f"state_dim      : {state_dim_expected}")
    print(f"chunks         : {len(chunks)}")
    print("=" * 90)

    total = 0
    nonfinite_state = 0
    nonfinite_action = 0
    nonfinite_tau = 0

    vx_min = float("inf")
    vx_max = float("-inf")
    vy_abs_max = 0.0
    turn_count = 0

    tau_abs_max = 0.0
    tau_count = 0
    tau_gt_40 = 0
    tau_gt_45 = 0
    tau_gt_47 = 0

    contact_bad = 0
    command_mismatch = 0

    gravity_norm_sum = 0.0
    gravity_norm_count = 0
    gravity_norm_min = float("inf")
    gravity_norm_max = float("-inf")

    foot_h_min = float("inf")
    foot_h_max = float("-inf")

    cache = {}
    valid_3frame_sequences = 0
    history_comparable = 0
    history_bad_q = 0
    history_bad_dq = 0
    discontinuity_count = 0
    episode_switch_count = 0

    def sl(name):
        a, b = layout[name]
        return slice(int(a), int(b))

    s_cmd = sl("command_velocity")
    s_contact = sl("contact")
    s_grav = sl("projected_gravity_b")
    s_foot_h = sl("foot_height")
    s_q = sl("joint_pos")
    s_dq = sl("joint_vel")
    s_qm2 = sl("joint_pos_t_minus_2")
    s_dqm2 = sl("joint_vel_t_minus_2")

    required_keys = {
        "encoder_state",
        "policy_action",
        "applied_torque",
        "env_id",
        "episode_id",
        "step_in_episode",
        "done_after_step",
        "command",
    }

    for ci, file in enumerate(chunks):
        d = torch.load(file, map_location="cpu", weights_only=False)
        missing = required_keys - set(d.keys())
        if missing:
            raise RuntimeError(f"{file.name}: missing keys: {sorted(missing)}")

        state = d["encoder_state"]
        action = d["policy_action"]
        tau = d["applied_torque"]
        env_id = d["env_id"].long()
        ep = d["episode_id"].long()
        step = d["step_in_episode"].long()
        command = d["command"]

        n = state.shape[0]
        total += n

        if state.ndim != 2 or state.shape[1] != state_dim_expected:
            raise RuntimeError(
                f"{file.name}: encoder_state shape {tuple(state.shape)} != (N,{state_dim_expected})"
            )
        if tau.ndim != 2 or tau.shape[1] != num_joints:
            raise RuntimeError(
                f"{file.name}: applied_torque shape {tuple(tau.shape)} != (N,{num_joints})"
            )
        if action.ndim != 2 or action.shape != (n, num_joints):
            raise RuntimeError(
                f"{file.name}: policy_action shape {tuple(action.shape)} != (N,{num_joints})"
            )
        if command.ndim != 2 or command.shape[1] < 3:
            raise RuntimeError(f"{file.name}: command shape invalid: {tuple(command.shape)}")

        nonfinite_state += int((~torch.isfinite(state)).sum().item())
        nonfinite_action += int((~torch.isfinite(action)).sum().item())
        nonfinite_tau += int((~torch.isfinite(tau)).sum().item())

        cmd_in_state = state[:, s_cmd]
        command_mismatch += int(
            (torch.max(torch.abs(cmd_in_state - command[:, :3]), dim=1).values > 1e-6).sum().item()
        )

        vx = command[:, 0]
        vy = command[:, 1]
        wz = command[:, 2]
        vx_min = min(vx_min, float(vx.min().item()))
        vx_max = max(vx_max, float(vx.max().item()))
        vy_abs_max = max(vy_abs_max, float(vy.abs().max().item()))
        turn_count += int((wz.abs() > args.turn_threshold).sum().item())

        contact = state[:, s_contact]
        contact_bad += int((~((contact == 0.0) | (contact == 1.0))).sum().item())

        gnorm = torch.linalg.vector_norm(state[:, s_grav].float(), dim=1)
        gravity_norm_sum += float(gnorm.sum().item())
        gravity_norm_count += gnorm.numel()
        gravity_norm_min = min(gravity_norm_min, float(gnorm.min().item()))
        gravity_norm_max = max(gravity_norm_max, float(gnorm.max().item()))

        fh = state[:, s_foot_h]
        foot_h_min = min(foot_h_min, float(fh.min().item()))
        foot_h_max = max(foot_h_max, float(fh.max().item()))

        at = tau.abs()
        tau_abs_max = max(tau_abs_max, float(at.max().item()))
        tau_count += at.numel()
        tau_gt_40 += int((at > 40.0).sum().item())
        tau_gt_45 += int((at > 45.0).sum().item())
        tau_gt_47 += int((at > 47.0).sum().item())

        for i in range(n):
            e = int(env_id[i].item())
            epi = int(ep[i].item())
            st = int(step[i].item())

            q = state[i, s_q]
            dq = state[i, s_dq]
            qm2 = state[i, s_qm2]
            dqm2 = state[i, s_dqm2]

            if e not in cache:
                cache[e] = deque(maxlen=4)

            hist = cache[e]

            if hist:
                prev_epi, prev_st, _, _ = hist[-1]
                if epi != prev_epi:
                    episode_switch_count += 1
                elif st != prev_st + 1:
                    discontinuity_count += 1

            prev1 = None
            prev2 = None
            for item in reversed(hist):
                h_epi, h_st, h_q, h_dq = item
                if h_epi != epi:
                    continue
                if h_st == st - 1:
                    prev1 = item
                elif h_st == st - 2:
                    prev2 = item

            if prev1 is not None and prev2 is not None:
                valid_3frame_sequences += 1

            if prev2 is not None:
                history_comparable += 1
                _, _, q_ref, dq_ref = prev2
                if not torch.allclose(qm2, q_ref, atol=args.history_atol, rtol=0.0):
                    history_bad_q += 1
                if not torch.allclose(dqm2, dq_ref, atol=args.history_atol, rtol=0.0):
                    history_bad_dq += 1

            hist.append((epi, st, q.clone(), dq.clone()))

        print(f"[{ci+1:02d}/{len(chunks):02d}] {file.name}: {n:,} samples")

    print("\n" + "=" * 90)
    print("RESULTS")
    print("=" * 90)

    print(f"Total frames                         : {total:,}")
    print(f"State shape                          : {state_dim_expected}")
    print(f"Torque shape                         : {num_joints}")
    print(f"Non-finite state elements            : {nonfinite_state}")
    print(f"Non-finite policy-action elements    : {nonfinite_action}")
    print(f"Non-finite torque elements           : {nonfinite_tau}")
    print(f"Command/state mismatched rows        : {command_mismatch}")
    print(f"Non-binary contact elements          : {contact_bad}")

    print("\n--- Command coverage ---")
    print(f"command vx range                     : [{vx_min:.6f}, {vx_max:.6f}] m/s")
    print(f"max |command vy|                     : {vy_abs_max:.6f} m/s")
    print(f"turning fraction |wz|>{args.turn_threshold:g}       : {pct(turn_count, total):.4f}%")

    print("\n--- Physical-state sanity ---")
    print(f"projected gravity norm mean          : {gravity_norm_sum / max(gravity_norm_count,1):.8f}")
    print(f"projected gravity norm range         : [{gravity_norm_min:.8f}, {gravity_norm_max:.8f}]")
    print(f"foot height global range             : [{foot_h_min:.6f}, {foot_h_max:.6f}] m")

    print("\n--- Torque ---")
    print(f"max |applied torque|                 : {tau_abs_max:.6f} Nm")
    print(f"|tau| > 40 Nm                        : {tau_gt_40:,} / {tau_count:,} ({pct(tau_gt_40,tau_count):.6f}%)")
    print(f"|tau| > 45 Nm                        : {tau_gt_45:,} / {tau_count:,} ({pct(tau_gt_45,tau_count):.6f}%)")
    print(f"|tau| > 47 Nm                        : {tau_gt_47:,} / {tau_count:,} ({pct(tau_gt_47,tau_count):.6f}%)")

    print("\n--- Temporal integrity ---")
    print(f"legal consecutive 3-frame sequences : {valid_3frame_sequences:,}")
    print(f"q(t-2) comparisons performed         : {history_comparable:,}")
    print(f"q(t-2) mismatches                    : {history_bad_q:,}")
    print(f"dq(t-2) mismatches                   : {history_bad_dq:,}")
    print(f"episode switches observed            : {episode_switch_count:,}")
    print(f"saved-step gaps within same episode  : {discontinuity_count:,}")

    print("\n--- PASS / WARN summary ---")
    failures = []

    if total == 0:
        failures.append("dataset is empty")
    if nonfinite_state != 0:
        failures.append("state contains NaN/Inf")
    if nonfinite_action != 0:
        failures.append("policy action contains NaN/Inf")
    if nonfinite_tau != 0:
        failures.append("torque contains NaN/Inf")
    if command_mismatch != 0:
        failures.append("command stored separately does not match command inside encoder_state")
    if contact_bad != 0:
        failures.append("contact entries are not binary")
    if history_bad_q != 0:
        failures.append("q(t-2) history mismatch detected")
    if history_bad_dq != 0:
        failures.append("dq(t-2) history mismatch detected")

    if failures:
        print("FAIL:")
        for x in failures:
            print(f"  - {x}")
    else:
        print("CORE DATA INTEGRITY: PASS")

    if vx_min < args.expected_vx_min - 1e-3 or vx_max > args.expected_vx_max + 1e-3:
        print(
            f"WARNING: commanded vx extends outside expected "
            f"[{args.expected_vx_min}, {args.expected_vx_max}]"
        )

    if tau_abs_max >= 47.9:
        print("WARNING: torque reaches very close to the 48 Nm actuator limit.")

    print("=" * 90)
    if failures:
        raise RuntimeError("Dataset integrity audit failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
