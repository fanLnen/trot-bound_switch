#!/usr/bin/env python3
import argparse
import json
from collections import deque
from pathlib import Path

import torch


REQ = {
    "encoder_state",
    "policy_action",
    "applied_torque",
    "env_id",
    "episode_id",
    "step_in_episode",
}


def build_one(dataset_dir: Path, gait: str, out_file: Path):
    dataset_dir = dataset_dir.resolve()
    meta = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
    chunks = sorted(dataset_dir.glob("chunk_*.pt"))
    if not chunks:
        raise FileNotFoundError(f"No chunk_*.pt in {dataset_dir}")

    if str(meta["gait"]).lower() != gait:
        raise RuntimeError(f"Gait mismatch: {meta['gait']} vs {gait}")
    if int(meta["single_frame_state_dim"]) != 85:
        raise RuntimeError("Expected state_dim=85")
    if int(meta["num_joints"]) != 16:
        raise RuntimeError("Expected 16 joints")

    history = {}  # env -> deque[(episode, step, global_idx)]
    seq_parts, env_parts, ep_parts, step_parts = [], [], [], []
    chunk_names, chunk_sizes, chunk_offsets = [], [], [0]

    global_offset = 0
    episode_switches = 0
    same_episode_gaps = 0

    print("=" * 88)
    print(f"BUILD {gait.upper()} 3-FRAME INDEX")
    print(f"dataset: {dataset_dir}")
    print("=" * 88)

    for k, path in enumerate(chunks):
        d = torch.load(path, map_location="cpu", weights_only=False)
        missing = REQ - set(d.keys())
        if missing:
            raise RuntimeError(f"{path.name}: missing {sorted(missing)}")

        s = d["encoder_state"]
        action = d["policy_action"]
        tau = d["applied_torque"]
        env = d["env_id"].long()
        ep = d["episode_id"].long()
        step = d["step_in_episode"].long()
        n = s.shape[0]

        if s.shape[1:] != (85,):
            raise RuntimeError(f"{path.name}: state shape {tuple(s.shape)}")
        if tau.shape[1:] != (16,):
            raise RuntimeError(f"{path.name}: torque shape {tuple(tau.shape)}")
        if action.shape[1:] != (16,):
            raise RuntimeError(f"{path.name}: policy action shape {tuple(action.shape)}")

        chunk_names.append(path.name)
        chunk_sizes.append(n)
        chunk_offsets.append(global_offset + n)

        seq_local, env_local, ep_local, st_local = [], [], [], []

        for i in range(n):
            e = int(env[i])
            p = int(ep[i])
            st = int(step[i])
            gi = global_offset + i

            h = history.setdefault(e, deque(maxlen=2))

            if h:
                last_ep, last_st, _ = h[-1]
                if p != last_ep:
                    episode_switches += 1
                    h.clear()
                elif st != last_st + 1:
                    same_episode_gaps += 1
                    h.clear()

            if len(h) == 2:
                ep0, st0, idx0 = h[0]
                ep1, st1, idx1 = h[1]
                if ep0 == p and ep1 == p and st0 == st - 2 and st1 == st - 1:
                    seq_local.append((idx0, idx1, gi))
                    env_local.append(e)
                    ep_local.append(p)
                    st_local.append(st - 2)

            h.append((p, st, gi))

        if seq_local:
            seq_parts.append(torch.tensor(seq_local, dtype=torch.int64))
            env_parts.append(torch.tensor(env_local, dtype=torch.int32))
            ep_parts.append(torch.tensor(ep_local, dtype=torch.int64))
            step_parts.append(torch.tensor(st_local, dtype=torch.int32))

        global_offset += n
        current = sum(x.shape[0] for x in seq_parts)
        print(f"[{k+1:02d}/{len(chunks):02d}] {path.name}: rows={n:,}, seq={current:,}")

    if same_episode_gaps:
        raise RuntimeError(f"Found {same_episode_gaps} gaps within same episode")

    seq = torch.cat(seq_parts, dim=0)
    env = torch.cat(env_parts, dim=0)
    ep = torch.cat(ep_parts, dim=0)
    st = torch.cat(step_parts, dim=0)

    payload = {
        "seq_indices": seq,          # [N,3] -> [s_t, s_t+1, s_t+2]
        "env_id": env,
        "episode_id": ep,
        "step_t": st,
        "gait": gait,
        "raw_dataset_dir": str(dataset_dir),
        "chunk_files": chunk_names,
        "chunk_sizes": torch.tensor(chunk_sizes, dtype=torch.int64),
        "chunk_offsets": torch.tensor(chunk_offsets, dtype=torch.int64),
        "state_dim": 85,
        "torque_dim": 16,
        "sequence_length": 3,
        "action_target_rule": "use raw policy_action at seq_indices[:,0]",
        "applied_torque_role": "dataset audit diagnostic; not the ActionDecoder target",
    }

    out_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_file)

    print("-" * 88)
    print(f"raw frames      : {global_offset:,}")
    print(f"legal sequences : {seq.shape[0]:,}")
    print(f"episode switches: {episode_switches:,}")
    print(f"same-ep gaps    : {same_episode_gaps:,}")
    print(f"saved           : {out_file}")
    print("=" * 88)

    return {
        "gait": gait,
        "raw_frames": global_offset,
        "legal_sequences": int(seq.shape[0]),
        "episode_switches": episode_switches,
        "same_episode_gaps": same_episode_gaps,
        "index_file": str(out_file),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trot_dataset", required=True)
    p.add_argument("--bound_dataset", required=True)
    p.add_argument("--output_dir", required=True)
    a = p.parse_args()

    out = Path(a.output_dir).resolve()
    trot = build_one(Path(a.trot_dataset), "trot", out / "trot_sequence_index.pt")
    bound = build_one(Path(a.bound_dataset), "bound", out / "bound_sequence_index.pt")

    summary = {
        "input_shape": [3, 85],
        "torque_dim": 16,
        "normalization_applied": False,
        "train_val_split_applied": False,
        "trot": trot,
        "bound": bound,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Trot : {trot['legal_sequences']:,} sequences")
    print(f"Bound: {bound['legal_sequences']:,} sequences")


if __name__ == "__main__":
    main()
