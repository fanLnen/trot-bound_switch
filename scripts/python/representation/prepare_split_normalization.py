#!/usr/bin/env python3
"""
Create leakage-free train/validation splits and shared state normalization stats
for the Trot/Bound TVAE dataset.

What this script does
---------------------
1. Loads:
   - trot_sequence_index.pt
   - bound_sequence_index.pt
2. Splits EACH gait by complete (env_id, episode_id) groups.
   No episode is ever split across train and validation.
3. Targets ~90% train / ~10% validation by number of legal 3-frame sequences.
4. Computes ONE shared 85-D mean/std using ONLY raw frames from TRAIN episodes
   from BOTH Trot and Bound.
5. Saves only indices/statistics; it does NOT duplicate state sequences.

Important
---------
- Validation uses the TRAIN mean/std.
- Normalization is computed over unique raw train frames, not over overlapping
  3-frame windows, so middle frames are not counted 2-3 times just because
  sequence windows overlap.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def encode_episode_key(env_id: torch.Tensor, episode_id: torch.Tensor) -> torch.Tensor:
    """Pack (env_id, episode_id) into one int64 key."""
    env_id = env_id.to(torch.int64)
    episode_id = episode_id.to(torch.int64)
    if torch.any(env_id < 0) or torch.any(episode_id < 0):
        raise ValueError("env_id and episode_id must be non-negative.")
    return (env_id << 32) | episode_id


def isin_sorted(values: torch.Tensor, sorted_keys: torch.Tensor) -> torch.Tensor:
    """Fast membership test for int64 tensors using searchsorted."""
    if sorted_keys.numel() == 0:
        return torch.zeros(values.shape, dtype=torch.bool)
    pos = torch.searchsorted(sorted_keys, values)
    pos_safe = torch.clamp(pos, max=sorted_keys.numel() - 1)
    return (pos < sorted_keys.numel()) & (sorted_keys[pos_safe] == values)


def choose_val_episodes(
    episode_keys: torch.Tensor,
    episode_counts: torch.Tensor,
    val_fraction: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Choose whole episodes for validation while targeting val_fraction by
    number of legal sequences (not merely by number of episodes).
    """
    n_eps = episode_keys.numel()
    if n_eps < 2:
        raise RuntimeError("Need at least two episodes for train/validation split.")

    gen = torch.Generator()
    gen.manual_seed(seed)
    perm = torch.randperm(n_eps, generator=gen)

    keys_shuf = episode_keys[perm]
    counts_shuf = episode_counts[perm]

    total_seq = int(counts_shuf.sum().item())
    target_val = total_seq * val_fraction

    cumsum = torch.cumsum(counts_shuf, dim=0).to(torch.float64)
    k = int(torch.searchsorted(cumsum, torch.tensor(target_val, dtype=torch.float64)).item())

    # Number of validation episodes is either k or k+1; choose whichever gets
    # closer to the target number of validation sequences.
    candidates = []
    for n_val in (k, k + 1):
        n_val = max(1, min(n_val, n_eps - 1))
        val_seq = int(counts_shuf[:n_val].sum().item())
        candidates.append((abs(val_seq - target_val), n_val))

    _, best_n_val = min(candidates, key=lambda x: x[0])

    val_keys = torch.sort(keys_shuf[:best_n_val]).values
    train_keys = torch.sort(keys_shuf[best_n_val:]).values
    return train_keys, val_keys


def split_one_gait(index_file: Path, gait: str, val_fraction: float, seed: int) -> dict:
    data = torch.load(index_file, map_location="cpu", weights_only=False)

    if str(data["gait"]).lower() != gait:
        raise RuntimeError(
            f"{index_file}: gait is {data['gait']!r}, expected {gait!r}."
        )
    if int(data["state_dim"]) != 85:
        raise RuntimeError(f"{gait}: expected state_dim=85.")
    if int(data["sequence_length"]) != 3:
        raise RuntimeError(f"{gait}: expected sequence_length=3.")

    env_id = data["env_id"].long()
    episode_id = data["episode_id"].long()
    n_seq = env_id.numel()

    seq_episode_key = encode_episode_key(env_id, episode_id)
    unique_keys, inverse, counts = torch.unique(
        seq_episode_key, sorted=True, return_inverse=True, return_counts=True
    )

    train_ep_keys, val_ep_keys = choose_val_episodes(
        unique_keys,
        counts,
        val_fraction=val_fraction,
        seed=seed,
    )

    val_mask = isin_sorted(seq_episode_key, val_ep_keys)
    train_mask = ~val_mask

    train_seq_ids = torch.nonzero(train_mask, as_tuple=False).squeeze(1).to(torch.int64)
    val_seq_ids = torch.nonzero(val_mask, as_tuple=False).squeeze(1).to(torch.int64)

    # Hard leakage check.
    overlap = isin_sorted(train_ep_keys, val_ep_keys)
    if bool(overlap.any()):
        raise RuntimeError(f"{gait}: episode leakage between train and validation.")

    # Every sequence must be assigned exactly once.
    if train_seq_ids.numel() + val_seq_ids.numel() != n_seq:
        raise RuntimeError(f"{gait}: not all sequences assigned.")

    train_fraction_actual = train_seq_ids.numel() / n_seq
    val_fraction_actual = val_seq_ids.numel() / n_seq

    print("=" * 92)
    print(f"{gait.upper()} SPLIT")
    print(f"sequence index       : {index_file}")
    print(f"total episodes       : {unique_keys.numel():,}")
    print(f"train episodes       : {train_ep_keys.numel():,}")
    print(f"val episodes         : {val_ep_keys.numel():,}")
    print(f"total sequences      : {n_seq:,}")
    print(f"train sequences      : {train_seq_ids.numel():,} ({100*train_fraction_actual:.4f}%)")
    print(f"val sequences        : {val_seq_ids.numel():,} ({100*val_fraction_actual:.4f}%)")
    print(f"episode leakage      : 0")
    print("=" * 92)

    return {
        "gait": gait,
        "sequence_index_file": str(index_file.resolve()),
        "raw_dataset_dir": str(Path(data["raw_dataset_dir"]).resolve()),
        "train_seq_ids": train_seq_ids,
        "val_seq_ids": val_seq_ids,
        "train_episode_keys": train_ep_keys,
        "val_episode_keys": val_ep_keys,
        "num_total_sequences": int(n_seq),
        "num_train_sequences": int(train_seq_ids.numel()),
        "num_val_sequences": int(val_seq_ids.numel()),
        "num_total_episodes": int(unique_keys.numel()),
        "num_train_episodes": int(train_ep_keys.numel()),
        "num_val_episodes": int(val_ep_keys.numel()),
    }


class RunningMoments:
    """Numerically stable batch-combine mean/variance accumulator."""

    def __init__(self, dim: int):
        self.count = 0
        self.mean = torch.zeros(dim, dtype=torch.float64)
        self.m2 = torch.zeros(dim, dtype=torch.float64)
        self.min = torch.full((dim,), float("inf"), dtype=torch.float64)
        self.max = torch.full((dim,), float("-inf"), dtype=torch.float64)

    def update(self, x: torch.Tensor):
        if x.numel() == 0:
            return
        x = x.to(torch.float64)

        batch_count = x.shape[0]
        batch_mean = x.mean(dim=0)
        batch_m2 = ((x - batch_mean) ** 2).sum(dim=0)

        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
        else:
            delta = batch_mean - self.mean
            new_count = self.count + batch_count
            self.mean = self.mean + delta * (batch_count / new_count)
            self.m2 = (
                self.m2
                + batch_m2
                + delta.pow(2) * self.count * batch_count / new_count
            )
            self.count = new_count

        self.min = torch.minimum(self.min, x.min(dim=0).values)
        self.max = torch.maximum(self.max, x.max(dim=0).values)

    def finalize(self, eps: float) -> dict:
        if self.count < 2:
            raise RuntimeError("Not enough train frames for normalization.")

        # Population variance is appropriate for feature normalization.
        var = self.m2 / self.count
        std_raw = torch.sqrt(torch.clamp(var, min=0.0))

        # Constant / near-constant dimensions must not divide by zero.
        std_safe = torch.where(std_raw < eps, torch.ones_like(std_raw), std_raw)
        constant_mask = std_raw < eps

        return {
            "count": self.count,
            "mean": self.mean.to(torch.float32),
            "std": std_safe.to(torch.float32),
            "std_raw": std_raw.to(torch.float32),
            "constant_mask": constant_mask,
            "min": self.min.to(torch.float32),
            "max": self.max.to(torch.float32),
        }


def compute_shared_train_normalization(
    split_infos: list[dict],
    state_dim: int,
    eps: float,
) -> dict:
    stats = RunningMoments(state_dim)

    print("\n" + "#" * 92)
    print("COMPUTING SHARED TRAIN-ONLY STATE NORMALIZATION")
    print("#" * 92)

    per_gait_frame_counts = {}

    for info in split_infos:
        gait = info["gait"]
        dataset_dir = Path(info["raw_dataset_dir"])
        train_ep_keys = info["train_episode_keys"]
        chunks = sorted(dataset_dir.glob("chunk_*.pt"))

        if not chunks:
            raise FileNotFoundError(f"{gait}: no raw chunks in {dataset_dir}")

        gait_count = 0

        for i, chunk in enumerate(chunks):
            d = torch.load(chunk, map_location="cpu", weights_only=False)

            state = d["encoder_state"]
            env_id = d["env_id"].long()
            episode_id = d["episode_id"].long()

            if state.ndim != 2 or state.shape[1] != state_dim:
                raise RuntimeError(
                    f"{gait}/{chunk.name}: state shape {tuple(state.shape)}"
                )

            row_keys = encode_episode_key(env_id, episode_id)
            train_mask = isin_sorted(row_keys, train_ep_keys)
            selected = state[train_mask]

            if selected.numel() > 0:
                stats.update(selected)
                gait_count += selected.shape[0]

            print(
                f"[{gait} {i+1:02d}/{len(chunks):02d}] "
                f"{chunk.name}: train raw frames={int(train_mask.sum()):,}"
            )

        per_gait_frame_counts[gait] = gait_count
        print(f"{gait.upper()} train raw frames used: {gait_count:,}")

    out = stats.finalize(eps=eps)
    out["per_gait_train_raw_frames"] = per_gait_frame_counts

    print("-" * 92)
    print(f"Total train raw frames used : {out['count']:,}")
    print(f"Near-constant dimensions    : {int(out['constant_mask'].sum().item())} / {state_dim}")
    print("#" * 92)

    return out


def main():
    parser = argparse.ArgumentParser(
        description="Episode-level TVAE train/val split + shared train-only normalization."
    )
    parser.add_argument(
        "--index_dir",
        type=str,
        required=True,
        help="Directory containing trot_sequence_index.pt and bound_sequence_index.pt",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--val_fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--std_epsilon",
        type=float,
        default=1.0e-6,
        help="Dimensions with raw std below this use normalization std=1.",
    )
    args = parser.parse_args()

    if not (0.0 < args.val_fraction < 0.5):
        raise ValueError("--val_fraction must be between 0 and 0.5")

    index_dir = Path(args.index_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    trot_info = split_one_gait(
        index_dir / "trot_sequence_index.pt",
        gait="trot",
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    bound_info = split_one_gait(
        index_dir / "bound_sequence_index.pt",
        gait="bound",
        val_fraction=args.val_fraction,
        # Different deterministic shuffle for the second gait.
        seed=args.seed + 1,
    )

    norm = compute_shared_train_normalization(
        [trot_info, bound_info],
        state_dim=85,
        eps=args.std_epsilon,
    )

    split_payload = {
        "seed": args.seed,
        "val_fraction_requested": args.val_fraction,
        "split_rule": "whole (env_id, episode_id); no episode crosses train/val",
        "trot": {
            k: v for k, v in trot_info.items()
            if k not in {"train_episode_keys", "val_episode_keys"}
        },
        "bound": {
            k: v for k, v in bound_info.items()
            if k not in {"train_episode_keys", "val_episode_keys"}
        },
        # Keep these tensors for later loaders / audits.
        "trot_train_episode_keys": trot_info["train_episode_keys"],
        "trot_val_episode_keys": trot_info["val_episode_keys"],
        "bound_train_episode_keys": bound_info["train_episode_keys"],
        "bound_val_episode_keys": bound_info["val_episode_keys"],
    }

    # Save sequence IDs separately for straightforward later Dataset use.
    torch.save(
        {
            "trot_train_seq_ids": trot_info["train_seq_ids"],
            "trot_val_seq_ids": trot_info["val_seq_ids"],
            "bound_train_seq_ids": bound_info["train_seq_ids"],
            "bound_val_seq_ids": bound_info["val_seq_ids"],
        },
        output_dir / "sequence_splits.pt",
    )

    torch.save(
        {
            "state_dim": 85,
            "mean": norm["mean"],
            "std": norm["std"],
            "std_raw": norm["std_raw"],
            "constant_mask": norm["constant_mask"],
            "min": norm["min"],
            "max": norm["max"],
            "train_raw_frame_count": norm["count"],
            "per_gait_train_raw_frames": norm["per_gait_train_raw_frames"],
            "std_epsilon": args.std_epsilon,
            "normalization_rule": "(x - mean) / std; stats computed on TRAIN raw frames only",
        },
        output_dir / "state_normalization.pt",
    )

    # JSON summary excludes tensors.
    summary = {
        "seed": args.seed,
        "val_fraction_requested": args.val_fraction,
        "split_rule": "whole (env_id, episode_id); no leakage",
        "normalization_rule": "shared Trot+Bound mean/std from TRAIN raw frames only",
        "trot": {
            "total_sequences": trot_info["num_total_sequences"],
            "train_sequences": trot_info["num_train_sequences"],
            "val_sequences": trot_info["num_val_sequences"],
            "total_episodes": trot_info["num_total_episodes"],
            "train_episodes": trot_info["num_train_episodes"],
            "val_episodes": trot_info["num_val_episodes"],
        },
        "bound": {
            "total_sequences": bound_info["num_total_sequences"],
            "train_sequences": bound_info["num_train_sequences"],
            "val_sequences": bound_info["num_val_sequences"],
            "total_episodes": bound_info["num_total_episodes"],
            "train_episodes": bound_info["num_train_episodes"],
            "val_episodes": bound_info["num_val_episodes"],
        },
        "normalization": {
            "state_dim": 85,
            "total_train_raw_frames": norm["count"],
            "trot_train_raw_frames": norm["per_gait_train_raw_frames"]["trot"],
            "bound_train_raw_frames": norm["per_gait_train_raw_frames"]["bound"],
            "near_constant_dimensions": int(norm["constant_mask"].sum().item()),
        },
    }

    (output_dir / "split_normalization_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 92)
    print("STEP 2 COMPLETE")
    print(f"sequence splits    : {output_dir / 'sequence_splits.pt'}")
    print(f"normalization      : {output_dir / 'state_normalization.pt'}")
    print(f"summary            : {output_dir / 'split_normalization_summary.json'}")
    print("=" * 92)


if __name__ == "__main__":
    main()
