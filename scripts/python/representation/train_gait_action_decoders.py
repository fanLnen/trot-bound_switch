#!/usr/bin/env python3
"""Train gait-specific ACTION decoders from a frozen shared TVAE encoder.

Pipeline:
  [s_t,s_{t+1},s_{t+2}] -> frozen encoder -> z -> gait action decoder -> a_hat_t
Target:
  expert policy_action at seq_indices[:, 0]

Deployment later:
  decoded raw action -> q_des = q_default + 0.25 * action -> original PD (Kp=40,Kd=1)

Notes:
- Trot and Bound are trained as separate decoders.
- Training samples z = mu + sigma*eps; validation uses z = mu.
- Decoder output is unbounded raw 16-D PPO action: NO tanh.
- Original episode-level train/val split is reused.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn

from train_shared_state_tvae import ModelConfig, SharedStateTVAE

STATE_DIM = 85
SEQ_LEN = 3
ACTION_DIM = 16


def seed_all(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class DecoderCfg:
    latent_dim: int = 16
    hidden_dim: int = 64
    action_dim: int = 16


class ActionDecoder(nn.Module):
    def __init__(self, cfg: DecoderCfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.action_dim),
        )

    def forward(self, z):
        return self.net(z)


class GaitStore:
    """Load one gait's encoder_state + policy_action into CPU RAM."""

    def __init__(self, index_file: Path, gait: str):
        idx = torch.load(index_file, map_location="cpu", weights_only=False)
        if str(idx["gait"]).lower() != gait:
            raise RuntimeError(f"gait mismatch in {index_file}")
        if int(idx["state_dim"]) != STATE_DIM or int(idx["sequence_length"]) != SEQ_LEN:
            raise RuntimeError("unexpected state/sequence dimensions")

        self.gait = gait
        self.seq_indices = idx["seq_indices"].long().contiguous()
        raw_dir = Path(idx["raw_dataset_dir"]).resolve()
        names = list(idx["chunk_files"])
        sizes = [int(x) for x in idx["chunk_sizes"].tolist()]
        n_total = sum(sizes)

        self.states = torch.empty((n_total, STATE_DIM), dtype=torch.float32)
        self.actions = torch.empty((n_total, ACTION_DIM), dtype=torch.float32)

        print(f"\nLoading {gait.upper()} state + policy_action into CPU RAM")
        off = 0
        for i, (name, n) in enumerate(zip(names, sizes)):
            d = torch.load(raw_dir / name, map_location="cpu", weights_only=False)
            s = d["encoder_state"].float()
            a = d["policy_action"].float()
            if s.shape != (n, STATE_DIM):
                raise RuntimeError(f"{name}: bad state shape {tuple(s.shape)}")
            if a.shape != (n, ACTION_DIM):
                raise RuntimeError(f"{name}: bad action shape {tuple(a.shape)}")
            if not torch.isfinite(s).all() or not torch.isfinite(a).all():
                raise RuntimeError(f"{name}: non-finite data")
            self.states[off:off+n].copy_(s)
            self.actions[off:off+n].copy_(a)
            off += n
            print(f"  [{i+1:02d}/{len(names):02d}] {name}: {n:,}")

        print(f"{gait.upper()} frames={n_total:,}, sequences={self.seq_indices.shape[0]:,}")
        print(f"{gait.upper()} raw action range=[{self.actions.min():.5f}, {self.actions.max():.5f}]")

    def batch(self, seq_ids: torch.Tensor):
        trip = self.seq_indices[seq_ids.long()]
        x = self.states[trip]                    # [B,3,85]
        a = self.actions[trip[:, 0]]             # [B,16], aligned with s_t
        return x, a


def load_encoder(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ckpt["model_config"])
    model = SharedStateTVAE(cfg)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    mean = ckpt["state_mean"].float().view(1, 1, STATE_DIM).to(device)
    std = ckpt["state_std"].float().view(1, 1, STATE_DIM).to(device)
    return model, cfg, mean, std, ckpt


@torch.no_grad()
def encode(encoder, x, mean, std, sample: bool):
    x = (x - mean) / std
    mu, logvar = encoder.encode(x)
    if sample:
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
    else:
        z = mu
    return z


def choose_ids(ids, max_samples, shuffle, gen):
    ids = ids.long()
    n = ids.numel() if max_samples <= 0 else min(ids.numel(), max_samples)
    if shuffle:
        return ids[torch.randperm(ids.numel(), generator=gen)[:n]]
    return ids[:n]


def train_epoch(encoder, decoder, opt, store, ids, mean, std, device,
                batch_size, max_samples, gen, log_every):
    decoder.train()
    ids = choose_ids(ids, max_samples, True, gen)
    sse = abse = 0.0
    n_elem = 0

    for bi, start in enumerate(range(0, ids.numel(), batch_size), 1):
        x_cpu, a_cpu = store.batch(ids[start:start+batch_size])
        x = x_cpu.to(device)
        target = a_cpu.to(device)
        z = encode(encoder, x, mean, std, sample=True)

        pred = decoder(z)
        loss = ((pred - target) ** 2).mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at batch {bi}")

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        with torch.no_grad():
            e = pred - target
            sse += e.pow(2).sum().item()
            abse += e.abs().sum().item()
            n_elem += e.numel()

        if log_every > 0 and bi % log_every == 0:
            mse = sse / n_elem
            print(f"    train batch {bi:,} | mse={mse:.8f} | rmse={math.sqrt(mse):.6f} | mae={abse/n_elem:.6f}")

    mse = sse / n_elem
    return {"mse": mse, "rmse": math.sqrt(mse), "mae": abse/n_elem, "nseq": int(ids.numel())}


@torch.inference_mode()
def validate(encoder, decoder, store, ids, mean, std, device,
             batch_size, max_samples):
    decoder.eval()
    ids = choose_ids(ids, max_samples, False, torch.Generator())
    preds, tgts = [], []

    for start in range(0, ids.numel(), batch_size):
        x_cpu, a_cpu = store.batch(ids[start:start+batch_size])
        x = x_cpu.to(device)
        target = a_cpu.to(device)
        z = encode(encoder, x, mean, std, sample=False)
        pred = decoder(z)
        preds.append(pred.cpu())
        tgts.append(target.cpu())

    pred = torch.cat(preds)
    target = torch.cat(tgts)
    e = pred - target
    mse = e.pow(2).mean().item()
    rmse = math.sqrt(mse)
    mae = e.abs().mean().item()
    abs_e = e.abs().flatten()

    per_rmse = torch.sqrt(e.pow(2).mean(0))
    per_mae = e.abs().mean(0)
    ss_res = e.pow(2).sum(0)
    ss_tot = (target - target.mean(0)).pow(2).sum(0)
    per_r2 = torch.where(ss_tot > 1e-12, 1 - ss_res/ss_tot, torch.zeros_like(ss_tot))

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "p95": torch.quantile(abs_e, 0.95).item(),
        "p99": torch.quantile(abs_e, 0.99).item(),
        "maxerr": abs_e.max().item(),
        "pred_min": pred.min().item(),
        "pred_max": pred.max().item(),
        "target_min": target.min().item(),
        "target_max": target.max().item(),
        "per_rmse": per_rmse,
        "per_mae": per_mae,
        "per_r2": per_r2,
        "nseq": int(ids.numel()),
    }


def save_ckpt(path, gait, decoder, opt, epoch, best_mse, dec_cfg, enc_path, args):
    torch.save({
        "gait": gait,
        "epoch": epoch,
        "best_val_mse": best_mse,
        "decoder_config": asdict(dec_cfg),
        "decoder_state_dict": decoder.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "encoder_checkpoint": str(enc_path),
        "target": "policy_action at seq_indices[:,0]",
        "output_activation": "none",
        "action_mapping": "q_des = q_default + 0.25 * decoded_action",
        "actuator": {"Kp": 40.0, "Kd": 1.0, "effort_limit_Nm": 48.0},
        "args": vars(args),
    }, path)


def write_joint_csv(path, m):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["joint_index", "rmse_action", "mae_action", "r2"])
        for j in range(ACTION_DIM):
            w.writerow([j, float(m["per_rmse"][j]), float(m["per_mae"][j]), float(m["per_r2"][j])])


def train_gait(gait, encoder, enc_cfg, mean, std, enc_path,
               index_dir, splits, out_root, device, args):
    store = GaitStore(index_dir / f"{gait}_sequence_index.pt", gait)
    tr = splits[f"{gait}_train_seq_ids"].long()
    va = splits[f"{gait}_val_seq_ids"].long()

    out = out_root / gait
    out.mkdir(parents=True, exist_ok=True)

    dec_cfg = DecoderCfg(enc_cfg.latent_dim, args.hidden_dim, ACTION_DIM)
    decoder = ActionDecoder(dec_cfg).to(device)
    opt = torch.optim.Adam(decoder.parameters(), lr=args.lr)

    # sanity
    x0, a0 = store.batch(tr[:min(args.batch_size, tr.numel())])
    with torch.no_grad():
        z0 = encode(encoder, x0.to(device), mean, std, sample=False)
        p0 = decoder(z0)
    print("\n" + "="*100)
    print(f"{gait.upper()} ACTION DECODER")
    print(f"[SANITY] x={tuple(x0.shape)}, z={tuple(z0.shape)}, pred={tuple(p0.shape)}, target={tuple(a0.shape)}")
    print(f"decoder: {enc_cfg.latent_dim}->{args.hidden_dim}->{args.hidden_dim}->16, output activation: NONE")
    print(f"train/val sequences: {tr.numel():,} / {va.numel():,}")
    print("="*100)

    (out / "config.json").write_text(json.dumps({
        "gait": gait,
        "encoder_checkpoint": str(enc_path),
        "decoder_config": asdict(dec_cfg),
        "target": "raw expert policy_action",
        "output_activation": "none",
        "action_mapping": "q_des = q_default + 0.25 * action",
        "actuator": {"Kp": 40.0, "Kd": 1.0, "effort_limit_Nm": 48.0},
    }, indent=2), encoding="utf-8")

    metrics = out / "metrics.csv"
    with metrics.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "epoch", "train_mse", "train_rmse", "train_mae", "train_sequences",
            "val_mse", "val_rmse", "val_mae", "val_p95", "val_p99", "val_maxerr", "val_sequences", "seconds"
        ])

    gen = torch.Generator().manual_seed(args.seed + (0 if gait == "trot" else 1000))
    best = float("inf")
    best_metrics = None

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tm = train_epoch(encoder, decoder, opt, store, tr, mean, std, device,
                         args.batch_size, args.train_samples_per_gait_per_epoch,
                         gen, args.log_every)
        vm = validate(encoder, decoder, store, va, mean, std, device,
                      args.batch_size, args.val_samples_per_gait)
        sec = time.time() - t0

        print(f"\n[{gait.upper()} EPOCH {epoch:03d}/{args.epochs}] "
              f"train mse={tm['mse']:.8f} rmse={tm['rmse']:.6f} mae={tm['mae']:.6f} | "
              f"val mse={vm['mse']:.8f} rmse={vm['rmse']:.6f} mae={vm['mae']:.6f} "
              f"p95={vm['p95']:.6f} p99={vm['p99']:.6f}")

        with metrics.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch, tm["mse"], tm["rmse"], tm["mae"], tm["nseq"],
                vm["mse"], vm["rmse"], vm["mae"], vm["p95"], vm["p99"], vm["maxerr"], vm["nseq"], sec
            ])

        save_ckpt(out / "last.pt", gait, decoder, opt, epoch, best, dec_cfg, enc_path, args)

        if vm["mse"] < best:
            best = vm["mse"]
            best_metrics = vm
            save_ckpt(out / "best.pt", gait, decoder, opt, epoch, best, dec_cfg, enc_path, args)
            write_joint_csv(out / "best_per_joint_metrics.csv", vm)
            (out / "best_summary.json").write_text(json.dumps({
                "best_epoch": epoch,
                "val_mse": vm["mse"],
                "val_rmse": vm["rmse"],
                "val_mae": vm["mae"],
                "val_p95": vm["p95"],
                "val_p99": vm["p99"],
                "val_maxerr": vm["maxerr"],
                "pred_action_range": [vm["pred_min"], vm["pred_max"]],
                "target_action_range": [vm["target_min"], vm["target_max"]],
                "per_joint_rmse": [float(x) for x in vm["per_rmse"]],
                "per_joint_mae": [float(x) for x in vm["per_mae"]],
                "per_joint_r2": [float(x) for x in vm["per_r2"]],
            }, indent=2), encoding="utf-8")
            print(f"[BEST {gait.upper()}] val MSE -> {best:.8f}")

        if args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0:
            save_ckpt(out / f"epoch_{epoch:04d}.pt", gait, decoder, opt, epoch, best, dec_cfg, enc_path, args)

    print(f"\n{gait.upper()} complete: {out / 'best.pt'}")
    result = {
        "val_mse": best_metrics["mse"], "val_rmse": best_metrics["rmse"],
        "val_mae": best_metrics["mae"], "val_p95": best_metrics["p95"], "val_p99": best_metrics["p99"]
    }

    del store, decoder, opt
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder_checkpoint", required=True)
    p.add_argument("--index_dir", required=True)
    p.add_argument("--split_norm_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--gait", choices=["trot", "bound", "both"], default="both")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--train_samples_per_gait_per_epoch", type=int, default=0)
    p.add_argument("--val_samples_per_gait", type=int, default=0)
    p.add_argument("--checkpoint_every", type=int, default=10)
    p.add_argument("--log_every", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)

    enc_path = Path(args.encoder_checkpoint).expanduser().resolve()
    index_dir = Path(args.index_dir).expanduser().resolve()
    split_dir = Path(args.split_norm_dir).expanduser().resolve()
    out_root = Path(args.output_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    splits = torch.load(split_dir / "sequence_splits.pt", map_location="cpu", weights_only=False)
    encoder, enc_cfg, mean, std, enc_ckpt = load_encoder(enc_path, device)

    print("="*100)
    print("GAIT-SPECIFIC ACTION DECODER TRAINING")
    print(f"device             : {device}")
    if device.type == "cuda":
        print(f"GPU                : {torch.cuda.get_device_name(device)}")
    print(f"encoder checkpoint : {enc_path}")
    print(f"encoder epoch      : {enc_ckpt.get('epoch', 'unknown')}")
    print(f"latent dim         : {enc_cfg.latent_dim}")
    print("target             : raw policy_action[t]")
    print("decoder output     : raw 16-D action, no tanh")
    print("="*100)

    gaits = ["trot", "bound"] if args.gait == "both" else [args.gait]
    summary = {}
    for gait in gaits:
        summary[gait] = train_gait(
            gait, encoder, enc_cfg, mean, std, enc_path,
            index_dir, splits, out_root, device, args
        )

    (out_root / "overall_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "#"*100)
    print("ALL REQUESTED ACTION DECODERS COMPLETE")
    for gait, m in summary.items():
        print(f"{gait.upper():5s} | RMSE={m['val_rmse']:.6f} | MAE={m['val_mae']:.6f} | P95={m['val_p95']:.6f}")
    print(f"summary: {out_root / 'overall_summary.json'}")
    print("#"*100)


if __name__ == "__main__":
    main()
