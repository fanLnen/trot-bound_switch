#!/usr/bin/env python3
"""
Train the shared state TVAE (Encoder + State Decoder) on Trot and Bound data.

Paper-aligned settings used by default:
- sequence length: 3
- state dimension: 85 for the user's 16-DOF robot (paper: 69 for 12-DOF)
- latent dimension: 16
- Transformer hidden/model dimension: 64
- attention heads: 8
- Transformer layers: 2
- activation: GELU
- batch size: 500
- learning rate: 1e-3
- epochs: 30 in the successful project run
- KL weight: 0.1

Important implementation note:
The paper explicitly specifies the Transformer encoder hyperparameters and that
the first two special tokens parameterize mu and log-variance. It does not
fully enumerate the layer-by-layer topology of the *state decoder*. Here the
state decoder is implemented as a symmetric lightweight Transformer:
    z -> 3 learned time queries -> 2-layer/8-head/64-d Transformer -> 3x85 state
This choice is saved in config.json/checkpoints so it is never confused with
an explicitly reported paper detail.

This stage trains ONLY:
    [s_t, s_t+1, s_t+2] -> shared TVAE -> reconstructed 3-frame state

It does NOT use torque yet.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------


@dataclass
class ModelConfig:
    state_dim: int = 85
    sequence_length: int = 3
    latent_dim: int = 16
    d_model: int = 64
    nhead: int = 8
    num_layers: int = 2
    dim_feedforward: int = 64
    dropout: float = 0.0


class SharedStateTVAE(nn.Module):
    """
    Transformer VAE with two special posterior tokens:
      token 0 -> mu
      token 1 -> log(sigma^2)

    Input:
      x: [B, 3, 85]

    Output:
      x_hat: [B, 3, 85]
      mu: [B, 16]
      logvar: [B, 16]
      z: [B, 16]
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # ----- shared state encoder -----
        self.state_input_proj = nn.Linear(cfg.state_dim, cfg.d_model)

        self.mu_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.logvar_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))

        # 2 posterior tokens + 3 state-frame tokens
        self.encoder_pos_embed = nn.Parameter(
            torch.zeros(1, 2 + cfg.sequence_length, cfg.d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.num_layers,
        )

        self.mu_head = nn.Linear(cfg.d_model, cfg.latent_dim)
        self.logvar_head = nn.Linear(cfg.d_model, cfg.latent_dim)

        # ----- state reconstruction decoder -----
        self.latent_to_hidden = nn.Linear(cfg.latent_dim, cfg.d_model)

        self.decoder_queries = nn.Parameter(
            torch.zeros(1, cfg.sequence_length, cfg.d_model)
        )
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, cfg.sequence_length, cfg.d_model)
        )

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.state_decoder = nn.TransformerEncoder(
            decoder_layer,
            num_layers=cfg.num_layers,
        )
        self.state_output_proj = nn.Linear(cfg.d_model, cfg.state_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.mu_token, mean=0.0, std=0.02)
        nn.init.normal_(self.logvar_token, mean=0.0, std=0.02)
        nn.init.normal_(self.encoder_pos_embed, mean=0.0, std=0.02)
        nn.init.normal_(self.decoder_queries, mean=0.0, std=0.02)
        nn.init.normal_(self.decoder_pos_embed, mean=0.0, std=0.02)

    def encode(self, x: torch.Tensor):
        b = x.shape[0]

        state_tokens = self.state_input_proj(x)  # [B, 3, 64]

        mu_tok = self.mu_token.expand(b, -1, -1)
        logvar_tok = self.logvar_token.expand(b, -1, -1)

        tokens = torch.cat((mu_tok, logvar_tok, state_tokens), dim=1)
        tokens = tokens + self.encoder_pos_embed

        h = self.encoder(tokens)

        mu = self.mu_head(h[:, 0])
        logvar = self.logvar_head(h[:, 1])
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_state(self, z: torch.Tensor):
        latent = self.latent_to_hidden(z).unsqueeze(1)  # [B,1,64]
        h = (
            self.decoder_queries
            + self.decoder_pos_embed
            + latent
        )
        h = self.state_decoder(h)
        return self.state_output_proj(h)  # [B,3,85]

    def forward(self, x: torch.Tensor, sample_latent: bool = True):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar) if sample_latent else mu
        x_hat = self.decode_state(z)
        return x_hat, mu, logvar, z


# -----------------------------------------------------------------------------
# Raw state store + sequence index
# -----------------------------------------------------------------------------


class GaitSequenceStore:
    """
    Loads only encoder_state from raw chunk files into CPU RAM and keeps the
    compact legal-sequence index.

    For 2M x 85 float32 frames, one gait is ~680 MB of state data.
    """

    def __init__(self, sequence_index_file: Path, gait: str):
        self.sequence_index_file = sequence_index_file.resolve()
        idx = torch.load(
            self.sequence_index_file,
            map_location="cpu",
            weights_only=False,
        )

        if str(idx["gait"]).lower() != gait:
            raise RuntimeError(
                f"{self.sequence_index_file}: gait={idx['gait']} expected={gait}"
            )

        self.gait = gait
        self.state_dim = int(idx["state_dim"])
        self.sequence_length = int(idx["sequence_length"])
        self.seq_indices = idx["seq_indices"].to(torch.int64).contiguous()

        if self.state_dim != 85:
            raise RuntimeError(f"{gait}: expected state_dim=85, got {self.state_dim}")
        if self.sequence_length != 3:
            raise RuntimeError(
                f"{gait}: expected sequence_length=3, got {self.sequence_length}"
            )

        raw_dir = Path(idx["raw_dataset_dir"]).resolve()
        chunk_files = list(idx["chunk_files"])
        chunk_sizes = idx["chunk_sizes"].tolist()
        total_rows = int(sum(chunk_sizes))

        self.raw_dataset_dir = raw_dir
        self.states = torch.empty(
            (total_rows, self.state_dim),
            dtype=torch.float32,
        )

        offset = 0
        print(f"\nLoading {gait} raw states into CPU RAM:")
        for i, (name, expected_n) in enumerate(zip(chunk_files, chunk_sizes)):
            path = raw_dir / name
            d = torch.load(path, map_location="cpu", weights_only=False)
            s = d["encoder_state"].to(torch.float32)

            if s.shape != (expected_n, self.state_dim):
                raise RuntimeError(
                    f"{path}: state shape {tuple(s.shape)} "
                    f"expected {(expected_n, self.state_dim)}"
                )

            self.states[offset : offset + expected_n].copy_(s)
            offset += expected_n

            print(
                f"  [{i+1:02d}/{len(chunk_files):02d}] "
                f"{name}: {expected_n:,} rows"
            )

        if offset != total_rows:
            raise RuntimeError(f"{gait}: raw-state row-count mismatch")

        if not torch.isfinite(self.states).all():
            raise RuntimeError(f"{gait}: non-finite raw state found")

        print(
            f"{gait.upper()} loaded: "
            f"{self.states.shape[0]:,} frames, "
            f"{self.seq_indices.shape[0]:,} legal sequences"
        )

    def get_sequences(self, seq_ids: torch.Tensor) -> torch.Tensor:
        """
        seq_ids: IDs into seq_indices, [B]
        return: raw states [B, 3, 85]
        """
        raw_rows = self.seq_indices[seq_ids]  # [B,3]
        return self.states[raw_rows]          # [B,3,85]


# -----------------------------------------------------------------------------
# Loss / metrics
# -----------------------------------------------------------------------------


def tvae_loss(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta_kl: float,
):
    # Paper form:
    # L_recon = sum_{three frames} ||s - s_hat||_2^2
    # We average that per-sample sum across the mini-batch.
    sq = (x_hat - x).pow(2)
    recon_sum = sq.sum(dim=(1, 2)).mean()

    # Useful scale-independent reporting metric only.
    recon_mse = sq.mean()

    kl_per_sample = -0.5 * torch.sum(
        1.0 + logvar - mu.pow(2) - logvar.exp(),
        dim=1,
    )
    kl = kl_per_sample.mean()

    total = recon_sum + beta_kl * kl
    return total, recon_sum, recon_mse, kl


# -----------------------------------------------------------------------------
# Balanced Trot/Bound batch iteration
# -----------------------------------------------------------------------------


def select_epoch_ids(
    split_ids: torch.Tensor,
    n_requested: int,
    shuffle: bool,
    generator: torch.Generator,
):
    n = split_ids.numel()
    if n_requested > 0:
        n = min(n, n_requested)

    if shuffle:
        perm = torch.randperm(split_ids.numel(), generator=generator)[:n]
        return split_ids[perm]
    return split_ids[:n]


def balanced_batches(
    trot_ids: torch.Tensor,
    bound_ids: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    generator: torch.Generator,
    samples_per_gait: int,
):
    if batch_size < 2:
        raise ValueError("batch_size must be >= 2")
    if batch_size % 2 != 0:
        raise ValueError(
            "Use an even batch_size so every batch is exactly 50% Trot / 50% Bound."
        )

    half = batch_size // 2

    t = select_epoch_ids(
        trot_ids, samples_per_gait, shuffle, generator
    )
    b = select_epoch_ids(
        bound_ids, samples_per_gait, shuffle, generator
    )

    # Equal gait contribution in every epoch.
    n = min(t.numel(), b.numel())
    t = t[:n]
    b = b[:n]

    for start in range(0, n, half):
        t_batch = t[start : start + half]
        b_batch = b[start : start + half]

        # Keep the final batch balanced too.
        m = min(t_batch.numel(), b_batch.numel())
        if m == 0:
            break

        yield t_batch[:m], b_batch[:m]


# -----------------------------------------------------------------------------
# Train / validation
# -----------------------------------------------------------------------------


def make_batch(
    trot_store: GaitSequenceStore,
    bound_store: GaitSequenceStore,
    trot_seq_ids: torch.Tensor,
    bound_seq_ids: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
):
    xt = trot_store.get_sequences(trot_seq_ids)
    xb = bound_store.get_sequences(bound_seq_ids)

    x = torch.cat((xt, xb), dim=0)
    x = x.to(device, non_blocking=False)

    # shared train-only normalization
    x = (x - mean) / std

    return x


def run_train_epoch(
    model,
    optimizer,
    trot_store,
    bound_store,
    trot_ids,
    bound_ids,
    mean,
    std,
    device,
    batch_size,
    beta_kl,
    samples_per_gait,
    generator,
    log_every,
):
    model.train()

    total_weight = 0
    total_loss = 0.0
    total_recon_sum = 0.0
    total_recon_mse = 0.0
    total_kl = 0.0

    batches = balanced_batches(
        trot_ids,
        bound_ids,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        samples_per_gait=samples_per_gait,
    )

    for batch_idx, (tid, bid) in enumerate(batches, start=1):
        x = make_batch(
            trot_store, bound_store, tid, bid, mean, std, device
        )

        optimizer.zero_grad(set_to_none=True)

        x_hat, mu, logvar, _ = model(x, sample_latent=True)
        loss, recon_sum, recon_mse, kl = tvae_loss(
            x_hat, x, mu, logvar, beta_kl
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite training loss at batch {batch_idx}: "
                f"loss={loss.item()}, recon={recon_sum.item()}, kl={kl.item()}"
            )

        loss.backward()
        optimizer.step()

        n = x.shape[0]
        total_weight += n
        total_loss += loss.item() * n
        total_recon_sum += recon_sum.item() * n
        total_recon_mse += recon_mse.item() * n
        total_kl += kl.item() * n

        if log_every > 0 and batch_idx % log_every == 0:
            print(
                f"    train batch {batch_idx:,} | "
                f"loss={total_loss/total_weight:.6f} | "
                f"recon_sum={total_recon_sum/total_weight:.6f} | "
                f"mse={total_recon_mse/total_weight:.8f} | "
                f"kl={total_kl/total_weight:.6f}"
            )

    return {
        "loss": total_loss / total_weight,
        "recon_sum": total_recon_sum / total_weight,
        "recon_mse": total_recon_mse / total_weight,
        "kl": total_kl / total_weight,
        "samples": total_weight,
    }


@torch.inference_mode()
def run_validation(
    model,
    trot_store,
    bound_store,
    trot_ids,
    bound_ids,
    mean,
    std,
    device,
    batch_size,
    beta_kl,
    samples_per_gait,
    generator,
):
    model.eval()

    total_weight = 0
    total_loss = 0.0
    total_recon_sum = 0.0
    total_recon_mse = 0.0
    total_kl = 0.0

    batches = balanced_batches(
        trot_ids,
        bound_ids,
        batch_size=batch_size,
        shuffle=False,
        generator=generator,
        samples_per_gait=samples_per_gait,
    )

    for tid, bid in batches:
        x = make_batch(
            trot_store, bound_store, tid, bid, mean, std, device
        )

        # Deterministic validation: decode posterior mean.
        x_hat, mu, logvar, _ = model(x, sample_latent=False)
        loss, recon_sum, recon_mse, kl = tvae_loss(
            x_hat, x, mu, logvar, beta_kl
        )

        n = x.shape[0]
        total_weight += n
        total_loss += loss.item() * n
        total_recon_sum += recon_sum.item() * n
        total_recon_mse += recon_mse.item() * n
        total_kl += kl.item() * n

    return {
        "loss": total_loss / total_weight,
        "recon_sum": total_recon_sum / total_weight,
        "recon_mse": total_recon_mse / total_weight,
        "kl": total_kl / total_weight,
        "samples": total_weight,
    }


@torch.inference_mode()
def save_latent_snapshot(
    model,
    trot_store,
    bound_store,
    trot_val_ids,
    bound_val_ids,
    mean,
    std,
    device,
    out_file: Path,
    per_gait: int,
    batch_size: int,
):
    model.eval()

    def encode_store(store, ids, gait_label):
        ids = ids[: min(per_gait, ids.numel())]
        mu_parts = []
        seq_id_parts = []

        for start in range(0, ids.numel(), batch_size):
            batch_ids = ids[start : start + batch_size]
            x = store.get_sequences(batch_ids).to(device)
            x = (x - mean) / std

            mu, _ = model.encode(x)
            mu_parts.append(mu.cpu())
            seq_id_parts.append(batch_ids.cpu())

        return (
            torch.cat(mu_parts, dim=0),
            torch.cat(seq_id_parts, dim=0),
            torch.full(
                (sum(x.shape[0] for x in mu_parts),),
                gait_label,
                dtype=torch.int64,
            ),
        )

    t_mu, t_ids, t_g = encode_store(trot_store, trot_val_ids, 0)
    b_mu, b_ids, b_g = encode_store(bound_store, bound_val_ids, 1)

    torch.save(
        {
            "latent_mu": torch.cat((t_mu, b_mu), dim=0),
            "gait": torch.cat((t_g, b_g), dim=0),  # 0=trot, 1=bound
            "trot_seq_ids": t_ids,
            "bound_seq_ids": b_ids,
            "note": "posterior mean mu; gait 0=trot, 1=bound",
        },
        out_file,
    )


# -----------------------------------------------------------------------------
# Checkpoint
# -----------------------------------------------------------------------------


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    epoch,
    best_val_loss,
    model_cfg,
    args,
    mean,
    std,
):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "model_config": asdict(model_cfg),
            "train_args": vars(args),
            "state_mean": mean.detach().cpu(),
            "state_std": std.detach().cpu(),
        },
        path,
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--index_dir",
        type=str,
        required=True,
        help="Contains trot_sequence_index.pt and bound_sequence_index.pt",
    )
    p.add_argument(
        "--split_norm_dir",
        type=str,
        required=True,
        help="Contains sequence_splits.pt and state_normalization.pt",
    )
    p.add_argument("--output_dir", type=str, required=True)

    # Paper-aligned defaults.
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta_kl", type=float, default=0.1)

    p.add_argument("--latent_dim", type=int, default=16)
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dim_feedforward", type=int, default=64)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )

    # 0 means use all available split samples from each gait per epoch.
    # This is useful for smoke tests without changing the full-dataset setup.
    p.add_argument(
        "--train_samples_per_gait_per_epoch",
        type=int,
        default=0,
    )
    p.add_argument(
        "--val_samples_per_gait",
        type=int,
        default=0,
    )

    p.add_argument("--val_every", type=int, default=1)
    p.add_argument("--checkpoint_every", type=int, default=10)
    p.add_argument("--log_every", type=int, default=500)
    p.add_argument("--latent_snapshot_per_gait", type=int, default=10000)

    args = p.parse_args()

    if args.batch_size % 2 != 0:
        raise ValueError("--batch_size must be even for balanced Trot/Bound batches.")
    if args.latent_dim != 16:
        print(
            f"[WARNING] latent_dim={args.latent_dim}; "
            "paper-aligned value is 16."
        )

    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    index_dir = Path(args.index_dir).expanduser().resolve()
    split_norm_dir = Path(args.split_norm_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Load split + normalization
    # -------------------------------------------------------------------------

    splits = torch.load(
        split_norm_dir / "sequence_splits.pt",
        map_location="cpu",
        weights_only=False,
    )
    norm = torch.load(
        split_norm_dir / "state_normalization.pt",
        map_location="cpu",
        weights_only=False,
    )

    if int(norm["state_dim"]) != 85:
        raise RuntimeError("Expected normalization state_dim=85.")

    mean_cpu = norm["mean"].to(torch.float32)
    std_cpu = norm["std"].to(torch.float32)

    if mean_cpu.shape != (85,) or std_cpu.shape != (85,):
        raise RuntimeError(
            f"Bad normalization shapes: mean={mean_cpu.shape}, std={std_cpu.shape}"
        )
    if torch.any(std_cpu <= 0):
        raise RuntimeError("Normalization std contains non-positive values.")

    mean = mean_cpu.to(device).view(1, 1, 85)
    std = std_cpu.to(device).view(1, 1, 85)

    # -------------------------------------------------------------------------
    # Load raw state stores
    # -------------------------------------------------------------------------

    trot_store = GaitSequenceStore(
        index_dir / "trot_sequence_index.pt",
        gait="trot",
    )
    bound_store = GaitSequenceStore(
        index_dir / "bound_sequence_index.pt",
        gait="bound",
    )

    trot_train = splits["trot_train_seq_ids"].long()
    trot_val = splits["trot_val_seq_ids"].long()
    bound_train = splits["bound_train_seq_ids"].long()
    bound_val = splits["bound_val_seq_ids"].long()

    # Range checks.
    for name, ids, store in (
        ("trot_train", trot_train, trot_store),
        ("trot_val", trot_val, trot_store),
        ("bound_train", bound_train, bound_store),
        ("bound_val", bound_val, bound_store),
    ):
        if ids.numel() == 0:
            raise RuntimeError(f"{name} is empty.")
        if int(ids.max()) >= store.seq_indices.shape[0]:
            raise RuntimeError(f"{name} contains out-of-range sequence IDs.")

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------

    model_cfg = ModelConfig(
        state_dim=85,
        sequence_length=3,
        latent_dim=args.latent_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=0.0,
    )

    model = SharedStateTVAE(model_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    n_params = sum(p.numel() for p in model.parameters())

    config = {
        "model_config": asdict(model_cfg),
        "training": vars(args),
        "paper_aligned_defaults": {
            "latent_dim": 16,
            "d_model": 64,
            "nhead": 8,
            "num_layers": 2,
            "dim_feedforward": 64,
            "activation": "GELU",
            "batch_size": 500,
            "learning_rate": 0.001,
            "epochs": 500,
            "beta_kl": 0.1,
        },
        "implementation_note": (
            "State decoder topology is a symmetric lightweight Transformer "
            "implementation choice because the paper does not fully enumerate "
            "its layer-by-layer topology."
        ),
        "num_parameters": n_params,
    }
    (out_dir / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 100)
    print("SHARED STATE TVAE TRAINING")
    print("=" * 100)
    print(f"device                      : {device}")
    if device.type == "cuda":
        print(f"GPU                         : {torch.cuda.get_device_name(device)}")
    print(f"parameters                  : {n_params:,}")
    print(f"input                       : [B, 3, 85]")
    print(f"latent                      : {args.latent_dim}")
    print(f"Transformer                 : d={args.d_model}, heads={args.nhead}, layers={args.num_layers}")
    print(f"batch size                  : {args.batch_size} (50% Trot / 50% Bound)")
    print(f"learning rate               : {args.lr}")
    print(f"KL weight                   : {args.beta_kl}")
    print(f"epochs                      : {args.epochs}")
    print(f"Trot train/val sequences    : {trot_train.numel():,} / {trot_val.numel():,}")
    print(f"Bound train/val sequences   : {bound_train.numel():,} / {bound_val.numel():,}")
    print("=" * 100)

    # Initial data/model sanity check.
    sanity_t = trot_train[: min(250, trot_train.numel())]
    sanity_b = bound_train[: min(250, bound_train.numel())]
    x0 = make_batch(
        trot_store, bound_store, sanity_t, sanity_b, mean, std, device
    )
    with torch.inference_mode():
        y0, mu0, logvar0, z0 = model(x0, sample_latent=False)

    print(
        f"[SANITY] x={tuple(x0.shape)}, recon={tuple(y0.shape)}, "
        f"mu={tuple(mu0.shape)}, logvar={tuple(logvar0.shape)}, z={tuple(z0.shape)}"
    )
    if not all(
        torch.isfinite(t).all()
        for t in (x0, y0, mu0, logvar0, z0)
    ):
        raise RuntimeError("Non-finite value in initial sanity forward pass.")

    # -------------------------------------------------------------------------
    # Metrics CSV
    # -------------------------------------------------------------------------

    metrics_file = out_dir / "metrics.csv"
    with metrics_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_recon_sum",
                "train_recon_mse",
                "train_kl",
                "train_samples",
                "val_loss",
                "val_recon_sum",
                "val_recon_mse",
                "val_kl",
                "val_samples",
                "epoch_seconds",
            ]
        )

    gen = torch.Generator()
    gen.manual_seed(args.seed)

    best_val_loss = float("inf")

    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        train_m = run_train_epoch(
            model=model,
            optimizer=optimizer,
            trot_store=trot_store,
            bound_store=bound_store,
            trot_ids=trot_train,
            bound_ids=bound_train,
            mean=mean,
            std=std,
            device=device,
            batch_size=args.batch_size,
            beta_kl=args.beta_kl,
            samples_per_gait=args.train_samples_per_gait_per_epoch,
            generator=gen,
            log_every=args.log_every,
        )

        val_m = {
            "loss": math.nan,
            "recon_sum": math.nan,
            "recon_mse": math.nan,
            "kl": math.nan,
            "samples": 0,
        }

        if epoch % args.val_every == 0:
            val_m = run_validation(
                model=model,
                trot_store=trot_store,
                bound_store=bound_store,
                trot_ids=trot_val,
                bound_ids=bound_val,
                mean=mean,
                std=std,
                device=device,
                batch_size=args.batch_size,
                beta_kl=args.beta_kl,
                samples_per_gait=args.val_samples_per_gait,
                generator=gen,
            )

        elapsed = time.time() - start_time

        print(
            f"\n[EPOCH {epoch:03d}/{args.epochs}] "
            f"train loss={train_m['loss']:.6f} "
            f"recon={train_m['recon_sum']:.6f} "
            f"mse={train_m['recon_mse']:.8f} "
            f"kl={train_m['kl']:.6f} | "
            f"val loss={val_m['loss']:.6f} "
            f"recon={val_m['recon_sum']:.6f} "
            f"mse={val_m['recon_mse']:.8f} "
            f"kl={val_m['kl']:.6f}"
        )

        with metrics_file.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [
                    epoch,
                    train_m["loss"],
                    train_m["recon_sum"],
                    train_m["recon_mse"],
                    train_m["kl"],
                    train_m["samples"],
                    val_m["loss"],
                    val_m["recon_sum"],
                    val_m["recon_mse"],
                    val_m["kl"],
                    val_m["samples"],
                    elapsed,
                ]
            )

        # Always keep last.
        save_checkpoint(
            out_dir / "last.pt",
            model,
            optimizer,
            epoch,
            best_val_loss,
            model_cfg,
            args,
            mean_cpu,
            std_cpu,
        )

        # Best validation checkpoint.
        if math.isfinite(val_m["loss"]) and val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            save_checkpoint(
                out_dir / "best.pt",
                model,
                optimizer,
                epoch,
                best_val_loss,
                model_cfg,
                args,
                mean_cpu,
                std_cpu,
            )

            save_latent_snapshot(
                model=model,
                trot_store=trot_store,
                bound_store=bound_store,
                trot_val_ids=trot_val,
                bound_val_ids=bound_val,
                mean=mean,
                std=std,
                device=device,
                out_file=out_dir / "best_val_latents.pt",
                per_gait=args.latent_snapshot_per_gait,
                batch_size=args.batch_size,
            )

            print(f"[BEST] val loss improved to {best_val_loss:.6f}")

        if args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0:
            save_checkpoint(
                out_dir / f"epoch_{epoch:04d}.pt",
                model,
                optimizer,
                epoch,
                best_val_loss,
                model_cfg,
                args,
                mean_cpu,
                std_cpu,
            )

    print("\n" + "=" * 100)
    print("TRAINING COMPLETE")
    print(f"best checkpoint : {out_dir / 'best.pt'}")
    print(f"last checkpoint : {out_dir / 'last.pt'}")
    print(f"metrics         : {metrics_file}")
    print(f"latent snapshot : {out_dir / 'best_val_latents.pt'}")
    print("=" * 100)


if __name__ == "__main__":
    main()
