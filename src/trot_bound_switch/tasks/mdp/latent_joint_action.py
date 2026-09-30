"""Latent action term backed by a frozen gait-specific action decoder."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass

logger = logging.getLogger(__name__)


class ActionDecoder(nn.Module):
    """Runtime-only 16D latent to 16D raw PPO action decoder."""

    def __init__(self, latent_dim: int = 16, hidden_dim: int = 64, action_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class LatentJointPositionAction(JointPositionAction):
    """Joint position action term whose policy input is a 16D latent action."""

    cfg: LatentJointPositionActionCfg

    def __init__(self, cfg: LatentJointPositionActionCfg, env):
        super().__init__(cfg, env)

        self._decoder = self._load_decoder(cfg.decoder_checkpoint)
        self._clipped_latent = torch.zeros_like(self._raw_actions)
        self._decoded_action = torch.zeros_like(self._raw_actions)
        self._processed_joint_target = self._processed_actions
        self._latent_clip_fraction = torch.tensor(0.0, device=self.device)
        self._process_count = 0
        resolved_joint_ids = list(range(self._num_joints)) if isinstance(self._joint_ids, slice) else self._joint_ids

        logger.warning(
            "LatentJointPositionAction resolved joint names: %s, joint IDs: %s, decoder output dimension: %d",
            self._joint_names,
            resolved_joint_ids,
            self.action_dim,
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
        return self._processed_joint_target

    @property
    def latent_clip_fraction(self) -> torch.Tensor:
        return self._latent_clip_fraction

    def process_actions(self, actions: torch.Tensor):
        actions = actions.to(self.device)
        if actions.shape != self._raw_actions.shape:
            raise RuntimeError(f"Expected latent action shape {tuple(self._raw_actions.shape)}, got {tuple(actions.shape)}")
        if not torch.isfinite(actions).all():
            self._raise_non_finite("latent", actions)

        self._raw_actions[:] = actions

        if self.cfg.latent_clip is None:
            self._clipped_latent[:] = actions
            self._latent_clip_fraction.fill_(0.0)
        else:
            clip = float(self.cfg.latent_clip)
            self._clipped_latent[:] = torch.clamp(actions, -clip, clip)
            clipped = torch.count_nonzero(self._clipped_latent != actions)
            self._latent_clip_fraction.copy_(clipped.to(dtype=self._latent_clip_fraction.dtype) / actions.numel())

        with torch.inference_mode():
            decoded_action = self._decoder(self._clipped_latent)

        if decoded_action.shape != self._decoded_action.shape:
            raise RuntimeError(
                f"Decoder output shape mismatch: expected {tuple(self._decoded_action.shape)}, "
                f"got {tuple(decoded_action.shape)}"
            )
        if not torch.isfinite(decoded_action).all():
            self._raise_non_finite("decoded action", decoded_action)

        self._decoded_action[:] = decoded_action
        processed_actions = self._decoded_action * self._scale + self._offset
        if self.cfg.clip is not None:
            processed_actions = torch.clamp(processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1])
        if not torch.isfinite(processed_actions).all():
            self._raise_non_finite("processed joint target", processed_actions)

        self._processed_actions[:] = processed_actions
        self._processed_joint_target = self._processed_actions

        self._process_count += 1
        if self.cfg.debug_interval > 0 and self._process_count % self.cfg.debug_interval == 0:
            self._print_debug_stats()

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        self._clipped_latent[env_ids] = 0.0
        self._decoded_action[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0

    def _load_decoder(self, checkpoint_path: str) -> ActionDecoder:
        if not checkpoint_path:
            raise ValueError("LatentJointPositionActionCfg.decoder_checkpoint must be set.")

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        gait = str(ckpt.get("gait", "")).lower()
        if gait != self.cfg.expected_gait.lower():
            raise RuntimeError(f"Decoder checkpoint gait mismatch: expected {self.cfg.expected_gait!r}, got {gait!r}")

        decoder_config = ckpt.get("decoder_config", {})
        latent_dim = int(decoder_config.get("latent_dim", -1))
        hidden_dim = int(decoder_config.get("hidden_dim", -1))
        action_dim = int(decoder_config.get("action_dim", -1))
        if (latent_dim, hidden_dim, action_dim) != (16, 64, 16):
            raise RuntimeError(
                "Unexpected decoder_config: "
                f"latent_dim={latent_dim}, hidden_dim={hidden_dim}, action_dim={action_dim}"
            )
        if action_dim != self.action_dim:
            raise RuntimeError(f"Decoder action_dim={action_dim} does not match resolved action_dim={self.action_dim}")

        decoder = ActionDecoder(latent_dim=latent_dim, hidden_dim=hidden_dim, action_dim=action_dim)
        decoder.load_state_dict(ckpt["decoder_state_dict"], strict=True)
        decoder.to(self.device)
        decoder.eval()
        for param in decoder.parameters():
            param.requires_grad_(False)

        logger.info("Loaded frozen %s action decoder from %s", gait, checkpoint_path)
        return decoder

    def _print_debug_stats(self) -> None:
        raw_latent = self._raw_actions
        clipped_latent = self._clipped_latent
        decoded = self._decoded_action
        q_des = self._processed_actions
        logger.warning(
            "LatentJointPositionAction stats step=%d "
            "raw_latent[min=%.4f max=%.4f mean=%.4f std=%.4f] "
            "clipped_latent[min=%.4f max=%.4f mean=%.4f std=%.4f] "
            "clip_fraction=%.6f "
            "decoded_action[min=%.4f max=%.4f mean=%.4f std=%.4f] "
            "processed_q_des[min=%.4f max=%.4f mean=%.4f std=%.4f]",
            self._process_count,
            raw_latent.min().item(),
            raw_latent.max().item(),
            raw_latent.mean().item(),
            raw_latent.std(unbiased=False).item(),
            clipped_latent.min().item(),
            clipped_latent.max().item(),
            clipped_latent.mean().item(),
            clipped_latent.std(unbiased=False).item(),
            self._latent_clip_fraction.item(),
            decoded.min().item(),
            decoded.max().item(),
            decoded.mean().item(),
            decoded.std(unbiased=False).item(),
            q_des.min().item(),
            q_des.max().item(),
            q_des.mean().item(),
            q_des.std(unbiased=False).item(),
        )

    @staticmethod
    def _raise_non_finite(name: str, value: torch.Tensor) -> None:
        finite = torch.isfinite(value)
        bad_count = torch.count_nonzero(~finite).item()
        finite_value = value[finite]
        if finite_value.numel() > 0:
            stats = f"finite range=[{finite_value.min().item():.6f}, {finite_value.max().item():.6f}]"
        else:
            stats = "no finite values"
        raise RuntimeError(f"Non-finite {name}: bad_count={bad_count}, {stats}")


@configclass
class LatentJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for frozen-decoder latent joint position actions."""

    class_type: type = LatentJointPositionAction

    decoder_checkpoint: str = ""
    """Path to the frozen gait action decoder checkpoint."""

    expected_gait: str = "trot"
    """Expected gait name stored in the decoder checkpoint."""

    latent_clip: float | None = 3.0
    """Symmetric latent clamp before decoding. Set to None to disable."""

    debug_interval: int = 0
    """Print latent/decoded/q_des stats every N process calls. Disabled when <= 0."""
