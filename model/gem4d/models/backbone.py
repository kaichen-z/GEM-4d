"""Common interface of the video backbones used by GEM-4D.

A backbone owns the frozen parts (VAE, text encoder, noise schedule) and exposes the trainable
video DiT as `self.transformer`. All latents use the layout [B, F, C, h, w] (frame-major), which
is also the layout of the geometry latent and of the Geometry DiT.

One training step, written with this interface:

    ctx   = bb.encode_text(prompts)
    x0    = bb.encode_video(video)                  # clean video latent
    cond  = bb.encode_condition(video, drop)        # first-frame conditioning
    t     = bb.sample_timesteps(B)
    x_t   = bb.add_noise(x0, eps, t)
    out   = bb.forward(transformer, x_t, t, cond, ctx)
    L_vid = bb.video_loss(out, x0, eps, x_t, t, cond)
    g_t   = bb.add_noise(g0, eps, t)                # geometry shares eps (coupled noise)
    v*    = bb.velocity_target(g0, eps, t)
"""
from dataclasses import dataclass
from typing import List

import torch


@dataclass
class Condition:
    latents: torch.Tensor          # backbone-specific conditioning latent
    active: bool                   # False if the image condition was dropped for this step


class VideoBackbone:
    name: str = "base"
    latent_channels: int
    spatial_patch: int
    hidden_dim: int
    num_heads: int

    # --- setup ---
    def to(self, device, dtype):
        raise NotImplementedError

    @property
    def blocks(self) -> List[torch.nn.Module]:
        raise NotImplementedError

    @staticmethod
    def block_output(out):
        """Token features from a transformer block's output (the hook target)."""
        return out[0] if isinstance(out, (tuple, list)) else out

    # --- encoding (no grad) ---
    def encode_text(self, prompts: List[str]) -> torch.Tensor:
        raise NotImplementedError

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """video [B, F, 3, H, W] in [-1, 1] -> latent [B, F', C, h, w]."""
        raise NotImplementedError

    def encode_condition(self, video: torch.Tensor, x0: torch.Tensor, drop: bool) -> Condition:
        raise NotImplementedError

    # --- diffusion ---
    def sample_timesteps(self, batch_size: int, device) -> torch.Tensor:
        raise NotImplementedError

    def fixed_timesteps(self, frac: float, batch_size: int, device) -> torch.Tensor:
        """Deterministic timestep at fraction `frac` of the schedule (low-variance validation loss)."""
        raise NotImplementedError

    def add_noise(self, x0, eps, t):
        raise NotImplementedError

    def velocity_target(self, x0, eps, t):
        """Target of the geometry flow (same parametrisation as the video branch)."""
        raise NotImplementedError

    def forward(self, transformer, x_t, t, cond: Condition, ctx):
        raise NotImplementedError

    def video_loss(self, out, x0, eps, x_t, t, cond: Condition) -> torch.Tensor:
        raise NotImplementedError

    def eps_hat(self, out, x_t, t):
        """Stop-grad RGB noise estimate (only for use_eps_hat, i.e. reproducing old checkpoints)."""
        raise NotImplementedError

    def geo_timestep(self, t):
        """Scalar timestep fed to the Geometry DiT, in [0, 1000]."""
        return t.float()
