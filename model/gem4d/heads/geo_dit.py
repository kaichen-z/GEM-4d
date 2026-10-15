"""Geometry DiT — the training-only geometry branch of GEM-4D.

Given the noised geometry latent g_t (with a fraction of its tokens replaced by a learned mask
token, the mask mechanism M in the paper) and the video backbone's intermediate features m_t,
predict the geometry flow / velocity:

    v_geo = GeoDiT(M(g_t), t, m_t)

m_t enters only through cross-attention and is the Geometry DiT's only scene-level input, so
minimising the geometry loss forces the video backbone to encode geometry in m_t. The branch is
discarded at inference.

Tokens are laid out frame-major over the latent grid (F, h/p, w/p) so the i-th geometry token is
aligned with the i-th video token of the backbone.
"""
import math

import torch
import torch.nn as nn


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
    a = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(a), torch.sin(a)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, emb[:, :1] * 0], dim=-1)
    return emb


class GeoBlock(nn.Module):
    """Pre-norm self-attention + cross-attention to the video features + MLP, zero-init gates."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.sa = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.ca = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        # zero-init gates: every block starts as the identity
        self.g1 = nn.Parameter(torch.zeros(1))
        self.g2 = nn.Parameter(torch.zeros(1))
        self.g3 = nn.Parameter(torch.zeros(1))

    def forward(self, x, temb, ctx):
        h = self.n1(x) + temb[:, None]
        x = x + self.g1 * self.sa(h, h, h, need_weights=False)[0]
        h = self.n2(x)
        x = x + self.g2 * self.ca(h, ctx, ctx, need_weights=False)[0]
        x = x + self.g3 * self.mlp(self.n3(x))
        return x


class GeoDiT(nn.Module):
    """
    Args:
        dim:        hidden size; must equal the video backbone's token width (cross-attention ctx).
        heads:      attention heads.
        depth:      number of GeoBlocks.
        in_ch:      geometry-latent channels (= the backbone's latent channels).
        patch:      spatial patch size (= the backbone's spatial patch size).
        mask_ratio: fraction of geometry tokens replaced by the mask token during training.
        use_eps_hat: also add a stop-grad RGB noise estimate as input. Off in the paper's method;
                    only needed to reproduce the released CogVideoX checkpoints (see readme).
    """

    def __init__(self, dim: int, heads: int, depth: int = 4, in_ch: int = 16, patch: int = 2,
                 mask_ratio: float = 0.5, use_eps_hat: bool = False):
        super().__init__()
        self.dim, self.patch, self.in_ch = dim, patch, in_ch
        self.mask_ratio = float(mask_ratio)
        pc = in_ch * patch * patch
        self.geo_embed = nn.Linear(pc, dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.masked_emb = nn.Parameter(torch.zeros(1, 1, dim))
        self.use_eps_hat = bool(use_eps_hat)
        self.eps_embed = nn.Linear(pc, dim) if self.use_eps_hat else None
        self.temb_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([GeoBlock(dim, heads) for _ in range(depth)])
        self.norm_out = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, pc)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        nn.init.normal_(self.mask_token, std=0.02)

    def _patchify(self, x):  # [B, F, C, h, w] -> [B, F*(h/p)*(w/p), C*p*p]
        B, F, C, h, w = x.shape
        p = self.patch
        x = x.reshape(B, F, C, h // p, p, w // p, p)
        return x.permute(0, 1, 3, 5, 2, 4, 6).reshape(B, F * (h // p) * (w // p), C * p * p)

    def _unpatchify(self, tok, F, h, w):  # [B, N, C*p*p] -> [B, F, C, h, w]
        B, p, C = tok.shape[0], self.patch, self.in_ch
        x = tok.reshape(B, F, h // p, w // p, C, p, p).permute(0, 1, 4, 2, 5, 3, 6)
        return x.reshape(B, F, C, h, w)

    def forward(self, video_feats: torch.Tensor, t: torch.Tensor, g_t: torch.Tensor,
                eps_hat: torch.Tensor = None) -> torch.Tensor:
        """
        video_feats: [B, N, dim]   backbone intermediate features m_t (NOT detached)
        t:           [B]           diffusion timestep in [0, 1000]
        g_t:         [B, F, C, h, w] noised geometry latent
        eps_hat:     optional [B, F, C, h, w] stop-grad RGB noise estimate (use_eps_hat only)
        returns      [B, F, C, h, w] predicted geometry velocity
        """
        wd = self.out.weight.dtype
        B, F, _, h, w = g_t.shape
        tok = self.geo_embed(self._patchify(g_t).to(wd))
        if self.mask_ratio > 0:  # masked in validation too, so val/geo_loss measures the same objective
            keep = (torch.rand(B, tok.shape[1], 1, device=tok.device) >= self.mask_ratio).to(wd)
            tok = keep * tok + (1.0 - keep) * self.mask_token.to(wd)
        tok = tok + self.masked_emb
        if self.use_eps_hat and eps_hat is not None:
            tok = tok + self.eps_embed(self._patchify(eps_hat).to(wd))
        temb = self.temb_mlp(timestep_embedding(t, self.dim).to(wd))
        ctx = video_feats.to(wd)
        for blk in self.blocks:
            tok = blk(tok, temb, ctx)
        return self._unpatchify(self.out(self.norm_out(tok)), F, h, w)
