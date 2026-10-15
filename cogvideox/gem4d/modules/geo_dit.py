"""ref_13 / CN-MIGV — auxiliary Geo-DiT  (+ Task-36 / Task-30 architecture variants).

Predicts the FULL geometry velocity v_geo from RGB-DiT hidden features (cross-attention),
with empty/present input modes. Train-only; dropped at inference (RGB path = ref7).

Token layout matches the RGB-DiT: tokens are frame-major over the (Fl, h/p, w/p) grid, so the
i-th geo token aligns 1:1 with the i-th RGB hidden token, and reshaping [B,N,dim] ->
[B, Fl, S, dim] (S = h/p * w/p) exposes the frame axis for the temporal modules below.

Task-36 adds the Task-30 proposals as env-gated, ZERO-INIT-residual modules (GEO_VARIANT):
  base    : ref13 (no extra module)
  convgru : P1 ConvGRU temporal recurrence over frames
  tattn   : P2 decoupled temporal (cross-frame) self-attention
  iter    : P3 iterative recurrent refinement (re-run the block stack K times)
  bias3d  : P4 geometry/depth-conditioned temporal attention (depth-diff attention bias)
  ssm     : P5 diagonal state-space (S4-lite) temporal scan
All extra modules are added as `tok = tok + g * module(tok)` with g zero-init -> identical to
ref13 at initialisation, then ramp in.
"""
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
    a = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(a), torch.sin(a)], dim=-1)
    if dim % 2: emb = torch.cat([emb, emb[:, :1]*0], dim=-1)
    return emb


class GeoBlock(nn.Module):
    """pre-norm self-attn + cross-attn(to RGB hidden) + MLP, with zero-init residual gates."""
    def __init__(self, dim, heads):
        super().__init__()
        self.n1 = nn.LayerNorm(dim); self.sa = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = nn.LayerNorm(dim); self.ca = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))
        # zero-init gates -> block starts as identity (no perturbation), then ramps in
        self.g1 = nn.Parameter(torch.zeros(1)); self.g2 = nn.Parameter(torch.zeros(1)); self.g3 = nn.Parameter(torch.zeros(1))

    def forward(self, x, temb, ctx):
        h = self.n1(x) + temb[:, None]
        x = x + self.g1 * self.sa(h, h, h, need_weights=False)[0]
        h = self.n2(x)
        x = x + self.g2 * self.ca(h, ctx, ctx, need_weights=False)[0]
        x = x + self.g3 * self.mlp(self.n3(x))
        return x


# ----------------------- Task-30 temporal modules -----------------------
def _to_frames(x, Fl, S):   # [B, Fl*S, D] frame-major -> [B*S, Fl, D]
    B, N, D = x.shape
    return x.reshape(B, Fl, S, D).permute(0, 2, 1, 3).reshape(B * S, Fl, D)

def _from_frames(y, B, Fl, S, D):   # [B*S, Fl, D] -> [B, Fl*S, D]
    return y.reshape(B, S, Fl, D).permute(0, 2, 1, 3).reshape(B, Fl * S, D)


class ConvGRUTime(nn.Module):
    """P1: GRU over the frame axis at each spatial token (shared across positions)."""
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.gru = nn.GRU(dim, dim, batch_first=True)

    def forward(self, x, Fl, S):
        B, N, D = x.shape
        y = _to_frames(self.norm(x), Fl, S)
        y, _ = self.gru(y)
        return _from_frames(y, B, Fl, S, D)


class TemporalAttn(nn.Module):
    """P2 (use_bias=False) / P4 (use_bias=True): cross-frame self-attention, optionally with a
    depth-difference attention bias (geometry-conditioned)."""
    def __init__(self, dim, heads, use_bias=False):
        super().__init__()
        self.h = heads; self.dh = dim // heads; self.use_bias = use_bias
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        if use_bias:
            self.depth = nn.Linear(dim, 1)                 # per-token depth scalar
            self.bscale = nn.Parameter(torch.zeros(1))     # init 0 -> plain attention

    def forward(self, x, Fl, S):
        B, N, D = x.shape
        y = _to_frames(self.norm(x), Fl, S)                # [M, Fl, D], M=B*S
        M = y.shape[0]
        qkv = self.qkv(y).reshape(M, Fl, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                   # [M, h, Fl, dh]
        attn = (q @ k.transpose(-1, -2)) / (self.dh ** 0.5)  # [M, h, Fl, Fl]
        if self.use_bias:
            d = self.depth(y).squeeze(-1)                  # [M, Fl]
            dbias = -(d[:, :, None] - d[:, None, :]).abs() # [M, Fl, Fl]
            attn = attn + self.bscale * dbias[:, None]
        attn = attn.softmax(-1)
        out = (attn @ v).permute(0, 2, 1, 3).reshape(M, Fl, D)
        return _from_frames(self.proj(out), B, Fl, S, D)


class DiagSSM(nn.Module):
    """P5: per-channel diagonal state-space scan over frames (S4-lite)."""
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.a_logit = nn.Parameter(torch.zeros(dim))      # decay a = sigmoid(.) in (0,1)
        self.b = nn.Linear(dim, dim); self.c = nn.Linear(dim, dim)

    def forward(self, x, Fl, S):
        B, N, D = x.shape
        y = _to_frames(self.norm(x), Fl, S)                # [M, Fl, D]
        a = torch.sigmoid(self.a_logit)                    # [D]
        bx = self.b(y)                                     # [M, Fl, D]
        h = torch.zeros(y.shape[0], D, device=y.device, dtype=y.dtype)
        outs = []
        for f in range(Fl):
            h = a * h + bx[:, f]
            outs.append(h)
        h_seq = torch.stack(outs, dim=1)                   # [M, Fl, D]
        return _from_frames(self.c(h_seq), B, Fl, S, D)


class GeoDiT(nn.Module):
    def __init__(self, dim, heads=24, depth=4, in_ch=16, patch=2):
        super().__init__()
        self.dim = dim; self.patch = patch; self.in_ch = in_ch
        pc = in_ch * patch * patch
        self.geo_embed   = nn.Linear(pc, dim)     # present-mode: patch-embed z_t_geo
        # Optional extra input: stop-grad RGB noise estimate eps_rgb_hat = sigma*z_t + alpha*v_rgb.
        # OFF by default (GEO_EPS_HAT=0) — the paper's Geometry DiT is conditioned only on the video
        # backbone's intermediate features. With coupled noise this input lets the Geo-DiT "echo" the
        # noise term of its velocity target instead of reading geometry from the backbone.
        # The released checkpoints (checkpoint-1 / checkpoint-2) were trained with GEO_EPS_HAT=1.
        self.use_eps_hat = os.environ.get("GEO_EPS_HAT", "0") == "1"
        self.eps_embed   = nn.Linear(pc, dim) if self.use_eps_hat else None
        self.null_token  = nn.Parameter(torch.zeros(1, 1, dim))   # empty-mode learned null
        self.present_emb = nn.Parameter(torch.zeros(1, 1, dim))
        self.empty_emb   = nn.Parameter(torch.zeros(1, 1, dim))
        # ref15 (Task-38): masked-geometry mode — learned mask token replaces masked geo tokens
        self.mask_token  = nn.Parameter(torch.zeros(1, 1, dim))
        self.masked_emb  = nn.Parameter(torch.zeros(1, 1, dim))
        self.mask_ratio  = float(os.environ.get("MASK_RATIO", "0.5"))
        self.temb_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([GeoBlock(dim, heads) for _ in range(depth)])
        self.norm_out = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, pc); nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        nn.init.normal_(self.null_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        # ---- Task-36 / Task-30 variant ----
        self.variant = os.environ.get("GEO_VARIANT", "base")
        self.geo_iters = int(os.environ.get("GEO_ITERS", "4"))
        self.tmix = None
        if self.variant == "convgru":
            self.tmix = ConvGRUTime(dim)
        elif self.variant == "tattn":
            self.tmix = TemporalAttn(dim, heads, use_bias=False)
        elif self.variant == "bias3d":
            self.tmix = TemporalAttn(dim, heads, use_bias=True)
        elif self.variant == "ssm":
            self.tmix = DiagSSM(dim)
        # 'iter' and 'base' have no tmix; 'iter' loops the block stack in forward.
        self.tmix_gate = nn.Parameter(torch.zeros(1))  # zero-init residual gate (identity at init)

    def _patchify(self, x):  # [B,Fl,C,h,w] -> [B, Fl*hp*wp, C*p*p]
        B, Fl, C, h, w = x.shape; p = self.patch
        x = x.reshape(B, Fl, C, h//p, p, w//p, p)
        x = x.permute(0,1,3,5,2,4,6).reshape(B, Fl*(h//p)*(w//p), C*p*p)
        return x

    def _unpatchify(self, tok, Fl, h, w):  # [B,N,C*p*p] -> [B,Fl,C,h,w]
        B = tok.shape[0]; p = self.patch; C = self.in_ch
        x = tok.reshape(B, Fl, h//p, w//p, C, p, p).permute(0,1,4,2,5,3,6)
        return x.reshape(B, Fl, C, h, w)

    def forward(self, h_rgb, t, shape, mode, z_t_geo=None, eps_rgb_hat=None):
        """h_rgb: [B,N,dim] RGB hidden (ctx). shape=(Fl,h,w). mode in {'empty','present'}.
           returns v_geo [B,Fl,16,h,w]."""
        wd = self.out.weight.dtype
        B, N, _ = h_rgb.shape; Fl, h, w = shape
        h_rgb = h_rgb.to(wd)
        if mode == 'present':
            assert z_t_geo is not None
            tok = self.geo_embed(self._patchify(z_t_geo).to(wd)) + self.present_emb
        elif mode == 'masked':
            # ref15: patch-embed the noised geometry latent, then mask a fraction of TOKENS with a
            # learned mask token, forcing the model to recover them from the RGB hidden features.
            assert z_t_geo is not None
            tok = self.geo_embed(self._patchify(z_t_geo).to(wd))
            if self.mask_ratio > 0:
                keep = (torch.rand(B, tok.shape[1], 1, device=tok.device) >= self.mask_ratio).to(wd)
                tok = keep * tok + (1.0 - keep) * self.mask_token.to(wd)
            tok = tok + self.masked_emb
        else:
            tok = self.null_token.expand(B, N, -1) + self.empty_emb
        if self.use_eps_hat and eps_rgb_hat is not None:
            tok = tok + self.eps_embed(self._patchify(eps_rgb_hat).to(wd))
        temb = self.temb_mlp(timestep_embedding(t, self.dim).to(wd))
        iters = self.geo_iters if self.variant == "iter" else 1
        for _ in range(iters):
            for blk in self.blocks:
                tok = blk(tok, temb, h_rgb)
        if self.tmix is not None:
            S = (Fl_S := tok.shape[1]) // Fl
            tok = tok + self.tmix_gate * self.tmix(tok, Fl, S).to(tok.dtype)
        v = self.out(self.norm_out(tok))
        return self._unpatchify(v, Fl, h, w)
