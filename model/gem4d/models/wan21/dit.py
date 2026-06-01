"""Wan2.1-I2V-14B diffusion transformer (original Wan2.1 parameter names).

Adapted from Wan2.1 (https://github.com/Wan-Video/Wan2.1, Apache-2.0) and DiffSynth-Studio
(https://github.com/modelscope/DiffSynth-Studio, Apache-2.0); see NOTICE.md in this folder.

Input: noisy latent [B, 16, T, h, w]. The image condition is concatenated on the channel axis
(4 mask channels + 16 condition-latent channels = 36 input channels): the mask is 1 on the
latent frames that carry a clean condition latent and 0 elsewhere. Cross-attention context =
257 CLIP image tokens followed by the text tokens. Video self-attention is bidirectional.
"""
import json
import math
import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    ctx_mask: Optional[torch.Tensor] = None,
):
    q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
    k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
    v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
    if ctx_mask is not None:
        if ctx_mask.dim() == 2:
            ctx_mask = ctx_mask.unsqueeze(0).unsqueeze(0)
        elif ctx_mask.dim() == 3:
            ctx_mask = ctx_mask.unsqueeze(1)
        ctx_mask = ctx_mask.to(device=q.device, dtype=torch.bool)
    x = F.scaled_dot_product_attention(q, k, v, attn_mask=ctx_mask)
    return rearrange(x, "b n s d -> b s (n d)", n=num_heads)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale) + shift


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor):
    assert dim % 2 == 0
    half = dim // 2
    dtype = position.dtype if position.is_floating_point() else torch.float32
    position = position.type(torch.float64)
    sinusoid = torch.outer(
        position,
        torch.pow(
            10000, -torch.arange(half, dtype=torch.float64, device=position.device).div(half)
        ),
    )
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1).to(dtype)


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).double()[: (dim // 2)] / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    return (
        precompute_freqs_cis(dim - 4 * (dim // 6), end, theta),
        precompute_freqs_cis(2 * (dim // 6), end, theta),
        precompute_freqs_cis(2 * (dim // 6), end, theta),
    )


def rope_apply(x: torch.Tensor, freqs: torch.Tensor, num_heads: int):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(*x.shape[:-1], -1, 2))
    freqs = freqs.to(device=x.device)
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


class WanRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x.to(dtype) * self.weight


class WanLayerNorm(nn.LayerNorm):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = False):
        super().__init__(dim, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x: torch.Tensor):
        weight = self.weight.float() if self.weight is not None else None
        bias = self.bias.float() if self.bias is not None else None
        return F.layer_norm(x.float(), self.normalized_shape, weight, bias, self.eps).type_as(x)


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, attn_head_dim: int, num_heads: int, eps: float = 1e-6,
                 qk_norm: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.attn_hidden_dim = num_heads * attn_head_dim
        self.q = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.k = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.v = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.o = nn.Linear(self.attn_hidden_dim, hidden_dim)
        self.norm_q = WanRMSNorm(self.attn_hidden_dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(self.attn_hidden_dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, freqs):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        return self.o(flash_attention(q, k, v, self.num_heads))


class I2VCrossAttention(nn.Module):
    """Cross-attention to [257 CLIP image tokens | text tokens], image and text attended separately."""

    def __init__(self, hidden_dim: int, attn_head_dim: int, num_heads: int, eps: float = 1e-6,
                 qk_norm: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.attn_hidden_dim = num_heads * attn_head_dim
        self.q = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.k = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.v = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.o = nn.Linear(self.attn_hidden_dim, hidden_dim)
        self.norm_q = WanRMSNorm(self.attn_hidden_dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(self.attn_hidden_dim, eps=eps) if qk_norm else nn.Identity()
        self.k_img = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.v_img = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.norm_k_img = WanRMSNorm(self.attn_hidden_dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, ctx_mask: Optional[torch.Tensor] = None):
        if ctx.shape[1] < 257:
            raise ValueError(f"Wan2.1 i2v context must include 257 image tokens, got {ctx.shape[1]}.")
        ctx_img = ctx[:, :257]
        ctx_txt = ctx[:, 257:]
        text_mask = None if ctx_mask is None else ctx_mask[..., 257:]

        q = self.norm_q(self.q(x))
        k_img = self.norm_k_img(self.k_img(ctx_img))
        v_img = self.v_img(ctx_img)
        img_x = flash_attention(q, k_img, v_img, self.num_heads)

        k = self.norm_k(self.k(ctx_txt))
        v = self.v(ctx_txt)
        txt_x = flash_attention(q, k, v, self.num_heads, ctx_mask=text_mask)
        return self.o(txt_x + img_x)


class GateModule(nn.Module):
    def forward(self, x, gate, residual):
        return x + gate * residual


class DiTBlock(nn.Module):
    def __init__(self, hidden_dim: int, attn_head_dim: int, num_heads: int, ffn_dim: int,
                 eps: float = 1e-6, qk_norm: bool = True, cross_attn_norm: bool = False):
        super().__init__()
        self.self_attn = SelfAttention(hidden_dim, attn_head_dim, num_heads, eps, qk_norm)
        self.cross_attn = I2VCrossAttention(hidden_dim, attn_head_dim, num_heads, eps, qk_norm)
        self.norm1 = WanLayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm2 = WanLayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm3 = WanLayerNorm(hidden_dim, eps=eps, elementwise_affine=bool(cross_attn_norm))
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, hidden_dim) / hidden_dim**0.5)
        self.gate = GateModule()

    def forward(self, x, context, t_mod, freqs, context_mask=None):
        if context_mask is not None and context_mask.dim() == 3:
            context_mask = context_mask.unsqueeze(1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=1)
        y = self.self_attn(modulate(self.norm1(x), shift_msa, scale_msa), freqs)
        x = self.gate(x, gate_msa, y)
        x = x + self.cross_attn(self.norm3(x), context, ctx_mask=context_mask)
        y = self.ffn(modulate(self.norm2(x), shift_mlp, scale_mlp))
        x = self.gate(x, gate_mlp, y)
        return x


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float = 1e-6):
        super().__init__()
        self.norm = WanLayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t):
        shift, scale = (self.modulation.to(dtype=t.dtype, device=t.device) + t.unsqueeze(1)).chunk(2, dim=1)
        return self.head(self.norm(x) * (1 + scale) + shift)


class MLPProj(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, image_embeds):
        return self.proj(image_embeds)


class WanDiT(nn.Module):
    """Wan2.1 I2V DiT. Constructor arguments are the keys of the checkpoint's config.json."""

    CONFIG_KEYS = ("model_type", "patch_size", "text_len", "in_dim", "dim", "ffn_dim", "freq_dim",
                   "text_dim", "out_dim", "num_heads", "num_layers", "qk_norm", "cross_attn_norm", "eps")

    def __init__(
        self,
        model_type: str = "i2v",
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        text_len: int = 512,
        in_dim: int = 36,
        dim: int = 5120,
        ffn_dim: int = 13824,
        freq_dim: int = 256,
        text_dim: int = 4096,
        out_dim: int = 16,
        num_heads: int = 40,
        num_layers: int = 40,
        qk_norm: bool = True,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        **_: Any,
    ):
        super().__init__()
        if model_type != "i2v":
            raise ValueError(f"only the Wan2.1 i2v DiT is supported, got model_type={model_type!r}")
        if dim % num_heads != 0:
            raise ValueError("`dim` must be divisible by `num_heads`.")
        self.config = dict(model_type=model_type, patch_size=list(patch_size), text_len=int(text_len),
                           in_dim=int(in_dim), dim=int(dim), ffn_dim=int(ffn_dim), freq_dim=int(freq_dim),
                           text_dim=int(text_dim), out_dim=int(out_dim), num_heads=int(num_heads),
                           num_layers=int(num_layers), qk_norm=bool(qk_norm),
                           cross_attn_norm=bool(cross_attn_norm), eps=float(eps))
        self.patch_size = tuple(patch_size)
        self.in_dim = int(in_dim)
        self.hidden_dim = int(dim)
        self.freq_dim = int(freq_dim)
        self.out_dim = int(out_dim)
        self.num_heads = int(num_heads)
        self.attn_head_dim = self.hidden_dim // self.num_heads

        self.patch_embedding = nn.Conv3d(self.in_dim, self.hidden_dim, kernel_size=self.patch_size,
                                         stride=self.patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, self.hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(self.freq_dim, self.hidden_dim), nn.SiLU(), nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(self.hidden_dim, self.hidden_dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock(self.hidden_dim, self.attn_head_dim, self.num_heads, int(ffn_dim), eps,
                     qk_norm=qk_norm, cross_attn_norm=cross_attn_norm)
            for _ in range(int(num_layers))
        ])
        self.head = Head(self.hidden_dim, self.out_dim, self.patch_size, eps)
        self.img_emb = MLPProj(1280, self.hidden_dim)
        self.init_freqs()
        self.gradient_checkpointing = False

    def init_freqs(self):
        """RoPE tables (plain CPU attributes, not parameters; call again after building on 'meta')."""
        with torch.device("cpu"):
            self.freqs = precompute_freqs_cis_3d(self.attn_head_dim)

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

    def unpatchify(self, x: torch.Tensor, grid_size):
        f, h, w = grid_size
        return rearrange(
            x,
            "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
            f=f, h=h, w=w, x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2],
        )

    def build_condition(self, x: torch.Tensor, condition_latents: torch.Tensor):
        """[mask (4) | condition latent (16)] for a noisy latent x [B, 16, T, h, w]. The condition
        latents fill the first latent frames (1 for a single image; 0 = no image condition)."""
        b, _, lat_t, lat_h, lat_w = x.shape
        if condition_latents.ndim != 5 or condition_latents.shape[0] != b or \
                condition_latents.shape[1] != self.out_dim or condition_latents.shape[3:] != (lat_h, lat_w):
            raise ValueError(f"`condition_latents` {tuple(condition_latents.shape)} does not match "
                             f"[B={b}, C={self.out_dim}, T<={lat_t}, H={lat_h}, W={lat_w}]")
        n = condition_latents.shape[2]
        if n > lat_t:
            raise ValueError(f"`condition_latents` cannot be longer than the noisy latent ({n} > {lat_t}).")
        mask = torch.zeros((b, 4, lat_t, lat_h, lat_w), dtype=x.dtype, device=x.device)
        mask[:, :, :n] = 1
        y = torch.zeros((b, self.out_dim, lat_t, lat_h, lat_w), dtype=x.dtype, device=x.device)
        y[:, :, :n] = condition_latents.to(device=x.device, dtype=x.dtype)
        return torch.cat([mask, y], dim=1)

    def pre_dit(self, x, timestep, context, context_mask=None, clip_fea=None, condition_latents=None) -> Dict[str, Any]:
        if x.ndim != 5:
            raise ValueError(f"`x` must be 5D [B,C,T,H,W], got {tuple(x.shape)}")
        if x.shape[1] == self.out_dim:
            if condition_latents is None:
                condition_latents = x[:, :, :0]
            x = torch.cat([x, self.build_condition(x, condition_latents)], dim=1)
        if x.shape[1] != self.in_dim:
            raise ValueError(f"DiT expected {self.in_dim} channels after conditioning, got {x.shape[1]}.")
        if context.ndim != 3:
            raise ValueError(f"`context` must be 3D [B,L,D], got {tuple(context.shape)}")
        batch_size = x.shape[0]
        if timestep.ndim != 1 or timestep.shape[0] not in (1, batch_size):
            raise ValueError(f"`timestep` must have shape [B] or [1], got {tuple(timestep.shape)}")
        if timestep.shape[0] == 1 and batch_size > 1:
            timestep = timestep.expand(batch_size)
        if context_mask is None:
            context_mask = torch.ones((batch_size, context.shape[1]), dtype=torch.bool, device=context.device)
        if clip_fea is None:
            clip_fea = torch.zeros((batch_size, 257, 1280), dtype=context.dtype, device=context.device)
        if clip_fea.ndim != 3 or clip_fea.shape[1:] != (257, 1280):
            raise ValueError(f"`clip_fea` must be [B,257,1280], got {tuple(clip_fea.shape)}")

        x = self.patch_embedding(x)
        f, h, w = x.shape[2:]
        seq_len = f * h * w

        time_weight = self.time_embedding[0].weight
        time_input = sinusoidal_embedding_1d(self.freq_dim, timestep).to(device=time_weight.device,
                                                                          dtype=time_weight.dtype)
        t = self.time_embedding(time_input)
        t_mod = self.time_projection(t).unflatten(1, (6, self.hidden_dim))

        text_context = self.text_embedding(context)
        image_context = self.img_emb(clip_fea.to(device=text_context.device, dtype=text_context.dtype))
        context_emb = torch.cat([image_context, text_context], dim=1)
        image_mask = torch.ones((batch_size, image_context.shape[1]), dtype=torch.bool, device=context_mask.device)
        combined_mask = torch.cat([image_mask, context_mask.to(dtype=torch.bool)], dim=1)
        context_attn_mask = combined_mask.unsqueeze(1).expand(-1, seq_len, -1)

        x_tokens = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
        freqs = (
            torch.cat(
                [
                    self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                    self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            )
            .reshape(seq_len, 1, -1)
            .to(x_tokens.device)
        )
        return {"tokens": x_tokens, "freqs": freqs, "t": t, "t_mod": t_mod, "context": context_emb,
                "context_mask": context_attn_mask, "grid_size": (f, h, w)}

    def forward(self, x, timestep, context, context_mask=None, clip_fea=None, condition_latents=None):
        """x [B, 16 or 36, T, h, w] -> velocity [B, 16, T, h, w]."""
        pre = self.pre_dit(x, timestep, context, context_mask, clip_fea, condition_latents)
        tokens = pre["tokens"]
        for block in self.blocks:
            if self.gradient_checkpointing and torch.is_grad_enabled():
                tokens = checkpoint(block, tokens, pre["context"], pre["t_mod"], pre["freqs"],
                                    context_mask=pre["context_mask"], use_reentrant=False)
            else:
                tokens = block(tokens, pre["context"], pre["t_mod"], pre["freqs"], context_mask=pre["context_mask"])
        return self.unpatchify(self.head(tokens, pre["t"]), pre["grid_size"])

    # ------------------------------------------------------------------ I/O
    def save_pretrained(self, save_directory: str, safe_serialization: bool = True, max_shard_size: str = "5GB"):
        """Write config.json + diffusion_pytorch_model[-0000k-of-0000n].safetensors (+ index), the
        layout of the original Wan2.1 checkpoints, so the result loads like the stock model."""
        save_wan_dit(self.state_dict(), self.config, save_directory, max_shard_size=max_shard_size)


def _parse_size(size) -> int:
    if isinstance(size, int):
        return size
    units = {"KB": 10**3, "MB": 10**6, "GB": 10**9}
    s = str(size).upper().strip()
    for u, m in units.items():
        if s.endswith(u):
            return int(float(s[: -len(u)]) * m)
    return int(s)


def save_wan_dit(state_dict: Dict[str, torch.Tensor], config: Dict[str, Any], save_directory: str,
                 max_shard_size="5GB", extra_config: Optional[Dict[str, Any]] = None):
    from safetensors.torch import save_file

    os.makedirs(save_directory, exist_ok=True)
    limit = _parse_size(max_shard_size)
    shards, cur, cur_size = [], {}, 0
    for k, v in state_dict.items():
        v = v.detach().cpu().contiguous()
        n = v.numel() * v.element_size()
        if cur and cur_size + n > limit:
            shards.append(cur)
            cur, cur_size = {}, 0
        cur[k] = v
        cur_size += n
    if cur:
        shards.append(cur)
    names = (["diffusion_pytorch_model.safetensors"] if len(shards) == 1 else
             [f"diffusion_pytorch_model-{i + 1:05d}-of-{len(shards):05d}.safetensors" for i in range(len(shards))])
    weight_map = {}
    for name, shard in zip(names, shards):
        save_file(shard, os.path.join(save_directory, name), metadata={"format": "pt"})
        weight_map.update({k: name for k in shard})
    if len(shards) > 1:
        total = sum(v.numel() * v.element_size() for s in shards for v in s.values())
        with open(os.path.join(save_directory, "diffusion_pytorch_model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f, indent=2)
    cfg = {"_class_name": "WanModel", **config, **(extra_config or {})}
    with open(os.path.join(save_directory, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
