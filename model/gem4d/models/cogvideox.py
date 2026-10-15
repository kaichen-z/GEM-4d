"""CogVideoX-5b-I2V backbone (diffusers).

Video branch (unchanged from CogVideoX fine-tuning):
    x_t   = sqrt(a_t) x0 + sqrt(1 - a_t) eps                 (DDPM, v-prediction)
    x0_hat = sqrt(a_t) x_t - sqrt(1 - a_t) v_hat
    L_vid = mean_B( mean( (x0_hat - x0)^2 / (1 - a_t) ) )
Geometry flow target (same parametrisation):  v* = sqrt(a_t) eps - sqrt(1 - a_t) g0
"""
import glob
import json
import os
from typing import List, Optional

import torch
from diffusers import AutoencoderKLCogVideoX, CogVideoXDPMScheduler, CogVideoXTransformer3DModel
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from safetensors.torch import load_file
from transformers import AutoTokenizer, T5EncoderModel

from .backbone import Condition, VideoBackbone


def _resize_crop_region(src, tgt_width, tgt_height):
    h, w = src
    if h / w > tgt_height / tgt_width:
        rh, rw = tgt_height, int(round(tgt_height / h * w))
    else:
        rw, rh = tgt_width, int(round(tgt_width / w * h))
    top, left = int(round((tgt_height - rh) / 2.0)), int(round((tgt_width - rw) / 2.0))
    return (top, left), (top + rh, left + rw)


def load_transformer_weights(path: str) -> dict:
    """All tensors of a saved transformer dir (config.json + *.safetensors, directly or in transformer/)."""
    if not glob.glob(os.path.join(path, "*.safetensors")) and os.path.isdir(os.path.join(path, "transformer")):
        path = os.path.join(path, "transformer")
    sd = {}
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        sd.update(load_file(f))
    if not sd:
        raise FileNotFoundError(f"no *.safetensors under {path}")
    return sd


class CogVideoXBackbone(VideoBackbone):
    name = "cogvideox"

    def __init__(self, base_model: str, init_weights: Optional[str] = None, max_text_len: int = 226,
                 image_noise_mean: float = -3.0, image_noise_std: float = 0.5):
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, subfolder="tokenizer")
        self.text_encoder = T5EncoderModel.from_pretrained(base_model, subfolder="text_encoder")
        self.vae = AutoencoderKLCogVideoX.from_pretrained(base_model, subfolder="vae")
        self.scheduler = CogVideoXDPMScheduler.from_pretrained(base_model, subfolder="scheduler")
        self.transformer = CogVideoXTransformer3DModel.from_pretrained(
            base_model, subfolder="transformer", torch_dtype=torch.bfloat16)
        # Train/test at 480x640: drop the 480x720 learned positional table (RoPE remains).
        pe = self.transformer.patch_embed
        if getattr(pe, "use_learned_positional_embeddings", False):
            if hasattr(pe, "pos_embedding"):
                del pe.pos_embedding
            pe.use_learned_positional_embeddings = False
            self.transformer.register_to_config(use_learned_positional_embeddings=False)
        self.extra_init_tensors = {}
        if init_weights:
            self.extra_init_tensors = self.load_weights(init_weights)

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.vae.enable_slicing()
        self.vae.enable_tiling()
        cfg = self.transformer.config
        self.max_text_len = max_text_len
        self.latent_channels = self.vae.config.latent_channels           # 16
        self.spatial_patch = cfg.patch_size                                # 2
        self.num_heads = cfg.num_attention_heads                           # 48
        self.hidden_dim = cfg.num_attention_heads * cfg.attention_head_dim  # 3072
        self.vae_scale_spatial = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.scaling_factor = self.vae.config.scaling_factor
        self.rope_base = (cfg.sample_height * self.vae_scale_spatial, cfg.sample_width * self.vae_scale_spatial)
        self.image_noise = (image_noise_mean, image_noise_std)
        self.device, self.dtype = torch.device("cpu"), torch.float32

    def load_weights(self, path: str) -> dict:
        """Load a saved / released transformer into the stock CogVideoX transformer.

        Keys from older GEM-4D checkpoints that the stock model does not have (Geo-DiT, legacy
        depth/normal heads, the geometry projector) are returned so the caller can reuse what it
        needs (the projector)."""
        sd = load_transformer_weights(path)
        own = self.transformer.state_dict()
        load = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
        missing = [k for k in own if k not in load]
        if missing:
            raise RuntimeError(f"{path}: {len(missing)} transformer tensors missing, e.g. {missing[:5]}")
        self.transformer.load_state_dict(load, strict=True)
        return {k: v for k, v in sd.items() if k not in own}

    def to(self, device, dtype):
        self.device, self.dtype = device, dtype
        self.text_encoder.to(device, dtype=dtype)
        self.vae.to(device, dtype=dtype)
        self.alphas_cumprod = self.scheduler.alphas_cumprod.to(device, dtype=torch.float32)
        return self

    @property
    def blocks(self):
        return self.transformer.transformer_blocks

    # ---------------- encoding ----------------
    @torch.no_grad()
    def encode_text(self, prompts: List[str]) -> torch.Tensor:
        ids = self.tokenizer(prompts, padding="max_length", max_length=self.max_text_len, truncation=True,
                             add_special_tokens=True, return_tensors="pt").input_ids
        return self.text_encoder(ids.to(self.device))[0].to(self.dtype)

    @torch.no_grad()
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        z = self.vae.encode(video.permute(0, 2, 1, 3, 4).to(self.dtype)).latent_dist.sample()
        return (z * self.scaling_factor).permute(0, 2, 1, 3, 4).contiguous()      # [B, F', 16, h, w]

    @torch.no_grad()
    def encode_condition(self, video, x0, drop: bool) -> Condition:
        B, Fl = x0.shape[:2]
        img = video[:, :1].permute(0, 2, 1, 3, 4).to(self.dtype)                  # [B, 3, 1, H, W]
        sigma = torch.exp(torch.normal(self.image_noise[0], self.image_noise[1], (B,), device=self.device))
        img = img + torch.randn_like(img) * sigma.to(img.dtype)[:, None, None, None, None]
        z = self.vae.encode(img).latent_dist.sample() * self.scaling_factor
        z = z.permute(0, 2, 1, 3, 4)                                                # [B, 1, 16, h, w]
        cond = torch.cat([z, z.new_zeros(B, Fl - 1, *z.shape[2:])], dim=1)
        if drop:
            cond = torch.zeros_like(cond)
        return Condition(latents=cond.to(self.dtype), active=not drop)

    # ---------------- diffusion ----------------
    def sample_timesteps(self, batch_size, device):
        return torch.randint(0, self.scheduler.config.num_train_timesteps, (batch_size,), device=device).long()

    def fixed_timesteps(self, frac, batch_size, device):
        T = self.scheduler.config.num_train_timesteps
        return torch.full((batch_size,), min(T - 1, int(frac * T)), dtype=torch.long, device=device)

    def _coef(self, t, like):
        a = self.alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        return a.sqrt().to(like.dtype), (1 - a).sqrt().to(like.dtype)

    def add_noise(self, x0, eps, t):
        return self.scheduler.add_noise(x0, eps, t)

    def velocity_target(self, x0, eps, t):
        al, sg = self._coef(t, x0)
        return al * eps - sg * x0

    def rotary(self, F, h, w):
        cfg = self.transformer.config
        gs = self.vae_scale_spatial * cfg.patch_size
        grid = (h * self.vae_scale_spatial // gs, w * self.vae_scale_spatial // gs)
        base_w, base_h = self.rope_base[1] // gs, self.rope_base[0] // gs
        cos, sin = get_3d_rotary_pos_embed(embed_dim=cfg.attention_head_dim,
                                           crops_coords=_resize_crop_region(grid, base_w, base_h),
                                           grid_size=grid, temporal_size=F)
        return cos.to(self.device), sin.to(self.device)

    def forward(self, transformer, x_t, t, cond: Condition, ctx):
        B, F, C, h, w = x_t.shape
        out = transformer(hidden_states=torch.cat([x_t, cond.latents], dim=2), encoder_hidden_states=ctx,
                          timestep=t, image_rotary_emb=self.rotary(F, h, w), return_dict=False)[0]
        return out                                                                  # v_hat [B, F, 16, h, w]

    def x0_hat(self, out, x_t, t):
        return self.scheduler.get_velocity(out, x_t, t)                             # = a x_t - s v_hat

    def video_loss(self, out, x0, eps, x_t, t, cond):
        w = (1.0 / (1.0 - self.alphas_cumprod[t])).view(-1, 1, 1, 1, 1)
        err = w * (self.x0_hat(out, x_t, t).float() - x0.float()) ** 2
        return err.reshape(err.shape[0], -1).mean(dim=1).mean()

    def eps_hat(self, out, x_t, t):
        al, sg = self._coef(t, x_t)
        return (sg * x_t + al * self.x0_hat(out, x_t, t)).detach()

    # ---------------- inference ----------------
    def pipeline(self, transformer=None):
        from diffusers import CogVideoXImageToVideoPipeline
        return CogVideoXImageToVideoPipeline(tokenizer=self.tokenizer, text_encoder=self.text_encoder,
                                             vae=self.vae, transformer=transformer or self.transformer,
                                             scheduler=self.scheduler)
