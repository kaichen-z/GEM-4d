"""Wan2.2-TI2V-5B backbone (diffusers, `Wan-AI/Wan2.2-TI2V-5B-Diffusers`).

Rectified flow, as in Wan:
    x_t = (1 - s) x0 + s eps,  t = 1000 s,  target v* = eps - x0,  L_vid = mean (v_hat - v*)^2
TI2V image conditioning (as in diffusers' WanImageToVideoPipeline with expand_timesteps=True):
the first latent frame is replaced by the clean image latent and its tokens get timestep 0.
That frame is excluded from the video loss.
Geometry flow target (same parametrisation): v* = eps - g0.
"""
from typing import List, Optional

import torch
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanTransformer3DModel
from transformers import AutoTokenizer, UMT5EncoderModel

from ..utils.compat import patch_sdpa_enable_gqa
from .backbone import Condition, VideoBackbone
from .cogvideox import load_transformer_weights

patch_sdpa_enable_gqa()


class WanTI2VBackbone(VideoBackbone):
    name = "wan2.2-ti2v-5b"

    def __init__(self, base_model: str, init_weights: Optional[str] = None, max_text_len: int = 512,
                 flow_shift: float = 5.0):
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, subfolder="tokenizer")
        self.text_encoder = UMT5EncoderModel.from_pretrained(base_model, subfolder="text_encoder")
        self.vae = AutoencoderKLWan.from_pretrained(base_model, subfolder="vae", torch_dtype=torch.float32)
        self.scheduler = UniPCMultistepScheduler.from_pretrained(base_model, subfolder="scheduler")
        self.transformer = WanTransformer3DModel.from_pretrained(
            base_model, subfolder="transformer", torch_dtype=torch.bfloat16)
        self.extra_init_tensors = {}
        if init_weights:
            sd = load_transformer_weights(init_weights)
            own = self.transformer.state_dict()
            missing = [k for k in own if k not in sd]
            if missing:
                raise RuntimeError(f"{init_weights}: {len(missing)} transformer tensors missing")
            self.transformer.load_state_dict({k: sd[k] for k in own}, strict=True)
            self.extra_init_tensors = {k: v for k, v in sd.items() if k not in own}
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        cfg = self.transformer.config
        self.max_text_len = max_text_len
        self.flow_shift = float(flow_shift)
        self.latent_channels = self.vae.config.z_dim                     # 48
        self.spatial_patch = cfg.patch_size[1]                            # 2
        self.num_heads = cfg.num_attention_heads                          # 24
        self.hidden_dim = cfg.num_attention_heads * cfg.attention_head_dim  # 3072
        self.device, self.dtype = torch.device("cpu"), torch.float32

    def to(self, device, dtype):
        self.device, self.dtype = device, dtype
        self.text_encoder.to(device, dtype=dtype)
        self.vae.to(device)                                               # VAE stays fp32
        z = self.vae.config.z_dim
        self.lat_mean = torch.tensor(self.vae.config.latents_mean, device=device).view(1, z, 1, 1, 1)
        self.lat_inv_std = 1.0 / torch.tensor(self.vae.config.latents_std, device=device).view(1, z, 1, 1, 1)
        return self

    @property
    def blocks(self):
        return self.transformer.blocks

    # ---------------- encoding ----------------
    @torch.no_grad()
    def encode_text(self, prompts: List[str]) -> torch.Tensor:
        tok = self.tokenizer(prompts, padding="max_length", max_length=self.max_text_len, truncation=True,
                             add_special_tokens=True, return_attention_mask=True, return_tensors="pt")
        ids, mask = tok.input_ids.to(self.device), tok.attention_mask.to(self.device)
        emb = self.text_encoder(ids, mask).last_hidden_state.to(self.dtype)
        lens = mask.gt(0).sum(dim=1)
        out = torch.zeros_like(emb)                                       # zero-pad after each prompt
        for i, n in enumerate(lens.tolist()):
            out[i, :n] = emb[i, :n]
        return out

    def _encode(self, pixels_bcfhw, sample: bool):
        dist = self.vae.encode(pixels_bcfhw.float()).latent_dist
        z = dist.sample() if sample else dist.mode()
        return (z - self.lat_mean) * self.lat_inv_std                     # [B, 48, F', h, w]

    @torch.no_grad()
    def encode_video(self, video):
        return self._encode(video.permute(0, 2, 1, 3, 4), sample=True).permute(0, 2, 1, 3, 4).to(self.dtype)

    @torch.no_grad()
    def encode_condition(self, video, x0, drop: bool) -> Condition:
        z = self._encode(video[:, :1].permute(0, 2, 1, 3, 4), sample=False)  # first frame, as the pipeline
        return Condition(latents=z.permute(0, 2, 1, 3, 4).to(self.dtype), active=not drop)

    # ---------------- diffusion ----------------
    def sample_timesteps(self, batch_size, device):
        u = torch.rand(batch_size, device=device)
        s = self.flow_shift * u / (1 + (self.flow_shift - 1) * u)        # shifted sigma, as at inference
        return s * 1000.0

    def fixed_timesteps(self, frac, batch_size, device):
        return torch.full((batch_size,), 1000.0 * frac, device=device)

    @staticmethod
    def _s(t, like):
        return (t / 1000.0).view(-1, 1, 1, 1, 1).to(like.dtype)

    def add_noise(self, x0, eps, t):
        s = self._s(t, x0)
        return (1 - s) * x0 + s * eps

    def velocity_target(self, x0, eps, t):
        return eps - x0

    def forward(self, transformer, x_t, t, cond: Condition, ctx):
        B, F, C, h, w = x_t.shape
        p = self.spatial_patch
        if cond.active:
            x_in = torch.cat([cond.latents, x_t[:, 1:]], dim=1)                     # frame 0 = clean image
            tok_t = t.view(B, 1, 1, 1).expand(B, F, h // p, w // p).clone()
            tok_t[:, 0] = 0
            timestep = tok_t.reshape(B, -1)                                           # [B, seq_len]
        else:
            x_in, timestep = x_t, t
        out = transformer(hidden_states=x_in.permute(0, 2, 1, 3, 4).to(self.dtype), timestep=timestep,
                          encoder_hidden_states=ctx, return_dict=False)[0]
        return out.permute(0, 2, 1, 3, 4)                                             # v_hat [B, F, 48, h, w]

    def video_loss(self, out, x0, eps, x_t, t, cond):
        err = (out.float() - (eps - x0).float()) ** 2
        if cond.active:
            err = err[:, 1:]                                                          # conditioned frame
        return err.mean()

    def eps_hat(self, out, x_t, t):
        return (x_t + (1 - self._s(t, x_t)) * out).detach()                          # x_t + (1-s) v_hat

    # ---------------- inference ----------------
    def pipeline(self, transformer=None):
        from diffusers import WanImageToVideoPipeline
        return WanImageToVideoPipeline(tokenizer=self.tokenizer, text_encoder=self.text_encoder, vae=self.vae,
                                       scheduler=self.scheduler, transformer=transformer or self.transformer,
                                       expand_timesteps=True)
