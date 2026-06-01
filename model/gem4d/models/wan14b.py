"""Wan2.1-I2V-14B backbone (original Wan2.1 checkpoint layout, `Wan-AI/Wan2.1-I2V-14B-480P`).

49-frame recipe (image + text -> event-level video, 416 x 240):
  conditioning   first frame -> CLIP ViT-H/14 tokens (257) and its own VAE latent; the DiT input is
                 [noisy latent | 4 mask channels (1 on latent frame 0) | condition latent at frame 0,
                 zeros elsewhere]. Text = umT5-xxl, zero-padded to 512 tokens, all positions attended.
  flow matching  x_t = (1 - s) x0 + s eps, s = 3u / (1 + 2u), u ~ U(0, 1), t = 1000 s, v* = eps - x0.
                 Per-sample loss = mean squared error over all latent frames (the conditioned frame
                 included), weighted by the Gaussian timestep weight of `wan21.flow.FlowMatching`.
  sampling       30 steps of the discrete shift-3 schedule (999 -> 0), Euler steps in fp32, CFG 4 with
                 an empty negative prompt; CLIP sees the input image as given, the VAE a bicubic resize.
Geometry flow target (same parametrisation): v* = eps - g0.
"""
from types import SimpleNamespace
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .backbone import Condition, VideoBackbone
from .wan21 import FlowMatching, flow_step, load_wan21_components, sampling_timesteps


class Wan14BBackbone(VideoBackbone):
    name = "wan2.1-i2v-14b"

    def __init__(self, base_model: str, init_weights: Optional[str] = None, max_text_len: int = 512,
                 flow_shift: float = 3.0, load_text_encoder: bool = True):
        c = load_wan21_components(base_model, dit_weights=init_weights, dtype=torch.bfloat16,
                                  load_text_encoder=load_text_encoder, max_text_len=max_text_len)
        self.transformer, self.vae, self.text_encoder, self.tokenizer, self.clip = (
            c.dit, c.vae, c.text_encoder, c.tokenizer, c.clip)
        self.flow = FlowMatching(shift=flow_shift)
        self.flow_shift = float(flow_shift)
        self.latent_channels = self.vae.z_dim                               # 16
        self.spatial_patch = self.transformer.patch_size[1]                 # 2
        self.num_heads = self.transformer.num_heads                         # 40
        self.hidden_dim = self.transformer.hidden_dim                       # 5120
        self.device, self.dtype = torch.device("cpu"), torch.bfloat16

    def to(self, device, dtype):
        self.device, self.dtype = torch.device(device), dtype
        if self.text_encoder is not None:
            self.text_encoder.to(device, dtype=dtype)
        self.vae.to(device, dtype=torch.float32)                            # VAE stays fp32
        self.clip.to(device, dtype=torch.float16)                           # CLIP stays fp16
        return self

    @property
    def blocks(self):
        return self.transformer.blocks

    # ---------------- encoding ----------------
    @torch.no_grad()
    def encode_text(self, prompts: List[str]) -> torch.Tensor:
        ids, mask = self.tokenizer(prompts, return_mask=True, add_special_tokens=True)
        ids, mask = ids.to(self.device), mask.to(self.device, dtype=torch.bool)
        emb = self.text_encoder(ids, mask)
        for i, n in enumerate(mask.gt(0).sum(dim=1).tolist()):              # zero-pad after each prompt
            emb[i, n:] = 0
        return emb.to(self.dtype)

    @torch.no_grad()
    def _vae_encode(self, video_bcthw: torch.Tensor) -> torch.Tensor:
        """[B, 3, T, H, W] in [-1, 1] -> normalised latent [B, 16, T', h, w] (posterior mean)."""
        return self.vae.encode([v.to(self.device, torch.float32) for v in video_bcthw], device=self.device)

    @torch.no_grad()
    def _clip(self, frames_bchw: torch.Tensor) -> torch.Tensor:
        """[B, 3, H, W] in [-1, 1] -> CLIP tokens [B, 257, 1280]."""
        return self.clip([f.to(self.device, torch.float16)[:, None] for f in frames_bchw]).to(self.dtype)

    @torch.no_grad()
    def encode_video(self, video):
        return self._vae_encode(video.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4).to(self.dtype)

    @torch.no_grad()
    def encode_condition(self, video, x0, drop: bool) -> Condition:
        # The causal VAE encodes frame 0 on its own, so latent frame 0 of the clip is the image latent.
        return Condition(latents=x0[:, :1], active=not drop, extra={"clip": self._clip(video[:, 0])})

    # ---------------- diffusion ----------------
    def sample_timesteps(self, batch_size, device):
        # the recipe draws t in the model dtype (bf16), so the noise level is quantised the same way
        return self.flow.sample_t(batch_size, device, dtype=torch.bfloat16).float()

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
        x = x_t.permute(0, 2, 1, 3, 4).to(self.dtype)                       # [B, 16, F, h, w]
        if cond.active:
            cl, clip = cond.latents.permute(0, 2, 1, 3, 4), cond.extra["clip"]
        else:                                                                # no image: empty condition
            cl, clip = x[:, :, :0], torch.zeros(x.shape[0], 257, 1280, device=x.device, dtype=self.dtype)
        out = transformer(x=x, timestep=t.to(self.dtype), context=ctx.to(self.dtype), clip_fea=clip.to(self.dtype),
                          condition_latents=cl.to(self.dtype))
        return out.permute(0, 2, 1, 3, 4)                                   # v_hat [B, F, 16, h, w]

    def video_loss(self, out, x0, eps, x_t, t, cond):
        per_frame = ((out.float() - (eps - x0).float()) ** 2).mean(dim=(2, 3, 4))   # [B, F]
        return (per_frame.mean(dim=1) * self.flow.weight(t).to(per_frame.device)).mean()

    # ---------------- inference ----------------
    def pipeline(self, transformer=None):
        return Wan14BPipeline(self, transformer or self.transformer)


class Wan14BPipeline:
    """First frame + prompt -> video with the 49-frame sampler. Call signature follows diffusers'
    image-to-video pipelines (`pipe(image=..., prompt=...).frames[0]`, uint8 [F, H, W, 3])."""

    def __init__(self, backbone: Wan14BBackbone, transformer):
        self.bb, self.transformer = backbone, transformer

    def to(self, device):
        self.transformer.to(device)
        return self

    def enable_model_cpu_offload(self):
        raise NotImplementedError("--cpu_offload is not supported for wan2.1-i2v-14b (needs ~45 GB GPU memory)")

    def _context(self, text):
        with torch.autocast(self.bb.device.type, dtype=self.bb.dtype):
            return self.bb.encode_text([text]).float()

    @torch.inference_mode()
    def __call__(self, image, prompt: str, height: int = 240, width: int = 416, num_frames: int = 49,
                 num_inference_steps: int = 30, guidance_scale: float = 4.0, negative_prompt: str = "",
                 shift: Optional[float] = None, generator: Optional[torch.Generator] = None,
                 output_type: str = "np"):
        bb, dit = self.bb, self.transformer
        device, dtype = bb.device, bb.dtype
        if height % 16 or width % 16 or (num_frames - 1) % 4:
            raise ValueError("height / width must be multiples of 16 and num_frames of the form 4k+1")
        shift = bb.flow_shift if shift is None else float(shift)
        img = torch.from_numpy(np.array(image.convert("RGB"), copy=True)).permute(2, 0, 1)
        img = img.to(device=device, dtype=torch.float32).div(255).sub(0.5).div(0.5)[None]   # [1, 3, H0, W0]

        latents = torch.randn((1, bb.latent_channels, (num_frames - 1) // 4 + 1, height // 8, width // 8),
                              dtype=torch.float32, device=device, generator=generator)
        use_cfg = not np.isclose(guidance_scale, 1.0)
        context = self._context(prompt)
        if use_cfg:
            context = torch.cat([context, self._context(negative_prompt)])
        with torch.autocast(device.type, dtype=torch.float16):
            clip = bb.clip([img[0, :, None].half()]).to(dtype)               # CLIP sees the image as given
        cond = bb._vae_encode(F.interpolate(img, size=(height, width), mode="bicubic",
                                            align_corners=False)[:, :, None]).to(dtype)
        if use_cfg:
            clip, cond = torch.cat([clip, clip]), torch.cat([cond, cond])
        ts = sampling_timesteps(num_inference_steps, shift, device)
        for i, t in enumerate(ts):
            x = torch.cat([latents, latents]) if use_cfg else latents
            with torch.autocast(device.type, dtype=dtype):
                pred = dit(x=x.to(dtype), timestep=t.expand(x.shape[0]), context=context.to(dtype),
                           clip_fea=clip.to(dtype), condition_latents=cond.to(dtype)).float()
            if use_cfg:
                pos, neg = pred.chunk(2)
                pred = neg + (pos - neg) * guidance_scale
            latents = flow_step(pred, latents, t, ts[i + 1] if i + 1 < len(ts) else 0)
        video = bb.vae.decode(latents, device=device)[0].float().clamp(-1, 1)        # [3, F, H, W]
        frames = ((video + 1.0) * 127.5).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
        return SimpleNamespace(frames=[frames])
