"""GEM-4D training module: video DiT + Geometry DiT (+ frozen geometry projector).

    L = L_vid + alpha * ramp(step) * L_geo

L_geo is the masked geometry-flow loss of the paper: the Geometry DiT, conditioned on the video
DiT's mid-layer features m_t, predicts the velocity of the noised geometry latent. Geometry and
video share the same noise sample (coupled noise). With `geometry=False` the module reduces to
plain RGB fine-tuning of the backbone.
"""
from typing import Optional

import torch
import torch.nn as nn

from ..geometry.teacher import GeometryProjector
from ..heads.geo_dit import GeoDiT
from .backbone import Condition, VideoBackbone


class GEM4D(nn.Module):
    def __init__(self, backbone: VideoBackbone, geometry: bool = True, hook_layer: Optional[int] = None,
                 geo_depth: int = 4, mask_ratio: float = 0.5, geo_scale: float = 1.0,
                 use_eps_hat: bool = False, x0_target: bool = False, coupled_noise: bool = True):
        super().__init__()
        self.backbone = backbone
        self.transformer = backbone.transformer
        self.geometry = geometry
        self.x0_target = x0_target
        self.coupled_noise = coupled_noise
        if geometry:
            self.projector = GeometryProjector(latent_channels=backbone.latent_channels, scale=geo_scale)
            self.geo_dit = GeoDiT(backbone.hidden_dim, backbone.num_heads, depth=geo_depth,
                                  in_ch=backbone.latent_channels, patch=backbone.spatial_patch,
                                  mask_ratio=mask_ratio, use_eps_hat=use_eps_hat)
            n = len(backbone.blocks)
            self.hook_layer = n // 2 if hook_layer is None or hook_layer < 0 else int(hook_layer)
            assert 0 <= self.hook_layer < n, f"hook_layer {self.hook_layer} not in [0, {n})"
            self._feats = []
            self.capture = True          # switched off for sampling (validation / inference)
            backbone.blocks[self.hook_layer].register_forward_hook(self._capture)
            # reuse a checkpoint's projector (released CogVideoX checkpoints carry compress/temporal/project)
            proj_sd = {k: v for k, v in backbone.extra_init_tensors.items()
                       if k.split(".")[0] in ("compress", "temporal", "project")}
            if proj_sd:
                self.projector.load_state_dict(proj_sd, strict=False)

    def _capture(self, module, inp, out):
        # Gradient checkpointing may re-run the block; the last call is the live one. Not detached:
        # the geometry loss must reach the backbone through these features.
        if self.capture:
            self._feats.append(self.backbone.block_output(out))

    @torch.no_grad()
    def geometry_latent(self, da3_feats: torch.Tensor, latent_shape) -> torch.Tensor:
        F, _, h, w = latent_shape
        return self.projector(da3_feats, (F, h, w))

    def forward(self, x0: torch.Tensor, cond: Condition, ctx: torch.Tensor, t: torch.Tensor,
                g0: Optional[torch.Tensor] = None):
        """Returns (video_loss, geo_loss); the trainer combines them as L_vid + alpha * ramp * L_geo."""
        bb = self.backbone
        eps = torch.randn_like(x0)
        x_t = bb.add_noise(x0, eps, t)
        self._feats = []
        out = bb.forward(self.transformer, x_t, t, cond, ctx)
        l_vid = bb.video_loss(out, x0, eps, x_t, t, cond)
        if not self.geometry or g0 is None:
            return l_vid, l_vid.new_zeros(())

        g0 = g0.to(x0.dtype)
        eps_g = eps if self.coupled_noise else torch.randn_like(g0)
        g_t = bb.add_noise(g0, eps_g, t)
        target = g0 if self.x0_target else bb.velocity_target(g0, eps_g, t)
        eps_hat = bb.eps_hat(out, x_t, t) if self.geo_dit.use_eps_hat else None
        v_geo = self.geo_dit(self._feats[-1], bb.geo_timestep(t), g_t, eps_hat=eps_hat)
        l_geo = ((v_geo.float() - target.float()) ** 2).mean()
        self._feats = []
        return l_vid, l_geo
