"""Frozen geometry foundation model G (DepthAnything-3) and the frozen projector that maps its
features onto the video backbone's latent grid.

    g_0 = scale * project( resize( temporal_pool( compress( DA3_layer_k(video) ) ) ) )

`DA3Teacher` returns DA3 ViT-L features of one intermediate layer, `[B, 1024, F, 24, 24]` for a
336x336 input. `GeometryProjector` turns them into a latent with the same shape as the video
latent `[B, F', C, h, w]` so geometry and video can share one noise sample. All of it is frozen.
"""
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

_DEPENDENCY = Path(__file__).resolve().parents[2] / "dependency"


class DA3Teacher(nn.Module):
    """DepthAnything-3 feature extractor (frozen, kept in fp32)."""

    def __init__(self, pretrained: str = "depth-anything/DA3-LARGE-1.1", feat_layer: int = 12,
                 input_size: int = 336, ref_view_strategy: str = "saddle_balanced"):
        super().__init__()
        if str(_DEPENDENCY) not in sys.path:
            sys.path.insert(0, str(_DEPENDENCY))
        from depth_anything_3.api import DepthAnything3

        root = DepthAnything3.from_pretrained(pretrained).model
        self.net = root.da3 if hasattr(root, "da3") else root   # skip the metric branch if nested
        self.net.eval().requires_grad_(False)
        self.feat_layer = int(feat_layer)
        self.input_size = int(input_size)
        self.patch = int(getattr(self.net, "PATCH_SIZE", 14))
        self.grid = self.input_size // self.patch
        self.ref_view_strategy = ref_view_strategy

    def train(self, mode: bool = True):  # always eval
        return super().train(False)

    @torch.no_grad()
    def forward(self, video01: torch.Tensor) -> torch.Tensor:
        """video01: [B, F, 3, H, W] in [0, 1]  ->  features [B, 1024, F, grid, grid] (fp32)."""
        b, f = video01.shape[:2]
        x = rearrange(video01.float(), "b f c h w -> (b f) c h w")
        x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        x = x.clamp(0.0, 1.0)
        mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x = rearrange((x - mean) / std, "(b f) c h w -> b f c h w", b=b)
        out = self.net(x, extrinsics=None, intrinsics=None, export_feat_layers=[self.feat_layer],
                       infer_gs=False, use_ray_pose=False, ref_view_strategy=self.ref_view_strategy)
        feat = out.aux[f"feat_layer_{self.feat_layer}"].float()          # [B, F, P, C]
        feat = feat.reshape(b, f, self.grid, self.grid, feat.shape[-1])
        return feat.permute(0, 4, 1, 2, 3).contiguous()                   # [B, C, F, g, g]


class GeometryProjector(nn.Module):
    """Frozen random projection of DA3 features onto the video latent grid.

    compress: 1x1x1 conv 1024 -> 256 (+GELU), xavier init (gain 0.5)
    temporal: stride-`temporal_down` conv initialised to a box average (49 frames -> 13)
    resize:   trilinear to the latent's (h, w)
    project:  1x1x1 conv 256 -> latent_channels
    The parameter names match the CogVideoX checkpoints (`compress.*`, `temporal.*`, `project.*`),
    so a released checkpoint's projector can be loaded to keep the same geometry space.
    """

    def __init__(self, in_ch: int = 1024, hidden: int = 256, latent_channels: int = 16,
                 temporal_down: int = 4, scale: float = 1.0):
        super().__init__()
        self.compress = nn.Sequential(nn.Conv3d(in_ch, hidden, kernel_size=1), nn.GELU())
        self.temporal = nn.Conv3d(hidden, hidden, kernel_size=(temporal_down, 1, 1),
                                  stride=(temporal_down, 1, 1), padding=(temporal_down - 1, 0, 0))
        self.project = nn.Conv3d(hidden, latent_channels, kernel_size=1)
        self.scale = float(scale)
        nn.init.xavier_uniform_(self.compress[0].weight, gain=0.5)
        nn.init.zeros_(self.compress[0].bias)
        with torch.no_grad():
            self.temporal.weight.zero_()
            for i in range(hidden):
                self.temporal.weight[i, i, :, 0, 0] = 1.0 / temporal_down
            self.temporal.bias.zero_()
        self.requires_grad_(False)

    @torch.no_grad()
    def forward(self, feats: torch.Tensor, latent_thw) -> torch.Tensor:
        """feats [B, 1024, F, g, g]  ->  geometry latent g_0 [B, F', C, h, w] (F' from the temporal conv)."""
        p = next(self.parameters())
        x = self.temporal(self.compress(feats.to(p.dtype)))
        T, h, w = latent_thw
        if x.shape[2] != T:
            raise ValueError(f"geometry frames {x.shape[2]} != latent frames {T}; "
                             "use 4k+1 video frames (e.g. 49)")
        x = F.interpolate(x, size=(T, h, w), mode="trilinear", align_corners=False)
        x = self.project(x) * self.scale
        return x.permute(0, 2, 1, 3, 4).contiguous()
