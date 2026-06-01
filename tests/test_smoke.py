"""CPU smoke tests with tiny random models (no weights needed).

    PYTHONPATH=model:. python -m pytest -q tests/test_smoke.py      # or: python tests/test_smoke.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "model"), str(ROOT)]

from gem4d import GEM4D                                          # noqa: E402
from gem4d.models.backbone import Condition                      # noqa: E402
from gem4d.models.wan5b import Wan5BBackbone                     # noqa: E402

torch.manual_seed(0)
B, F, H, W = 1, 13, 8, 12            # latent grid (F'=13 <-> 49 frames)


def tiny_wan5b():
    from diffusers import WanTransformer3DModel
    bb = Wan5BBackbone.__new__(Wan5BBackbone)
    bb.transformer = WanTransformer3DModel(
        patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=16, in_channels=48, out_channels=48,
        text_dim=32, freq_dim=32, ffn_dim=64, num_layers=4, rope_max_seq_len=64)
    bb.flow_shift = 5.0
    bb.latent_channels, bb.spatial_patch, bb.num_heads, bb.hidden_dim = 48, 2, 2, 32
    bb.device, bb.dtype = torch.device("cpu"), torch.float32
    return bb


def condition(bb, x0):
    if bb.name == "wan2.1-i2v-14b":
        return Condition(latents=x0[:, :1], active=True, extra={"clip": torch.randn(B, 257, 1280)})
    return Condition(latents=torch.randn(B, 1, bb.latent_channels, H, W), active=True)


def run_step(bb, geometry=True):
    C = bb.latent_channels
    model = GEM4D(bb, geometry=geometry, hook_layer=None, geo_depth=2, mask_ratio=0.5)
    x0 = torch.randn(B, F, C, H, W)
    cond = condition(bb, x0)
    ctx = torch.randn(B, 8, 32)
    g0 = model.geometry_latent(torch.randn(B, 1024, 49, 4, 4), x0.shape[1:]) if geometry else None
    t = bb.sample_timesteps(B, "cpu")
    l_vid, l_geo = model(x0, cond, ctx, t, g0)
    assert torch.isfinite(l_vid) and torch.isfinite(l_geo)
    # make the zero-init Geometry DiT output non-trivial so the geometry gradient is visible
    if geometry:
        with torch.no_grad():
            model.geo_dit.out.weight.normal_(std=0.02)
            for blk in model.geo_dit.blocks:
                blk.g2.fill_(1.0)
        model.zero_grad()
        _, l_geo = model(x0, cond, ctx, t, g0)
        l_geo.backward()
        early = bb.blocks[0]
        late = bb.blocks[-1]
        g_early = sum(p.grad.abs().sum() for p in early.parameters() if p.grad is not None)
        g_late = sum(p.grad.abs().sum() for p in late.parameters() if p.grad is not None)
        assert g_early > 0, "geometry loss must reach blocks before the hook"
        assert g_late == 0, "blocks after the hook must not get geometry gradient"
        assert all(p.grad is None for p in model.projector.parameters()), "projector must stay frozen"
    return float(l_vid), float(l_geo)


def test_geo_dit_masking():
    from gem4d.heads.geo_dit import GeoDiT
    m = GeoDiT(32, 2, depth=1, in_ch=16, patch=2)
    v = m(torch.randn(1, F * (H // 2) * (W // 2), 32), torch.tensor([500.0]), torch.randn(1, F, 16, H, W))
    assert v.shape == (1, F, 16, H, W)


def test_projector_shapes():
    from gem4d.geometry.teacher import GeometryProjector
    p = GeometryProjector(latent_channels=48)
    g = p(torch.randn(1, 1024, 49, 24, 24), (13, 30, 40))
    assert g.shape == (1, 13, 48, 30, 40)
    assert not any(q.requires_grad for q in p.parameters())


def test_wan5b_step():
    print("wan5b", run_step(tiny_wan5b()))


def test_wan5b_rgb_only():
    l_vid, l_geo = run_step(tiny_wan5b(), geometry=False)
    assert l_geo == 0.0


def test_wan5b_t2v_mode():
    bb = tiny_wan5b()
    x0 = torch.randn(B, F, 48, H, W)
    out = bb.forward(bb.transformer, x0, bb.sample_timesteps(B, "cpu"), Condition(x0[:, :1], False),
                     torch.randn(B, 8, 32))
    assert out.shape == x0.shape


def test_dataset_and_manifest():
    import cv2
    sys.path.insert(0, str(ROOT))
    from training.data.video_dataset import ManifestVideoDataset, collate
    with tempfile.TemporaryDirectory() as d:
        w = cv2.VideoWriter(os.path.join(d, "a.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 8, (320, 256))
        for i in range(30):
            w.write(np.full((256, 320, 3), i * 5, np.uint8))
        w.release()
        os.makedirs(os.path.join(d, "frames"))
        for i in range(60):
            cv2.imwrite(os.path.join(d, "frames", f"{i}.png"), np.full((300, 400, 3), 128, np.uint8))
        json.dump([{"video": "a.mp4", "prompt": "Pick cube", "robot": "Franka robot arm"},
                   ["push", "frames"]], open(os.path.join(d, "m.json"), "w"))
        ds = ManifestVideoDataset(os.path.join(d, "m.json"), caption_dropout=0.0)
        b = collate([ds[0], ds[1]])
        assert b["video"].shape == (2, 49, 3, 480, 640)
        assert b["prompt"] == ["pick cube franka robot arm", "push"]
        assert -1.0 <= b["video"].min() and b["video"].max() <= 1.0
        # event-level sampling: frames [10, 20) of the clip resampled to 49, prompt kept verbatim
        json.dump([{"video": "a.mp4", "prompt": "Pick cube", "start": 10, "end": 20}], open(os.path.join(d, "e.json"), "w"))
        ds = ManifestVideoDataset(os.path.join(d, "e.json"), caption_dropout=0.0, height=240, width=416,
                                  frame_sampling="event", lowercase_prompt=False)
        s = ds[0]
        assert s["video"].shape == (49, 3, 240, 416) and s["prompt"] == "Pick cube"
        from training.data.video_dataset import read_video
        decoded = [int(f[0, 0, 0]) for f in read_video(Path(d) / "a.mp4")]
        level = ((s["video"][:, 0, 0, 0] + 1) * 127.5).round().int().tolist()
        assert level == [decoded[10 + i] for i in np.linspace(0, 9, 49).round().astype(int)]


def test_save_mp4_colors():
    import imageio.v3 as iio
    from gem4d.utils.video_io import save_mp4
    with tempfile.TemporaryDirectory() as d:
        f = np.zeros((16, 32, 32, 3), np.uint8); f[..., 0] = 200; f[..., 2] = 30
        save_mp4(list(f), os.path.join(d, "a.mp4"), fps=8)
        back = iio.imread(os.path.join(d, "a.mp4"), plugin="pyav")
        assert abs(int(back[..., 0].mean()) - 200) < 15 and abs(int(back[..., 2].mean()) - 30) < 15


def test_checkpoint_prune():
    from training.train_utils.checkpoint import list_steps, prune
    with tempfile.TemporaryDirectory() as d:
        for s in (500, 1000, 1500, 2000, 2500, 3000, 3500):
            os.makedirs(os.path.join(d, f"step_{s:06d}"))
        prune(d, keep_last=2, keep_every=2000)
        assert list_steps(d) == [2000, 3000, 3500]



if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
