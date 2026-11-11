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
from gem4d.models.wan14b import Wan14BBackbone, Wan14BPipeline   # noqa: E402
from gem4d.models.wan21 import FlowMatching, WanDiT              # noqa: E402

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


def tiny_dit():
    return WanDiT(dim=32, num_heads=2, ffn_dim=64, num_layers=4, text_dim=32, freq_dim=32)


def tiny_wan14b():
    bb = Wan14BBackbone.__new__(Wan14BBackbone)
    bb.transformer = tiny_dit()
    bb.flow, bb.flow_shift = FlowMatching(shift=3.0), 3.0
    bb.latent_channels, bb.spatial_patch, bb.num_heads, bb.hidden_dim = 16, 2, 2, 32
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


def test_wan14b_step():
    print("wan14b", run_step(tiny_wan14b()))


def test_wan14b_rgb_only_and_no_image():
    bb = tiny_wan14b()
    l_vid, l_geo = run_step(bb, geometry=False)
    assert l_geo == 0.0
    x0 = torch.randn(B, F, 16, H, W)
    out = bb.forward(bb.transformer, x0, bb.sample_timesteps(B, "cpu"), Condition(x0[:, :1], False),
                     torch.randn(B, 8, 32))
    assert out.shape == x0.shape


def test_wan14b_condition_channels():
    dit = tiny_dit()
    x = torch.randn(2, 16, F, H, W)
    y = dit.build_condition(x, torch.ones(2, 16, 1, H, W))
    assert y.shape == (2, 20, F, H, W)
    assert (y[:, :4, 0] == 1).all() and (y[:, :4, 1:] == 0).all()      # mask: latent frame 0 only
    assert (y[:, 4:, 0] == 1).all() and (y[:, 4:, 1:] == 0).all()      # condition latent, zero-padded


def test_wan14b_flow_schedule():
    from gem4d.models.wan21 import flow_step, sampling_timesteps
    ts = sampling_timesteps(30, 3.0, "cpu")
    assert len(ts) == 30 and int(ts[0]) == 999 and (ts[1:] < ts[:-1]).all()
    fm = FlowMatching(shift=3.0)
    grid = torch.linspace(1.0, 0.0, 1001, dtype=torch.float64)[:-1]
    t = 3 * grid / (1 + 2 * grid) * 1000
    assert abs(float(fm.weight(t).mean()) - 1.0) < 1e-4                  # normalised over the train grid
    x0, eps = torch.randn(4, 3), torch.randn(4, 3)
    t0 = torch.tensor(700.0)
    x_t = (1 - 0.7) * x0 + 0.7 * eps
    assert torch.allclose(flow_step(eps - x0, x_t, t0, 0), x0, atol=1e-5)  # exact velocity -> clean sample


def test_wan14b_pipeline_loop():
    from types import SimpleNamespace
    from PIL import Image
    bb = tiny_wan14b()
    bb.encode_text = lambda prompts: torch.randn(len(prompts), 8, 32)
    bb.clip = lambda frames: torch.randn(len(frames), 257, 1280)
    bb.vae = SimpleNamespace(
        encode=lambda videos, device: torch.randn(len(videos), 16, 1, 30, 52),
        decode=lambda lat, device: torch.zeros(1, 3, 4 * (lat.shape[2] - 1) + 1, 240, 416))
    pipe = Wan14BPipeline(bb, bb.transformer)
    out = pipe(image=Image.new("RGB", (416, 240)), prompt="open the drawer", num_inference_steps=3,
               generator=torch.Generator().manual_seed(0)).frames[0]
    assert out.shape == (49, 240, 416, 3) and out.dtype == np.uint8


def test_wan14b_checkpoint_roundtrip():
    from gem4d.models.wan21.loader import build_dit
    dit = tiny_dit().eval()
    with tempfile.TemporaryDirectory() as d:
        dit.save_pretrained(d, max_shard_size="200KB")                   # several shards + index
        assert len(list(Path(d).glob("diffusion_pytorch_model-*.safetensors"))) > 1
        back = build_dit(d, dtype=torch.float32).eval()
    x, t, ctx = torch.randn(1, 16, F, H, W), torch.tensor([500.0]), torch.randn(1, 8, 32)
    cl, clip = torch.randn(1, 16, 1, H, W), torch.randn(1, 257, 1280)
    with torch.no_grad():
        assert torch.equal(dit(x, t, ctx, clip_fea=clip, condition_latents=cl),
                           back(x, t, ctx, clip_fea=clip, condition_latents=cl))


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


def _synthetic_4d(drift=1.0):
    """Plane at depth 2 seen by a camera moving +0.1 in x; a 12x12 patch moves by (0, 0.2, 0).
    drift scales frame 1's depth (a global DA3-like scale error)."""
    from gem4d.process4d import unproject
    h, w, f = 48, 64, 40.0
    K = torch.tensor([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]]).expand(2, 3, 3)
    c2w = torch.eye(4).repeat(2, 1, 1)
    c2w[1, 0, 3] = 0.1
    depth = torch.full((2, h, w), 2.0)
    points = unproject(depth, K, c2w)
    patch = torch.zeros(h, w, dtype=torch.bool)
    patch[10:22, 20:32] = True
    delta = torch.tensor([0.0, 0.2, 0.0])
    moved = points[0] + patch[..., None] * delta                                 # world position at t+1
    cam1 = moved - c2w[1, :3, 3]
    q = (cam1 / cam1[..., 2:]) @ K[0].T
    v, u = torch.meshgrid(torch.arange(h) + 0.5, torch.arange(w) + 0.5, indexing="ij")
    fw = torch.stack([q[..., 0] - u, q[..., 1] - v])[None]
    if drift != 1.0:
        depth = depth.clone()
        depth[1] *= drift
        points = unproject(depth, K, c2w)
    return points, depth, fw, patch, delta


def test_process4d_scene_flow():
    from gem4d.process4d import scene_flow
    from gem4d.process4d.scene_flow import _DEPENDENCY
    assert (_DEPENDENCY / "depth_anything_3" / "api.py").is_file()
    points, depth, fw, patch, delta = _synthetic_4d()
    ok = torch.ones(depth.shape, dtype=torch.bool)
    sf, valid, dyn, thr, A, shake = scene_flow(points, ok, fw, -fw, gap=1, fb_abs=1e9, depth=depth)
    inner = torch.zeros_like(patch)
    inner[12:20, 22:30] = True
    static = valid[0] & ~patch
    static[8:24, 18:34] = False                                                  # away from the patch border
    assert valid[0].float().mean() > 0.8
    assert sf[0][static].abs().max() < 1e-4, sf[0][static].abs().max()
    assert torch.allclose(sf[0][inner & valid[0]], delta.expand(int((inner & valid[0]).sum()), 3), atol=1e-4)
    assert dyn[0][inner & valid[0]].all() and not dyn[0][static].any()
    assert 0.03 < float(thr[0]) < 0.05                                           # 2 % of the median depth
    assert torch.allclose(A[0], torch.eye(4), atol=1e-4)                         # nothing to correct


def test_process4d_alignment_removes_depth_drift():
    from gem4d.process4d import scene_flow
    points, depth, fw, patch, delta = _synthetic_4d(drift=1.06)
    ok = torch.ones(depth.shape, dtype=torch.bool)
    inner = torch.zeros_like(patch)
    inner[12:20, 22:30] = True
    static = ~patch
    static[8:24, 18:34] = False
    sf, valid, dyn, thr, A, shake = scene_flow(points, ok, fw, -fw, gap=1, fb_abs=1e9, depth=depth, align=False)
    # without alignment the drift moves the whole static scene and hides the real motion
    assert sf[0][static & valid[0]].norm(dim=-1).median() > 0.05 and not dyn[0][inner & valid[0]].all()
    sf, valid, dyn, thr, A, shake = scene_flow(points, ok, fw, -fw, gap=1, fb_abs=1e9, depth=depth)
    m = static & valid[0]
    assert sf[0][m].norm(dim=-1).max() < 1e-3 and not dyn[0][m].any()
    assert abs(float(torch.det(A[0, :3, :3])) ** (1 / 3) - 1 / 1.06) < 1e-3
    got = sf[0][inner & valid[0]]
    assert dyn[0][inner & valid[0]].all() and (got - delta).norm(dim=-1).max() < 1e-3


def test_process4d_foreground_and_shake_gates():
    from gem4d.process4d import scene_flow
    points, depth, fw, patch, delta = _synthetic_4d()
    ok = torch.ones(depth.shape, dtype=torch.bool)
    inner = torch.zeros_like(patch)
    inner[12:20, 22:30] = True
    # foreground mask excludes the moving patch: nothing may be dynamic
    fg = torch.zeros(depth.shape, dtype=torch.bool)
    *_, dyn, _, _, _ = scene_flow(points, ok, fw, -fw, gap=1, fb_abs=1e9, depth=depth, foreground=fg)
    assert not dyn.any()
    # foreground covers the patch: it is found, and nothing outside the foreground is dynamic
    fg[:, 8:24, 18:34] = True
    sf, valid, dyn, thr, A, shake = scene_flow(points, ok, fw, -fw, gap=1, fb_abs=1e9, depth=depth, foreground=fg)
    assert dyn[0][inner & valid[0]].all() and not dyn[0][~fg[0]].any()
    # shake = median background flow (the camera translation); a huge min_flow_px suppresses everything
    bg = valid[0] & ~fg[0]
    assert torch.allclose(shake[0], fw[0].permute(1, 2, 0)[bg].median(0).values)
    *_, dyn, _, _, _ = scene_flow(points, ok, fw, -fw, gap=1, fb_abs=1e9, depth=depth, foreground=fg,
                                  min_flow_px=1e6)
    assert not dyn.any()


def test_process4d_viewer_and_summary():
    from gem4d.process4d import Result4D, viewer_sequence, write_viewer
    from gem4d.process4d.viewer import summary_frames
    from gem4d.process4d import scene_flow
    points, depth, fw, patch, delta = _synthetic_4d()
    ok = torch.ones(depth.shape, dtype=torch.bool)
    sf, valid, dyn, thr, A, shake = scene_flow(points, ok, fw, -fw, gap=1, fb_abs=1e9, depth=depth)
    h, w = depth.shape[1:]
    K = np.array([[40.0, 0, w / 2], [0, 40.0, h / 2], [0, 0, 1]])
    c2w = np.stack([np.eye(4), np.eye(4)])
    c2w[1, 0, 3] = 0.1
    r = Result4D(images=np.zeros((2, h, w, 3), np.uint8), depth=depth.numpy(), conf=np.ones((2, h, w), np.float32),
                 K=np.stack([K, K]), c2w=c2w, points=points.numpy(), conf_ok=ok.numpy(),
                 flow=fw.permute(0, 2, 3, 1).numpy(), sf=sf.numpy(), valid=valid.numpy(), dynamic=dyn.numpy(),
                 gap=1, threshold=thr.numpy(), align=A.numpy(), shake=shake.numpy(),
                 foreground=np.ones(depth.shape, bool))
    assert summary_frames(r).shape == (1, h, 2 * w, 3)                     # frame | depth & scene-flow arrows
    seq = viewer_sequence(r, "toy", max_points=500)
    fr = seq["frames"][0]
    assert fr["n"] == 500 and fr["m"] == int(dyn[0].sum()) and len(fr["c2w"]) == 16
    import base64
    d0 = np.frombuffer(base64.b64decode(fr["dyn0"]), np.float32).reshape(-1, 3)
    d1 = np.frombuffer(base64.b64decode(fr["dyn1"]), np.float32).reshape(-1, 3)
    move = (d1 - d0) / np.linalg.norm(d1 - d0, axis=1, keepdims=True)
    assert np.allclose(move, [0, -1, 0], atol=1e-3)                              # +y (down) in OpenCV = -y in three.js
    with tempfile.TemporaryDirectory() as d:
        write_viewer(os.path.join(d, "v.html"), [seq])
        html = open(os.path.join(d, "v.html")).read()
        assert "window.P4D_DATA = [" in html and "__DATA__" not in html



def test_process4d_render_and_end_effector():
    from gem4d.process4d import Result4D
    from gem4d.process4d.end_effector import EndEffectorTracker, box_corners, draw_box, project
    from gem4d.process4d.render import orbit, render_points
    K = np.array([[50.0, 0, 32], [0, 50.0, 24], [0, 0, 1]])
    # z-buffer: the nearer of two points on the same ray wins
    img = render_points(np.array([[0, 0, 2.0], [0, 0, 1.0]]), np.array([[255, 0, 0], [0, 0, 255]], np.uint8),
                        K, np.eye(4), (48, 64), radius=2)
    assert tuple(img[24, 32]) == (0, 0, 255) and (img != 255).any(-1).sum() == 25
    # orbit keeps the distance to the pivot and still looks at it
    c = orbit(np.eye(4), np.array([0, 0, 2.0]), yaw_deg=30, pitch_deg=10)
    assert abs(np.linalg.norm(c[:3, 3] - [0, 0, 2]) - 2) < 1e-6
    uv, z = project(np.array([[0, 0, 2.0]]), K, c)
    assert np.allclose(uv[0], [32, 24], atol=1e-6) and z[0] > 0
    # tracker: a fake segmenter marks a 6x6 "gripper" moving right by 2 px per frame on a plane at depth 2
    f, h, w = 5, 48, 64
    depth = np.full((f, h, w), 2.0, np.float32)
    v, u = np.mgrid[0:h, 0:w] + 0.5
    cam = np.stack([(u - 32) / 50 * 2, (v - 24) / 50 * 2, np.full_like(u, 2.0)], -1).astype(np.float32)
    points = np.repeat(cam[None], f, 0)
    masks = np.zeros((f, h, w), bool)
    for t in range(f):
        masks[t, 20:26, 20 + 2 * t:26 + 2 * t] = True

    class FakeSeg:
        prompts, threshold, dilate = ["x"], 0.5, 0

        def __call__(self, images):
            return torch.from_numpy(masks)

    r = Result4D(images=np.zeros((f, h, w, 3), np.uint8), depth=depth, conf=np.ones_like(depth), K=np.stack([K] * f),
                 c2w=np.stack([np.eye(4)] * f), points=points, conf_ok=np.ones((f, h, w), bool),
                 flow=np.zeros((f - 1, h, w, 2), np.float32), sf=np.zeros((f - 1, h, w, 3), np.float32),
                 valid=np.ones((f - 1, h, w), bool), dynamic=np.zeros((f - 1, h, w), bool), gap=1,
                 threshold=np.zeros(f - 1), align=np.stack([np.eye(4)] * (f - 1)))
    ee = EndEffectorTracker(segmenter=FakeSeg(), smooth=0.0)(r)
    assert ee["found"].all() and ee["corners"].shape == (f, 8, 3)
    assert np.allclose(np.diff(ee["center"][:, 0]), 2 / 50 * 2, atol=1e-5)         # 2 px at depth 2
    uv, _ = project(ee["center"], K, np.eye(4))
    assert np.allclose(uv[:, 1], 23.0, atol=0.3)
    boxed = draw_box(np.zeros((h, w, 3), np.uint8), ee["corners"][0], K, np.eye(4))
    assert (boxed[..., 1] > 200).sum() > 20
    assert np.allclose(box_corners(np.zeros(3), np.ones(3)).max(0), 0.5)


def _dit_out(dit):
    torch.manual_seed(1)
    x, t, ctx = torch.randn(1, 16, F, H, W), torch.tensor([500.0]), torch.randn(1, 8, 32)
    cl, clip = torch.randn(1, 16, 1, H, W), torch.randn(1, 257, 1280)
    with torch.no_grad():
        return dit(x, t, ctx, clip_fea=clip, condition_latents=cl)


def test_lora_wan14b_inject_merge_roundtrip():
    from gem4d.models.lora import (WAN14B_TARGETS, inject_lora, load_lora, lora_modules, merge_lora,
                                   read_lora, save_lora)
    dit = tiny_dit().eval()
    ref = _dit_out(dit)
    names = inject_lora(dit, WAN14B_TARGETS, r=4, alpha=8)
    assert len(names) == 4 * 10                                       # self q k v o, cross q k v o, ffn.0, ffn.2
    assert not any("k_img" in n or "v_img" in n for n in names)
    trainable = [n for n, p in dit.named_parameters() if p.requires_grad]
    assert trainable and all(".lora_" in n for n in trainable)
    assert torch.allclose(_dit_out(dit), ref, atol=1e-6)             # B = 0: LoRA starts as a no-op
    with torch.no_grad():
        for m in lora_modules(dit).values():
            m.lora_B.normal_(std=0.05)
    tuned = _dit_out(dit)
    assert not torch.allclose(tuned, ref, atol=1e-4)
    with tempfile.TemporaryDirectory() as d:
        save_lora(dit, d, WAN14B_TARGETS)
        sd, cfg = read_lora(d)
        assert cfg["r"] == 4 and cfg["alpha"] == 8 and len(sd) == 2 * 40
        fresh = tiny_dit().eval()
        fresh.load_state_dict({k.replace(".base.", "."): v for k, v in dit.state_dict().items()
                               if ".lora_" not in k})
        assert load_lora(fresh, d) == 40                              # inject + load + merge
        assert not lora_modules(fresh)
        assert torch.allclose(_dit_out(fresh), tuned, atol=1e-5)
        # peft-style keys in a .pt payload load the same way
        peft = {f"base_model.model.{k.replace('.weight', '.default.weight')}": v for k, v in sd.items()}
        torch.save({"lora": peft, "lora_config": cfg}, os.path.join(d, "step.pt"))
        fresh2 = tiny_dit().eval()
        fresh2.load_state_dict({k.replace(".base.", "."): v for k, v in dit.state_dict().items()
                                if ".lora_" not in k})
        load_lora(fresh2, os.path.join(d, "step.pt"))
        assert torch.allclose(_dit_out(fresh2), tuned, atol=1e-5)
    assert merge_lora(dit) == 40 and torch.allclose(_dit_out(dit), tuned, atol=1e-5)


def test_lora_wan14b_training_step():
    from gem4d.models.lora import WAN14B_TARGETS, inject_lora
    bb = tiny_wan14b()
    inject_lora(bb.transformer, WAN14B_TARGETS, r=4, alpha=4)
    model = GEM4D(bb, geometry=False)
    with torch.no_grad():                                              # B != 0 so A gets gradient too
        for n, p in model.named_parameters():
            if n.endswith("lora_B"):
                p.normal_(std=0.02)
    x0 = torch.randn(B, F, 16, H, W)
    l_vid, l_geo = model(x0, condition(bb, x0), torch.randn(B, 8, 32), bb.sample_timesteps(B, "cpu"))
    l_vid.backward()
    grads = {n: p.grad for n, p in model.named_parameters() if p.requires_grad}
    assert l_geo == 0 and grads and all(".lora_" in n for n in grads)
    assert all(g is not None and g.abs().sum() > 0 for g in grads.values())


def test_flow_panel_arrows_only_on_moving_foreground():
    from gem4d.process4d.flow_vis import FlowPanel
    f, h, w = 4, 90, 120
    depth = np.tile(np.linspace(1, 3, h, dtype=np.float32)[:, None], (f, 1, w))
    flow = np.zeros((f - 1, h, w, 2), np.float32)
    flow[..., 0] = 0.4                                                 # camera shake everywhere ...
    flow[:, 30:60, 60:90, 0] += 6.0                                    # ... plus a moving arm
    fg = np.zeros((f, h, w), bool)
    fg[:, 30:60, 60:90] = True
    panel = FlowPanel(depth, flow, np.zeros((f - 1, h, w), bool), fg, np.full((f - 1, 2), [0.4, 0]), step=15)
    arrows = panel.arrows(1)
    assert arrows and all(60 <= x <= 90 and 30 <= y <= 60 for x, y, _ in arrows)
    assert all(abs(a[2][0] - 6.0) < 1e-4 and abs(a[2][1]) < 1e-4 for a in arrows)   # shake removed
    img = panel(1)
    assert img.shape == (h, w, 3)
    clean = FlowPanel(depth, flow, np.zeros((f - 1, h, w), bool), np.zeros((f, h, w), bool),
                      np.full((f - 1, 2), [0.4, 0]), step=15)
    assert not clean.arrows(1) and np.array_equal(clean(1)[:20], img[:20])   # background untouched


def test_robot_composite():
    from gem4d.utils.video_io import robot_composite
    img = np.full((480, 800, 3), 200, np.uint8)
    out = robot_composite(img)
    assert out.shape == (240, 416, 3)
    assert out[:, :295].min() > 190 and out[:, 310:].max() < 10            # main view left, black right


def test_lora_wan5b_targets():
    from gem4d.models.lora import WAN5B_TARGETS, inject_lora
    bb = tiny_wan5b()
    names = inject_lora(bb.transformer, WAN5B_TARGETS, r=4, alpha=4)
    assert len(names) == 4 * 10, names                                   # attn1 / attn2 q k v out, ffn x2
    l_vid, _ = run_step(bb, geometry=False)
    assert np.isfinite(l_vid)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
