"""Teaser figure (left part) from one generated video: generated RGB frames, the 3D point cloud and a green
3D box tracking the end-effector ("inverse dynamics" panel).

    # 1) generate from the initial image
    python inference/infer.py --backbone wan2.1-i2v-14b --base_model weights/Wan2.1-I2V-14B-480P \
        --weights weights/gem-4d-wan14b --image examples/teaser_initial.png \
        --prompt "The Franka robot arm performs the task: touch the mouse. ..." --output_dir outputs/teaser
    # 2) figure + assets
    python inference/teaser.py --video outputs/teaser/0000_teaser_initial.mp4 --initial examples/teaser_initial.png \
        --crop 0,0,302,240 --da3 weights/da3 --output_dir outputs/teaser/figure

Writes initial.png, frame_<t>.png, ee_<t>.png (frame + projected box), pointcloud_<t>[_view<k>].png,
teaser_left.png (the assembled panel), ee_track.mp4 / ee_track.json, and the 4D-process outputs
(teaser_4d.html viewer, teaser_4d.mp4, teaser_4d.json).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "model"), str(ROOT)]

from PIL import Image                                                                 # noqa: E402

from gem4d.process4d import Process4D, crop_frames, run_4d_process                   # noqa: E402
from gem4d.process4d.end_effector import EndEffectorTracker, draw_box, scale_K      # noqa: E402
from gem4d.process4d.render import orbit, render_points, titled                      # noqa: E402
from gem4d.utils.video_io import save_mp4                                            # noqa: E402
from inference.process4d import add_args, parse_crop                                 # noqa: E402
from training.data.video_dataset import read_video                                   # noqa: E402


def resize(img: np.ndarray, hw) -> np.ndarray:
    return np.asarray(Image.fromarray(img).resize((hw[1], hw[0]), Image.BICUBIC))


def point_cloud_view(r, ee, t: int, hw, yaw: float, pitch: float, back: float, radius: int = 1) -> np.ndarray:
    """Point cloud of frame t (DA3 points above the confidence cut, RGB colours) seen from an orbited camera,
    with the end-effector box."""
    pts, col = r.points[t][r.conf_ok[t]], r.images[t][r.conf_ok[t]]
    pivot = np.median(pts, 0)
    depth = float(np.median(r.depth[t]))
    view = orbit(r.c2w[t], pivot, yaw_deg=yaw, pitch_deg=pitch, back=back * depth)
    K = scale_K(r.K[t], r.depth.shape[1:], hw)
    img = render_points(pts, col, K, view, hw, radius=radius)
    if ee["found"].any():
        img = draw_box(img, ee["corners"][t], K, view, width=3)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="generated video")
    ap.add_argument("--initial", default=None, help="the condition image (default: first frame of --video)")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--frames", default="12,22,32", help="frames shown as generated RGB / inverse dynamics")
    ap.add_argument("--cloud_frame", type=int, default=-1, help="frame of the large point cloud (default: last of --frames)")
    ap.add_argument("--views", default="-10,6,0.05;0,0,0;12,6,0.05", help="yaw,pitch,back;... point-cloud views")
    ap.add_argument("--figure_view", type=int, default=0, help="which of --views goes into teaser_left.png")
    ap.add_argument("--fps", type=int, default=8)
    add_args(ap)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    out = lambda name: os.path.join(args.output_dir, name)                          # noqa: E731

    frames = crop_frames(read_video(Path(args.video)), parse_crop(args.crop))
    proc = Process4D(da3=args.da3, gap=args.gap, process_res=args.process_res, seg=args.seg,
                     min_flow_px=args.min_flow_px)
    r = run_4d_process(proc, frames, args.output_dir, "teaser", fps=args.fps)["result"]
    tracker = EndEffectorTracker(segmenter=proc.segmenter, seg_weights=args.seg)
    ee = tracker(r)
    f, (h, w) = r.depth.shape[0], r.depth.shape[1:]
    print(f"end-effector found in {int(ee['found'].sum())}/{f} frames", flush=True)

    shown = [min(int(t), f - 1) for t in args.frames.split(",")]
    big = (2 * frames.shape[1], 2 * frames.shape[2])                                # panels at 2x the main view
    init = np.asarray(Image.open(args.initial).convert("RGB")) if args.initial else frames[0]
    init = resize(init, big)
    Image.fromarray(init).save(out("initial.png"))
    gen, ids = [], []
    for t in shown:
        g = resize(frames[t], big)
        Image.fromarray(g).save(out(f"frame_{t:02d}.png"))
        K = scale_K(r.K[t], (h, w), big)
        e = (draw_box(resize(r.images[t], big), ee["corners"][t], K, r.c2w[t], width=4)
             if ee["found"].any() else g)
        Image.fromarray(e).save(out(f"ee_{t:02d}.png"))
        gen.append(g)
        ids.append(e)
    track = []
    for t in range(f):
        g = resize(r.images[t], big)                                          # DA3 frame: K matches exactly
        track.append(draw_box(g, ee["corners"][t], scale_K(r.K[t], (h, w), big), r.c2w[t], width=4)
                     if ee["found"].any() else g)
    save_mp4(np.stack(track), out("ee_track.mp4"), fps=args.fps)
    with open(out("ee_track.json"), "w") as fh:
        json.dump(dict(found=ee["found"].tolist(), center=ee["center"].round(5).tolist(),
                       size=ee["size"][0].round(5).tolist(), units="DA3 scene units, world = DA3 frame-0 camera"),
                  fh, indent=1)

    tc = (args.cloud_frame if args.cloud_frame >= 0 else shown[-1]) % f
    clouds = []
    for k, v in enumerate(args.views.split(";")):
        yaw, pitch, back = (float(x) for x in v.split(","))
        img = resize(point_cloud_view(r, ee, tc, (480, 640), yaw, pitch, back, radius=1), (720, 960))
        Image.fromarray(img).save(out(f"pointcloud_{tc:02d}_view{k}.png"))
        clouds.append(img)

    # assembled left part of the teaser: [Initial | Generated RGB Frames] over [Inverse Dynamics | 3D Point Cloud]
    gap = 12
    top_l = titled(init, "Initial", 26)
    top_r = titled(np.concatenate(sum([[g, np.full((g.shape[0], gap // 2, 3), 255, np.uint8)] for g in gen], [])[:-1], 1),
                   "Generated RGB Frames", 26)
    col = np.concatenate(sum([[e, np.full((gap // 2, e.shape[1], 3), 255, np.uint8)] for e in ids], [])[:-1], 0)
    bot_l = titled(col, "Inverse Dynamics", 26)
    cloud = resize(clouds[args.figure_view], (bot_l.shape[0] - 38, int((bot_l.shape[0] - 38) * 4 / 3)))
    bot_r = titled(cloud, "Generated 3D Point Cloud", 26)
    width = max(top_l.shape[1] + gap + top_r.shape[1], bot_l.shape[1] + gap + bot_r.shape[1])
    canvas = np.full((top_l.shape[0] + gap + bot_l.shape[0], width, 3), 255, np.uint8)
    canvas[:top_l.shape[0], :top_l.shape[1]] = top_l
    canvas[:top_r.shape[0], top_l.shape[1] + gap:top_l.shape[1] + gap + top_r.shape[1]] = top_r
    y = top_l.shape[0] + gap
    canvas[y:y + bot_l.shape[0], :bot_l.shape[1]] = bot_l
    canvas[y:y + bot_r.shape[0], bot_l.shape[1] + gap:bot_l.shape[1] + gap + bot_r.shape[1]] = bot_r
    Image.fromarray(canvas).save(out("teaser_left.png"))
    print("teaser ->", out("teaser_left.png"), flush=True)


if __name__ == "__main__":
    main()
