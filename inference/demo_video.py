"""Demo video: one row per example, columns [input image | generated video | depth & scene flow].

    python inference/demo_video.py --output outputs/demo.mp4 --crop 0,0,302,240 \
        --row "DROID|first_frame.png|outputs/0002_droid_00.mp4|outputs/0002_droid_00_4d.npz" \
        --row "Bridge|..." ...

Each --row is "label|input image|generated video|4D-process npz" (npz from inference/process4d.py or
infer.py --process4d). The depth & scene flow panel is gem4d.process4d.flow_vis.FlowPanel: DA3 depth
(turbo) with white arrows for the motion of the moving foreground to frame t+gap; the background has no arrows.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "model"), str(ROOT)]

from PIL import Image, ImageDraw                                                      # noqa: E402

from gem4d.process4d import crop_frames                                               # noqa: E402
from gem4d.process4d.flow_vis import FlowPanel, resize                               # noqa: E402
from gem4d.process4d.render import font                                               # noqa: E402
from gem4d.utils.video_io import save_mp4                                             # noqa: E402
from inference.process4d import parse_crop                                            # noqa: E402
from training.data.video_dataset import read_video                                    # noqa: E402


def text_box(text: str, hw, size: int, bg=(255, 255, 255), fg=(20, 20, 20), vertical=False) -> np.ndarray:
    h, w = hw
    f = font(size)
    if vertical:
        return np.asarray(Image.fromarray(text_box(text, (w, h), size, bg, fg)).rotate(90, expand=True))
    im = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(im)
    lines = text.split("\n")
    total = len(lines) * (size + 4)
    for i, line in enumerate(lines):
        tw = d.textlength(line, font=f)
        d.text(((w - tw) / 2, (h - total) / 2 + i * (size + 4)), line, fill=fg, font=f)
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--row", action="append", required=True, help="label|input image|generated video|4D npz")
    ap.add_argument("--output", required=True)
    ap.add_argument("--crop", default=None, help="x0,y0,x1,y1 of the main view in input / generated frames")
    ap.add_argument("--scale", type=float, default=1.5, help="panel size relative to the cropped main view")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--title", default="GEM-4D: image + instruction → video → depth & scene flow")
    args = ap.parse_args()
    crop = parse_crop(args.crop)

    rows = []
    for spec in args.row:
        label, image, video, npz = spec.split("|")
        frames = crop_frames(read_video(Path(video)), crop)
        hw = (int(round(frames.shape[1] * args.scale)) // 2 * 2, int(round(frames.shape[2] * args.scale)) // 2 * 2)
        inp = np.asarray(Image.open(image).convert("RGB"))
        if crop and inp.shape[1] > crop[2] - crop[0] and inp.shape[0] >= crop[3]:
            inp = crop_frames(inp[None], crop)[0]
        rows.append(dict(label=label.replace("\\n", "\n"), input=resize(inp, hw), frames=frames,
                         flow=FlowPanel.from_npz(npz, hw)))
    hw = rows[0]["input"].shape[:2]
    n = min(min(r["flow"].n, len(r["frames"])) for r in rows)
    k = hw[0] / 360                                                               # text and margins follow panel size
    gap, lab_w, head_h, title_h = (int(round(v * k)) for v in (8, 70, 44, 52))
    width = lab_w + 3 * hw[1] + 4 * gap
    white = lambda hh, ww: np.full((hh, ww, 3), 255, np.uint8)                    # noqa: E731
    title = text_box(args.title, (title_h, width), int(round(26 * k)))
    heads = [white(head_h, lab_w + gap)]
    for name in ("Input image", "Generated video", "Depth & scene flow"):
        heads += [text_box(name, (head_h, hw[1]), int(round(22 * k))), white(head_h, gap)]
    head = np.concatenate(heads, 1)
    labels = [text_box(r["label"], (hw[0], lab_w), int(round(22 * k)), vertical=True) for r in rows]
    out = []
    for t in range(n):
        parts = [title, head]
        for r, lab in zip(rows, labels):
            parts.append(np.concatenate([lab, white(hw[0], gap), r["input"], white(hw[0], gap),
                                         resize(r["frames"][t], hw), white(hw[0], gap), r["flow"](t),
                                         white(hw[0], gap)], 1))
            parts.append(white(gap, width))
        out.append(np.concatenate(parts, 0))
    frame = np.stack(out)
    ph, pw = -frame.shape[1] % 16, -frame.shape[2] % 16                           # codec-friendly size
    frame = np.pad(frame, ((0, 0), (0, ph), (pw // 2, pw - pw // 2), (0, 0)), constant_values=255)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    save_mp4(frame, args.output, fps=args.fps)
    Image.fromarray(frame[n // 2]).save(os.path.splitext(args.output)[0] + "_mid.png")
    print(f"demo video -> {args.output} ({n} frames, {frame.shape[2]}x{frame.shape[1]})", flush=True)


if __name__ == "__main__":
    main()
