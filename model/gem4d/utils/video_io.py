"""Small video helpers shared by validation and inference."""
from pathlib import Path
from typing import List

import cv2
import numpy as np
from PIL import Image


def center_crop_resize(img: np.ndarray, size) -> np.ndarray:
    th, tw = size
    h, w = img.shape[:2]
    if w / h > tw / th:
        nw = int(round(h * tw / th)); x0 = (w - nw) // 2
        img = img[:, x0:x0 + nw]
    else:
        nh = int(round(w * th / tw)); y0 = (h - nh) // 2
        img = img[y0:y0 + nh]
    return cv2.resize(img, (tw, th), interpolation=cv2.INTER_LINEAR)


def robot_composite(img: np.ndarray, size=(240, 416)) -> np.ndarray:
    """A single view -> the GEM-4D-Wan14B training composite: 4:3 center crop resized to 320 x 240 on the
    left of a black 440 x 240 canvas, resized to `size` (H, W) (main view = x < 302 of 416)."""
    canvas = np.zeros((240, 440, 3), np.uint8)
    canvas[:, :320] = center_crop_resize(img, (240, 320))
    return np.asarray(Image.fromarray(canvas).resize((size[1], size[0]), Image.BICUBIC))


def first_frame(path: str, size, compose: bool = False) -> Image.Image:
    """First frame of a video (or the image itself), center-cropped and resized to (H, W); with `compose`,
    placed into the robot composite instead (robot_composite)."""
    p = Path(path)
    if p.is_dir():
        img = np.array(Image.open(sorted(p.glob("*.png"), key=lambda x: int(x.stem))[0]).convert("RGB"))
    elif p.suffix.lower() in (".mp4", ".webm", ".avi", ".mov"):
        cap = cv2.VideoCapture(str(p))
        ok, f = cap.read()
        cap.release()
        if not ok:
            raise ValueError(f"cannot read {path}")
        img = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    else:
        img = np.array(Image.open(p).convert("RGB"))
    return Image.fromarray(robot_composite(img, size) if compose else center_crop_resize(img, size))


def side_by_side(gt: List[np.ndarray], pred: List[np.ndarray]) -> List[np.ndarray]:
    """[GT | prediction] frames; GT is resampled to the prediction length."""
    idx = np.linspace(0, len(gt) - 1, len(pred)).round().astype(int)
    return [np.concatenate([gt[i], np.asarray(p)], axis=1) for i, p in zip(idx, pred)]


def to_uint8(frames) -> List[np.ndarray]:
    out = []
    for f in frames:
        a = np.asarray(f)
        if a.dtype != np.uint8:
            a = (np.clip(a, 0, 1) * 255).round().astype(np.uint8)
        out.append(a)
    return out


def save_mp4(frames, path: str, fps: int = 8):
    """H.264 mp4 from uint8 frames (or floats in [0, 1]); written as uint8, no float round trip."""
    import imageio
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(path, to_uint8(frames), fps=fps, codec="libx264", quality=8, macro_block_size=16)
