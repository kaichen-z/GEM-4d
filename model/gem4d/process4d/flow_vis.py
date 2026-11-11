"""Clean depth & scene-flow panel: DA3 depth (turbo) with white arrows on the moving foreground only.

The moving foreground is the dynamic points plus CLIPSeg-foreground pixels whose optical flow (camera
shake removed) is >= `min_flow_px`; the background gets no arrows. Flow is averaged over a +-1 frame
window (no flicker) and over each `step`-pixel grid cell (one arrow per cell, centred on it). Arrow
length = cell flow / the video's 95th-percentile flow, times `max_len` cells, capped (no overlaps) and at
least `min_len` cells (slow motion stays readable).
Arrows are drawn at `ss`x and downsampled (anti-aliased). Used for `<clip>_4d.mp4` and the demo video.
"""
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw

from .viewer import depth_to_rgb


def resize(img: np.ndarray, hw, nearest: bool = False) -> np.ndarray:
    return np.asarray(Image.fromarray(img).resize((hw[1], hw[0]), Image.NEAREST if nearest else Image.BICUBIC))


def draw_arrow(d, p0, p1, scale: float, fill=(255, 255, 255), outline=(25, 25, 25)):
    """Filled arrow (tapered shaft + triangular head) from p0 to p1 on an ImageDraw; widths in pixels * scale."""
    v = np.subtract(p1, p0, dtype=np.float64)
    length = float(np.hypot(*v))
    if length < 1e-6:
        return
    u, n = v / length, np.array([-v[1], v[0]]) / length
    head = min(0.4 * length, 8.0 * scale)
    hw, w0, w1 = 0.6 * head, 0.8 * scale, 1.6 * scale                  # head half-width, shaft half-widths
    p0, p1 = np.asarray(p0, np.float64), np.asarray(p1, np.float64)
    base = p1 - u * head
    poly = [tuple(q) for q in (p0 + n * w0, base + n * w1, base + n * hw, p1, base - n * hw, base - n * w1,
                               p0 - n * w0)]
    d.polygon(poly, fill=outline, outline=outline, width=int(round(1.6 * scale)))        # dark rim
    d.polygon(poly, fill=fill)


class FlowPanel:
    """depth [F, H, W], flow [F', H, W, 2] (t -> t+gap), dynamic [F', H, W]; foreground [F, H, W] and
    shake [F', 2] optional. `panel(t)` -> [h, w, 3] uint8 at output size `hw` (default: depth size)."""

    def __init__(self, depth, flow, dynamic, foreground=None, shake=None, hw=None, step: int = 30,
                 max_len: float = 1.25, min_len: float = 0.4, min_flow_px: float = 1.5, min_arrow_px: float = 1.5,
                 min_cell: float = 0.5, temporal: int = 1, ss: int = 4):
        self.depth = np.asarray(depth, np.float32)
        flow = np.asarray(flow, np.float32)
        if shake is not None:
            flow = flow - np.asarray(shake, np.float32)[:, None, None]
        n = flow.shape[0]
        moving = np.linalg.norm(flow, axis=-1) >= min_flow_px
        fg = np.asarray(foreground)[:n] if foreground is not None else np.zeros_like(moving)
        self.dyn = np.asarray(dynamic)[:n] | (fg & moving)
        # temporal smoothing over the moving pixels only
        wsum = np.zeros(flow.shape[:3], np.float32)
        fsum = np.zeros_like(flow)
        for k in range(-temporal, temporal + 1):
            idx = np.clip(np.arange(n) + k, 0, n - 1)
            m = self.dyn[idx].astype(np.float32)
            wsum += m
            fsum += flow[idx] * m[..., None]
        self.flow = np.where(wsum[..., None] > 0, fsum / np.maximum(wsum, 1)[..., None], flow)
        self.lo, self.hi = np.percentile(self.depth, [2, 98])
        mags = np.linalg.norm(self.flow, axis=-1)[self.dyn]
        self.ref = max(float(np.percentile(mags, 95)) if mags.size else 1.0, 1e-6)
        self.hw = tuple(hw) if hw is not None else self.depth.shape[1:]
        self.step, self.max_len, self.min_len = step, max_len, min_len
        self.min_arrow_px, self.min_cell, self.ss = min_arrow_px, min_cell, ss
        self.n = n

    @classmethod
    def from_npz(cls, path: str, hw=None, **kw) -> "FlowPanel":
        z = np.load(path)
        get = lambda k: z[k] if k in z else None                              # noqa: E731
        return cls(z["depth"], z["flow"], z["dynamic"], get("foreground"), get("shake"), hw=hw, **kw)

    @classmethod
    def from_result(cls, r, hw=None, **kw) -> "FlowPanel":
        return cls(r.depth, r.flow, r.dynamic, r.foreground, r.shake, hw=hw, **kw)

    def arrows(self, t: int):
        """[(x, y, flow)] per grid cell with enough moving pixels, in source pixels."""
        sh, sw = self.depth.shape[1:]
        s, out = self.step, []
        for y0 in range(0, sh, s):
            for x0 in range(0, sw, s):
                m = self.dyn[t, y0:y0 + s, x0:x0 + s]
                if m.mean() < self.min_cell:
                    continue
                f = self.flow[t, y0:y0 + s, x0:x0 + s][m].mean(0)
                if np.hypot(*f) >= self.min_arrow_px:
                    out.append((x0 + s / 2, y0 + s / 2, f))
        return out

    def __call__(self, t: int, background: Optional[np.ndarray] = None) -> np.ndarray:
        """Panel of frame t; `background` replaces the depth image (e.g. the RGB frame)."""
        h, w = self.hw
        sh, sw = self.depth.shape[1:]
        ss = self.ss
        base = depth_to_rgb(self.depth[t], self.lo, self.hi) if background is None else background
        img = Image.fromarray(resize(base, (h * ss, w * ss)))
        d = ImageDraw.Draw(img)
        sx, sy = w * ss / sw, h * ss / sh
        for x, y, f in sorted(self.arrows(t), key=lambda a: np.hypot(*a[2])):    # long arrows on top
            mag = float(np.hypot(*f))
            length = max(min(mag / self.ref, 1.0) * self.max_len, self.min_len) * self.step   # source pixels
            u = f / mag
            p0 = np.array([x * sx, y * sy]) - 0.5 * length * u * (sx, sy)      # centred on the cell
            p1 = p0 + length * u * (sx, sy)
            draw_arrow(d, tuple(p0), tuple(p1), scale=ss * w / 450)
        return np.asarray(img.resize((w, h), Image.LANCZOS))
