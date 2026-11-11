"""End-effector tracking on a 4D-processed video: a 3D box around the gripper in every frame.

The gripper is segmented with CLIPSeg ("a robot gripper"), lifted to world points with DA3's depth and
cameras (Result4D.points), and boxed by the robust (10th-90th percentile) extent of those points along
the world axes. The mask is eroded first (depth at object borders mixes foreground and background) and cut to
its lowest `tip_frac` of rows (CLIPSeg's "gripper" region also covers the wrist; in tabletop views the
fingers are the lowest part), and the box is at most as deep as it is wide or high. Centre and size are smoothed over time; the box is drawn back into each frame by projecting
its 8 corners with that frame's camera, and into point-cloud renders (render.py).
"""
from typing import Optional

import numpy as np
import torch

from .scene_flow import ForegroundSegmenter, Result4D

EDGES = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def largest_component(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    lab, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return lab == (1 + int(np.argmax(sizes)))


def box_corners(center: np.ndarray, size: np.ndarray) -> np.ndarray:
    """Axis-aligned box -> [8, 3] corners; corner i has bit k set -> +size/2 along axis k."""
    signs = np.array([[(i >> k) & 1 for k in range(3)] for i in range(8)], np.float32) * 2 - 1
    return center + signs * size / 2


def project(points: np.ndarray, K: np.ndarray, c2w: np.ndarray):
    """world [N, 3] -> pixels [N, 2], depth [N]."""
    w2c = np.linalg.inv(c2w)
    cam = points @ w2c[:3, :3].T + w2c[:3, 3]
    uv = cam @ K.T
    return uv[:, :2] / np.clip(uv[:, 2:], 1e-6, None), cam[:, 2]


class EndEffectorTracker:
    def __init__(self, segmenter: Optional[ForegroundSegmenter] = None, seg_weights: str = "CIDAS/clipseg-rd64-refined",
                 device="cuda", prompts=("a robot gripper",), threshold: float = 0.4, smooth: float = 0.6,
                 min_pixels: int = 30, erode: int = 2, pct: float = 10.0, tip_frac: float = 0.45):
        self.seg = segmenter or ForegroundSegmenter(seg_weights, device=device, dilate=0)
        self.prompts, self.threshold, self.smooth, self.min_pixels = list(prompts), threshold, smooth, min_pixels
        self.erode, self.pct, self.tip_frac = erode, pct, tip_frac

    def masks(self, images: np.ndarray) -> np.ndarray:
        prompts, thr, dil = self.seg.prompts, self.seg.threshold, self.seg.dilate
        self.seg.prompts, self.seg.threshold, self.seg.dilate = self.prompts, self.threshold, 0
        try:
            m = self.seg(images).cpu().numpy()
        finally:
            self.seg.prompts, self.seg.threshold, self.seg.dilate = prompts, thr, dil
        out = np.stack([largest_component(x) for x in m])
        if self.erode:
            from scipy import ndimage

            eroded = np.stack([ndimage.binary_erosion(x, iterations=self.erode) for x in out])
            out = np.where(eroded.reshape(len(out), -1).sum(1)[:, None, None] >= self.min_pixels, eroded, out)
        if self.tip_frac < 1:
            for t in range(len(out)):
                rows = np.flatnonzero(out[t].any(1))
                if rows.size:
                    cut = rows[-1] - self.tip_frac * (rows[-1] - rows[0] + 1)
                    tip = out[t].copy()
                    tip[:int(np.ceil(cut))] = False
                    if tip.sum() >= self.min_pixels:
                        out[t] = tip
        return out

    @torch.no_grad()
    def __call__(self, r: Result4D) -> dict:
        """-> dict(mask [F, H, W], center [F, 3], size [F, 3], corners [F, 8, 3], found [F] bool)."""
        mask = self.masks(r.images)
        f = mask.shape[0]
        center, size, found = np.zeros((f, 3), np.float32), np.zeros((f, 3), np.float32), np.zeros(f, bool)
        for t in range(f):
            m = mask[t] & r.conf_ok[t] if (mask[t] & r.conf_ok[t]).sum() >= self.min_pixels else mask[t]
            if m.sum() < self.min_pixels:
                continue
            p = r.points[t][m]
            lo, hi = np.percentile(p, self.pct, axis=0), np.percentile(p, 100 - self.pct, axis=0)
            c, s = (lo + hi) / 2, hi - lo
            # depth axis = the world axis closest to the camera's viewing direction; cap it
            k = int(np.argmax(np.abs(r.c2w[t][:3, 2])))
            s[k] = min(s[k], max(np.delete(s, k)))
            c[k] = np.median(p[:, k])
            center[t], size[t], found[t] = c, s, True
        if not found.any():
            return dict(mask=mask, center=center, size=size, corners=np.zeros((f, 8, 3), np.float32), found=found)
        # fill gaps with the nearest detection, then smooth (forward-backward exponential average)
        idx = np.flatnonzero(found)
        near = idx[np.abs(np.arange(f)[:, None] - idx[None]).argmin(1)]
        center, size = center[near], size[near]
        size = np.repeat(np.median(size[found], 0, keepdims=True), f, 0)      # rigid gripper: one size
        for arr in (center,):
            a = self.smooth
            for t in range(1, f):
                arr[t] = a * arr[t - 1] + (1 - a) * arr[t]
            for t in range(f - 2, -1, -1):
                arr[t] = a * arr[t + 1] + (1 - a) * arr[t]
        corners = np.stack([box_corners(center[t], size[t]) for t in range(f)])
        return dict(mask=mask, center=center, size=size, corners=corners, found=found)


def draw_box(img: np.ndarray, corners: np.ndarray, K: np.ndarray, c2w: np.ndarray, color=(40, 230, 70),
             width: int = 2) -> np.ndarray:
    """Draw the projected 3D box (12 edges) onto img [H, W, 3] uint8 (K for img's resolution)."""
    from PIL import Image, ImageDraw

    uv, z = project(corners, K, c2w)
    if (z <= 0).any():
        return img
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    for a, b in EDGES:
        d.line([tuple(uv[a]), tuple(uv[b])], fill=color, width=width)
    return np.asarray(im)


def scale_K(K: np.ndarray, src_hw, dst_hw) -> np.ndarray:
    K = K.copy()
    K[0] *= dst_hw[1] / src_hw[1]
    K[1] *= dst_hw[0] / src_hw[0]
    return K
