"""4D-process: scene flow of a (generated) video from DepthAnything-3 and RAFT-small optical flow.

For every frame t, DA3 gives depth D_t, confidence, intrinsics K_t and a world-to-camera pose; RAFT
gives the optical flow t -> t+k and back. A pixel p of frame t is lifted to the world point
X_t(p) = c2w_t (D_t(p) K_t^-1 [p, 1]); its match q = p + flow_{t->t+k}(p) is lifted the same way
in frame t+k, and the scene flow is

    S_t(p) = A_t(X_{t+k}(q)) - X_t(p)          (world frame, DA3 units)

Camera motion cancels in world coordinates, so static surfaces have S ~ 0 and moving ones do not.
DA3's per-frame depth scale and pose drift slightly from frame to frame, which would make the whole
static scene "move"; A_t is a similarity transform (scale, rotation, translation) fitted robustly to
all valid matches of the pair, so the static majority defines the reference and the drift cancels.
Matches failing the forward-backward check, leaving the image or with low DA3 confidence are
invalid. A point is dynamic only if all of these hold:
  - |S| exceeds the pair's robust noise level (and 2 % of the median depth);
  - it lies on the foreground (robot arm, gripper, hand, held object: CLIPSeg text-prompted masks),
    so the background never carries scene flow;
  - its optical flow differs from the background's median flow (the global camera shake of the pair)
    by at least `min_flow_px` pixels, so small jitter is not reported as motion.
Outside the foreground the scene flow is set to zero.
"""
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

_DEPENDENCY = Path(__file__).resolve().parents[2] / "dependency"


def homogeneous(m: np.ndarray) -> np.ndarray:
    """[..., 3, 4] or [..., 4, 4] -> [..., 4, 4]."""
    if m.shape[-2] == 4:
        return m
    out = np.zeros(m.shape[:-2] + (4, 4), dtype=m.dtype)
    out[..., :3, :] = m
    out[..., 3, 3] = 1
    return out


def unproject(depth: torch.Tensor, K: torch.Tensor, c2w: torch.Tensor) -> torch.Tensor:
    """depth [N, H, W], K [N, 3, 3], c2w [N, 4, 4] -> world points [N, H, W, 3] (OpenCV camera)."""
    n, h, w = depth.shape
    v, u = torch.meshgrid(torch.arange(h, device=depth.device, dtype=depth.dtype) + 0.5,
                          torch.arange(w, device=depth.device, dtype=depth.dtype) + 0.5, indexing="ij")
    pix = torch.stack([u, v, torch.ones_like(u)], -1).reshape(1, -1, 3)              # [1, HW, 3]
    rays = pix @ torch.linalg.inv(K).transpose(1, 2)                                # [N, HW, 3]
    cam = rays * depth.reshape(n, -1, 1)
    world = cam @ c2w[:, :3, :3].transpose(1, 2) + c2w[:, None, :3, 3]
    return world.reshape(n, h, w, 3)


def sample(img: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """img [N, C, H, W], coords [N, H, W, 2] pixel (x, y) -> [N, C, H, W] bilinear (zeros outside)."""
    h, w = img.shape[-2:]
    grid = torch.stack([coords[..., 0] / w * 2 - 1, coords[..., 1] / h * 2 - 1], -1)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def umeyama(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Similarity [4, 4] minimising |dst - (s R src + t)|^2 for src, dst [N, 3]."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    U, S, Vt = torch.linalg.svd(dc.T @ sc / src.shape[0])
    d = torch.ones(3, dtype=src.dtype, device=src.device)
    d[2] = torch.sign(torch.det(U) * torch.det(Vt))
    R = U @ torch.diag(d) @ Vt
    s = (S * d).sum() / sc.pow(2).sum(1).mean().clamp_min(1e-12)
    T = torch.eye(4, dtype=src.dtype, device=src.device)
    T[:3, :3] = s * R
    T[:3, 3] = mu_d - s * R @ mu_s
    return T


def robust_similarity(src: torch.Tensor, dst: torch.Tensor, iters: int = 4, k: float = 2.5,
                      max_points: int = 20000) -> torch.Tensor:
    """Umeyama refitted on the matches whose residual is below k x the median (static majority)."""
    if src.shape[0] > max_points:
        idx = torch.linspace(0, src.shape[0] - 1, max_points, device=src.device).long()      # deterministic
        src, dst = src[idx], dst[idx]
    keep = torch.ones(src.shape[0], dtype=torch.bool, device=src.device)
    T = torch.eye(4, dtype=src.dtype, device=src.device)
    for _ in range(iters):
        if keep.sum() < 10:
            break
        T = umeyama(src[keep], dst[keep])
        res = (src @ T[:3, :3].T + T[:3, 3] - dst).norm(dim=-1)
        keep = res <= k * res.median().clamp_min(1e-12)
    return T


def scene_flow(points: torch.Tensor, conf_ok: torch.Tensor, flow_fw: torch.Tensor, flow_bw: torch.Tensor,
               gap: int, fb_abs: float = 1.0, fb_rel: float = 0.05, dyn_k: float = 4.0, dyn_rel: float = 0.02,
               depth: Optional[torch.Tensor] = None, align: bool = True,
               foreground: Optional[torch.Tensor] = None, min_flow_px: float = 1.5):
    """Scene flow between frames t and t+gap.

    points  [F, H, W, 3] world points;   conf_ok [F, H, W] bool
    flow_fw [F-gap, 2, H, W] flow t -> t+gap;   flow_bw [F-gap, 2, H, W] flow t+gap -> t (pixels)
    depth   [F, H, W] (for the scale of the dynamic threshold; default: distance to the median point)
    align   fit a robust similarity per pair that maps frame t+gap's matched points onto frame t's
    foreground [F, H, W] bool: only these pixels may be dynamic (None: every pixel)
    min_flow_px  minimum 2D flow after removing the background's median flow (camera shake)
    With a foreground, the alignment, noise level and camera shake are estimated on the background only,
    and foreground pixels skip the confidence cut (the forward-backward check still applies).
    returns sf [F-gap, H, W, 3], valid [F-gap, H, W], dynamic [F-gap, H, W],
            threshold [F-gap] (per pair), A [F-gap, 4, 4] (alignment applied to frame t+gap),
            shake [F-gap, 2] (background median optical flow per pair, pixels)
    """
    n = flow_fw.shape[0]
    h, w = points.shape[1:3]
    v, u = torch.meshgrid(torch.arange(h, device=points.device, dtype=points.dtype) + 0.5,
                          torch.arange(w, device=points.device, dtype=points.dtype) + 0.5, indexing="ij")
    pix = torch.stack([u, v], -1)[None]                                             # [1, H, W, 2]
    fw = flow_fw.permute(0, 2, 3, 1)                                                # [n, H, W, 2]
    q = pix + fw
    inside = (q[..., 0] >= 0) & (q[..., 0] <= w) & (q[..., 1] >= 0) & (q[..., 1] <= h)
    bw_at_q = sample(flow_bw, q).permute(0, 2, 3, 1)
    fb_err = (fw + bw_at_q).norm(dim=-1)
    fb_ok = fb_err < fb_abs + fb_rel * fw.norm(dim=-1)
    target = sample(points[gap:gap + n].permute(0, 3, 1, 2), q).permute(0, 2, 3, 1)
    fg = foreground[:n] if foreground is not None else torch.ones_like(fb_ok)
    if foreground is not None:
        # DA3 is least confident on fast, thin or dark moving parts: the foreground skips the confidence cut
        fg_q = sample(foreground[gap:gap + n, None].float(), q)[:, 0] > 0.5
        conf_t, conf_q = conf_ok[:n] | fg, (sample(conf_ok[gap:gap + n, None].float(), q)[:, 0] > 0.99) | fg_q
    else:
        conf_t, conf_q = conf_ok[:n], sample(conf_ok[gap:gap + n, None].float(), q)[:, 0] > 0.99
    valid = inside & fb_ok & conf_t & conf_q
    # the static reference (alignment, noise level, camera shake) comes from the background only
    ref = valid & ~fg if foreground is not None else valid
    ref = torch.stack([ref[i] if ref[i].sum() >= 10 else valid[i] for i in range(n)])
    A = torch.eye(4, dtype=points.dtype, device=points.device).repeat(n, 1, 1)
    if align:
        for i in range(n):
            if ref[i].sum() >= 10:
                A[i] = robust_similarity(target[i][ref[i]], points[i][ref[i]])
        target = target @ A[:, None, :3, :3].transpose(-1, -2) + A[:, None, None, :3, 3]
    sf = torch.where(valid[..., None], target - points[:n], torch.zeros_like(target))
    mag = sf.norm(dim=-1)
    if depth is None:
        depth = (points - points.flatten(0, 2).median(0).values).norm(dim=-1)
    scale = depth[:n][valid].median() if valid.any() else depth.new_tensor(1.0)
    noise = torch.stack([mag[i][ref[i]].median() if ref[i].any() else mag.new_tensor(0.0) for i in range(n)])
    thr = torch.maximum(dyn_k * noise, dyn_rel * scale)                             # [n], per pair
    shake = torch.stack([fw[i][ref[i]].median(0).values if ref[i].any() else fw.new_zeros(2) for i in range(n)])
    moving_2d = (fw - shake[:, None, None]).norm(dim=-1) > min_flow_px
    dyn = valid & fg & moving_2d & (mag > thr[:, None, None])
    # drop isolated pixels: keep a dynamic pixel only if most of its 3x3 neighbourhood is dynamic too
    dyn = (F.avg_pool2d(dyn[:, None].float(), 3, 1, 1)[:, 0] > 0.5) & dyn
    return sf, valid, dyn, thr, A, shake


@dataclass
class Result4D:
    images: np.ndarray        # [F, H, W, 3] uint8, at DA3's processing resolution
    depth: np.ndarray         # [F, H, W] float32
    conf: np.ndarray          # [F, H, W] float32
    K: np.ndarray             # [F, 3, 3]
    c2w: np.ndarray           # [F, 4, 4]  OpenCV camera -> world
    points: np.ndarray        # [F, H, W, 3] world
    conf_ok: np.ndarray       # [F, H, W] bool
    flow: np.ndarray          # [F-gap, H, W, 2]  optical flow t -> t+gap (pixels)
    sf: np.ndarray            # [F-gap, H, W, 3]  scene flow t -> t+gap (world)
    valid: np.ndarray         # [F-gap, H, W] bool
    dynamic: np.ndarray       # [F-gap, H, W] bool
    gap: int
    threshold: np.ndarray     # [F-gap] dynamic threshold per pair
    align: np.ndarray         # [F-gap, 4, 4] similarity mapping frame t+gap's points onto frame t's
    foreground: Optional[np.ndarray] = None   # [F, H, W] bool, CLIPSeg mask (None: not segmented)
    shake: Optional[np.ndarray] = None        # [F-gap, 2] background median optical flow (pixels)
    background_residual: float = 0.0          # median |S| of valid background pixels before zeroing

    def _static_median(self, mag):
        m = self.valid & ~self.dynamic
        if self.foreground is not None:
            m &= self.foreground[:m.shape[0]]                                       # background is zeroed
        return float(np.median(mag[m])) if m.any() else 0.0

    def stats(self):
        mag = np.linalg.norm(self.sf, axis=-1)
        dyn = self.dynamic
        return dict(frames=int(self.depth.shape[0]), resolution=list(self.depth.shape[1:]), gap=self.gap,
                    valid_fraction=float(self.valid.mean()), dynamic_fraction=float(dyn.mean()),
                    dynamic_threshold=float(np.mean(self.threshold)),
                    align_scale_dev=float(np.abs(np.cbrt(np.linalg.det(self.align[:, :3, :3])) - 1).mean()),
                    static_median_flow=self._static_median(mag),
                    dynamic_mean_flow=float(mag[dyn].mean()) if dyn.any() else 0.0,
                    background_residual=self.background_residual,
                    foreground_fraction=float(self.foreground.mean()) if self.foreground is not None else 1.0,
                    shake_px=float(np.linalg.norm(self.shake, axis=-1).mean()) if self.shake is not None else 0.0,
                    median_depth=float(np.median(self.depth)))


FOREGROUND_PROMPTS = ("a robot arm", "a white robot arm", "a robot gripper", "a human hand",
                      "an object held by a robot gripper")


class ForegroundSegmenter:
    """Text-prompted foreground masks (CLIPSeg): a pixel is foreground if any prompt scores above
    `threshold`; the mask is dilated by `dilate` pixels so object borders keep their flow."""

    def __init__(self, weights: str = "CIDAS/clipseg-rd64-refined", device="cuda",
                 prompts=FOREGROUND_PROMPTS, threshold: float = 0.3, dilate: int = 3, batch: int = 32):
        from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

        self.device = torch.device(device)
        self.model = CLIPSegForImageSegmentation.from_pretrained(weights).to(self.device).eval()
        self.processor = CLIPSegProcessor.from_pretrained(weights)
        self.prompts, self.threshold, self.dilate, self.batch = list(prompts), threshold, dilate, batch

    @torch.no_grad()
    def __call__(self, images: np.ndarray) -> torch.Tensor:
        """images [F, H, W, 3] uint8 -> foreground [F, H, W] bool (on self.device)."""
        f, h, w = images.shape[:3]
        p = len(self.prompts)
        pairs = [(img, txt) for img in images for txt in self.prompts]
        probs = []
        for i in range(0, len(pairs), self.batch):
            chunk = pairs[i:i + self.batch]
            x = self.processor(text=[t for _, t in chunk], images=[im for im, _ in chunk], padding=True,
                               return_tensors="pt").to(self.device)
            logits = self.model(**x).logits.reshape(len(chunk), -1)                # [B, side * side]
            probs.append(torch.sigmoid(logits.float()))
        side = int(round(probs[0].shape[-1] ** 0.5))                                 # 352 for rd64
        prob = torch.cat(probs).reshape(f, p, side, side).amax(1, keepdim=True)
        prob = F.interpolate(prob, (h, w), mode="bilinear", align_corners=False)
        fg = prob > self.threshold
        if self.dilate:
            fg = F.max_pool2d(fg.float(), 2 * self.dilate + 1, 1, self.dilate) > 0
        return fg[:, 0]


class Process4D:
    """DA3 (depth, cameras) + RAFT-small (optical flow) + CLIPSeg (foreground) -> world-space scene flow."""

    def __init__(self, da3: str = "depth-anything/DA3-LARGE-1.1", device="cuda", process_res: int = 504,
                 gap: int = 3, conf_percentile: float = 30.0, flow_batch: int = 8,
                 seg: Optional[str] = "CIDAS/clipseg-rd64-refined", min_flow_px: float = 1.5):
        if str(_DEPENDENCY) not in sys.path:
            sys.path.insert(0, str(_DEPENDENCY))
        from depth_anything_3.api import DepthAnything3
        from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

        self.device = torch.device(device)
        self.da3 = DepthAnything3.from_pretrained(da3).to(self.device).eval()
        weights = Raft_Small_Weights.DEFAULT
        self.raft = raft_small(weights=weights).to(self.device).eval()
        self.raft_tf = weights.transforms()
        self.process_res, self.gap = process_res, gap
        self.conf_percentile, self.flow_batch = conf_percentile, flow_batch
        self.segmenter = ForegroundSegmenter(seg, device=device) if seg and seg != "none" else None
        self.min_flow_px = min_flow_px

    @torch.no_grad()
    def optical_flow(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """a, b [N, 3, H, W] float in [0, 1] -> flow a -> b [N, 2, H, W] (pixels)."""
        h, w = a.shape[-2:]
        h8, w8 = (h + 7) // 8 * 8, (w + 7) // 8 * 8
        out = []
        for i in range(0, a.shape[0], self.flow_batch):
            x = F.interpolate(a[i:i + self.flow_batch], (h8, w8), mode="bilinear", align_corners=False)
            y = F.interpolate(b[i:i + self.flow_batch], (h8, w8), mode="bilinear", align_corners=False)
            x, y = self.raft_tf(x, y)
            f = self.raft(x, y)[-1]
            f = F.interpolate(f, (h, w), mode="bilinear", align_corners=False)
            out.append(f * f.new_tensor([w / w8, h / h8]).view(1, 2, 1, 1))
        return torch.cat(out)

    @torch.no_grad()
    def __call__(self, frames: np.ndarray) -> Result4D:
        """frames: [F, H, W, 3] uint8 (one camera view; crop composite videos first)."""
        pred = self.da3.inference(list(frames), process_res=self.process_res)
        images = pred.processed_images                                              # [F, h, w, 3] uint8
        dev = self.device
        depth = torch.from_numpy(pred.depth).float().to(dev)
        conf = torch.from_numpy(pred.conf).float().to(dev)
        K = torch.from_numpy(pred.intrinsics).float().to(dev)
        c2w = torch.linalg.inv(torch.from_numpy(homogeneous(pred.extrinsics)).float().to(dev))
        points = unproject(depth, K, c2w)
        conf_ok = (conf >= torch.quantile(conf.flatten()[::7], self.conf_percentile / 100)) & (depth > 0)

        img = torch.from_numpy(images).to(dev).permute(0, 3, 1, 2).float() / 255
        g = self.gap
        fw = self.optical_flow(img[:-g], img[g:])
        bw = self.optical_flow(img[g:], img[:-g])
        fg = self.segmenter(images) if self.segmenter is not None else None
        sf, valid, dyn, thr, A, shake = scene_flow(points, conf_ok, fw, bw, g, depth=depth, foreground=fg,
                                                   min_flow_px=self.min_flow_px)
        bg_res = 0.0
        if fg is not None:
            bg = valid & ~fg[:sf.shape[0]]
            bg_res = float(sf.norm(dim=-1)[bg].median()) if bg.any() else 0.0
            sf = sf * fg[:sf.shape[0], ..., None]                                   # background: no scene flow
        cpu = lambda t: t.cpu().numpy()                                            # noqa: E731
        return Result4D(images=images, depth=cpu(depth), conf=cpu(conf), K=cpu(K), c2w=cpu(c2w),
                        points=cpu(points), conf_ok=cpu(conf_ok), flow=cpu(fw.permute(0, 2, 3, 1)),
                        sf=cpu(sf), valid=cpu(valid), dynamic=cpu(dyn), gap=g, threshold=cpu(thr), align=cpu(A),
                        foreground=cpu(fg) if fg is not None else None, shake=cpu(shake),
                        background_residual=bg_res)
