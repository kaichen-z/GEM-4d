#!/usr/bin/env python3
"""Task-62 — shared, framework-agnostic validation-visualization helper.

Produces the standardized Task-62 output layout, identical across CogVideoX/Tesseract,
Wan-1.3B and Cosmos-2.5:

    output/
    ├── train_rgb_loss.png   ├── train_geo_loss.png
    ├── val_rgb_loss.png     ├── val_geo_loss.png
    ├── Step_000001/
    │   ├── rgb_sample0.mp4   (GT-left | pred-right, RGB)
    │   ├── geo_sample0.mp4   (GT-left | pred-right, geometry feature PCA->RGB)
    │   └── ...
    └── Step_000500/ ...

All inputs accept either numpy arrays or torch tensors (duck-typed via np.asarray, so
torch is never imported here). Designed to be called from each framework's existing
validation hook. Pure CPU; depends only on numpy + imageio (already in every env).
"""
import os
import numpy as np
import imageio.v2 as imageio


# --------------------------------------------------------------------------- utils
def _to_numpy(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach().to("cpu")
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


def _to_THWC_uint8(v):
    """Normalize an RGB clip of arbitrary common layout to [T,H,W,3] uint8 in [0,255].

    Accepts [T,H,W,3] / [T,3,H,W] / [3,T,H,W]; float in [0,1] or [-1,1] or uint8.
    """
    v = _to_numpy(v).astype(np.float32) if not np.issubdtype(_to_numpy(v).dtype, np.uint8) else _to_numpy(v)
    v = np.asarray(v)
    if v.ndim != 4:
        raise ValueError(f"expected 4D RGB clip, got shape {v.shape}")
    # move channel-3 axis to last
    if v.shape[-1] == 3:
        thwc = v
    elif v.shape[0] == 3:                 # [3,T,H,W]
        thwc = np.transpose(v, (1, 2, 3, 0))
    elif v.shape[1] == 3:                 # [T,3,H,W]
        thwc = np.transpose(v, (0, 2, 3, 1))
    else:
        raise ValueError(f"cannot find RGB channel axis in shape {v.shape}")
    if not np.issubdtype(thwc.dtype, np.uint8):
        lo, hi = float(thwc.min()), float(thwc.max())
        if lo < -0.01:                    # assume [-1,1]
            thwc = (thwc + 1.0) / 2.0
        elif hi > 1.01:                   # assume [0,255]
            thwc = thwc / 255.0
        thwc = np.clip(thwc, 0, 1)
        thwc = (thwc * 255.0).round().astype(np.uint8)
    return thwc


def _pca_rgb(feat_CHW):
    """[C,H,W] geometry-feature map -> [H,W,3] uint8 via PCA-to-RGB (percentile-normalized)."""
    feat = _to_numpy(feat_CHW).astype(np.float32)
    C, H, W = feat.shape
    x = feat.reshape(C, H * W).T
    x = x - x.mean(0, keepdims=True)
    try:
        _, _, vt = np.linalg.svd(x, full_matrices=False)
        comp = x @ vt[:3].T
    except np.linalg.LinAlgError:
        comp = x[:, :3] if C >= 3 else np.repeat(x[:, :1], 3, axis=1)
    comp = comp.reshape(H, W, 3)
    lo, hi = np.percentile(comp, 2), np.percentile(comp, 98)
    comp = np.clip((comp - lo) / (hi - lo + 1e-8), 0, 1)
    return (comp * 255).round().astype(np.uint8)


def _hconcat(a, b):
    """Side-by-side [T,H,W,3] | [T,H,W,3] -> [T,H,2W,3] (matches T, H by min-crop/pad)."""
    T = min(a.shape[0], b.shape[0])
    a, b = a[:T], b[:T]
    H = min(a.shape[1], b.shape[1])
    a, b = a[:, :H], b[:, :H]
    return np.concatenate([a, b], axis=2)


def _write_mp4(path, frames_THWC, fps=8):
    frames = _to_THWC_uint8(frames_THWC)
    # h264 needs even dims
    H, W = frames.shape[1], frames.shape[2]
    if H % 2 or W % 2:
        frames = frames[:, : H - (H % 2), : W - (W % 2)]
    imageio.mimsave(path, list(frames), fps=fps, macro_block_size=1)


def _step_dir(root, step):
    d = os.path.join(root, f"Step_{int(step):06d}")
    os.makedirs(d, exist_ok=True)
    return d


# ----------------------------------------------------------------------- public API
def save_rgb_sample(root, step, idx, gt_clip, pred_clip, fps=8):
    """Write Step_{step}/rgb_sample{idx}.mp4 = GT(left) | pred(right)."""
    gt = _to_THWC_uint8(gt_clip)
    pred = _to_THWC_uint8(pred_clip)
    out = os.path.join(_step_dir(root, step), f"rgb_sample{idx}.mp4")
    _write_mp4(out, _hconcat(gt, pred), fps=fps)
    return out


def save_geo_sample(root, step, idx, gt_latent, pred_latent, upscale=8, fps=8, layout=None):
    """Write Step_{step}/geo_sample{idx}.mp4 = GT(left) | pred(right), geometry PCA->RGB.

    gt_latent / pred_latent: geometry-feature latents [C,T,h,w] (or [T,C,h,w]).
    layout: explicit axis order of the inputs — "CTHW" or "TCHW". When None (default)
    a size heuristic is used (assumes the smaller of axis0/axis1 is the channel axis),
    which is correct only when channels <= time. Pass layout explicitly when the
    channel count can exceed the (latent-)time length, e.g. Tesseract's [T,16,h,w].
    """
    g = _to_numpy(gt_latent)
    p = _to_numpy(pred_latent)
    # normalize to [C,T,h,w]
    def _ctHW(z):
        if z.ndim != 4:
            raise ValueError(f"geo latent must be 4D, got {z.shape}")
        if layout == "CTHW":
            return z
        if layout == "TCHW":
            return np.transpose(z, (1, 0, 2, 3))
        # heuristic: time axis is the larger of axis0/axis1 vs channel; assume [C,T,h,w] if axis0<=axis1
        return z if z.shape[0] <= z.shape[1] else np.transpose(z, (1, 0, 2, 3))
    g, p = _ctHW(g), _ctHW(p)
    T = min(g.shape[1], p.shape[1])
    frames = []
    for t in range(T):
        gi = _pca_rgb(g[:, t])
        pi = _pca_rgb(p[:, t])
        pair = np.concatenate([gi, pi], axis=1)            # [h, 2w, 3]
        if upscale > 1:
            pair = np.kron(pair, np.ones((upscale, upscale, 1), dtype=np.uint8))
        frames.append(pair)
    out = os.path.join(_step_dir(root, step), f"geo_sample{idx}.mp4")
    _write_mp4(out, np.stack(frames, 0), fps=fps)
    return out


def save_loss_curves(root, history):
    """Write the 4 root loss PNGs from a history dict. Missing/empty series -> skipped.

    history: {"train_rgb": [(step,val),...], "train_geo": [...], "val_rgb": [...], "val_geo": [...]}
    Each value list is (step, loss) pairs. Saves train_rgb_loss.png etc. at `root`.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(root, exist_ok=True)
    written = []
    for key, fname, color in [
        ("train_rgb", "train_rgb_loss.png", "C0"),
        ("train_geo", "train_geo_loss.png", "C1"),
        ("val_rgb", "val_rgb_loss.png", "C2"),
        ("val_geo", "val_geo_loss.png", "C3"),
    ]:
        series = history.get(key) or []
        fig, ax = plt.subplots(figsize=(7, 4))
        if series:
            arr = np.asarray(series, dtype=np.float64)
            s, y = arr[:, 0], arr[:, 1]
            ax.plot(s, y, color=color, alpha=0.35)
            if len(y) >= 11:
                k = 11
                ax.plot(s, np.convolve(y, np.ones(k) / k, mode="same"), color=color, label=key)
            ax.set_yscale("log")
        else:
            ax.text(0.5, 0.5, f"no {key} data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(fname[:-4]); ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(root, fname), dpi=120); plt.close(fig)
        written.append(fname)
    return written


# ------------------------------------------------------------------------- self-test
if __name__ == "__main__":
    import tempfile, sys
    d = tempfile.mkdtemp(prefix="valviz_")
    rng = np.random.RandomState(0)
    gt = rng.rand(9, 3, 64, 64).astype(np.float32)
    pred = rng.rand(9, 3, 64, 64).astype(np.float32)
    save_rgb_sample(d, 1, 0, gt, pred)
    geo_gt = rng.rand(16, 3, 16, 16).astype(np.float32)
    geo_pred = geo_gt + 0.05 * rng.rand(16, 3, 16, 16).astype(np.float32)
    save_geo_sample(d, 1, 0, geo_gt, geo_pred)
    hist = {"train_rgb": [(i, 1.0 / (i + 1)) for i in range(50)],
            "train_geo": [(i, 0.5 / (i + 1)) for i in range(50)],
            "val_rgb": [(i, 1.1 / (i + 1)) for i in range(0, 50, 10)],
            "val_geo": []}
    save_loss_curves(d, hist)
    import glob
    print("self-test wrote:")
    for f in sorted(glob.glob(d + "/**/*", recursive=True)):
        print("  ", f, os.path.getsize(f) if os.path.isfile(f) else "(dir)")
    sys.exit(0)
