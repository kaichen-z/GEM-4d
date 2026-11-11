"""Offline point-cloud renders (z-buffered square splats, numpy only) for figures and demo videos."""
import numpy as np

from .end_effector import project


def orbit(c2w: np.ndarray, pivot: np.ndarray, yaw_deg: float = 0.0, pitch_deg: float = 0.0,
          back: float = 0.0) -> np.ndarray:
    """Rotate an OpenCV camera (c2w) about `pivot`: yaw about the camera's up axis, pitch about its right
    axis (positive = look from above), then move it `back` along its viewing direction (scene units)."""
    def rot(axis, deg):
        a = np.deg2rad(deg)
        axis = axis / np.linalg.norm(axis)
        k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * k @ k

    up, right = -c2w[:3, 1], c2w[:3, 0]
    R = rot(right, -pitch_deg) @ rot(up, yaw_deg)
    out = np.eye(4)
    out[:3, :3] = R @ c2w[:3, :3]
    out[:3, 3] = pivot + R @ (c2w[:3, 3] - pivot)
    out[:3, 3] -= back * out[:3, 2]
    return out


def render_points(points: np.ndarray, colors: np.ndarray, K: np.ndarray, c2w: np.ndarray, hw, radius: int = 1,
                  background=(255, 255, 255)) -> np.ndarray:
    """points [N, 3] world, colors [N, 3] uint8 -> image [H, W, 3]; squares of side 2*radius+1, nearest wins."""
    h, w = hw
    uv, z = project(points.astype(np.float64), K, c2w)
    keep = (z > 1e-4) & (uv[:, 0] >= -radius) & (uv[:, 0] < w + radius) & (uv[:, 1] >= -radius) & (uv[:, 1] < h + radius)
    uv, z, col = uv[keep], z[keep], colors[keep]
    x, y = np.floor(uv[:, 0]).astype(np.int64), np.floor(uv[:, 1]).astype(np.int64)
    off = np.arange(-radius, radius + 1)
    dx, dy = np.meshgrid(off, off)
    xx, yy = (x[:, None] + dx.ravel()).ravel(), (y[:, None] + dy.ravel()).ravel()
    zz, src = np.repeat(z, dx.size), np.repeat(np.arange(len(z)), dx.size)
    ok = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
    pix, zz, src = yy[ok] * w + xx[ok], zz[ok], src[ok]
    order = np.lexsort((zz, pix))                                           # per pixel, nearest first
    pix, src = pix[order], src[order]
    first = np.r_[True, pix[1:] != pix[:-1]]
    img = np.empty((h * w, 3), np.uint8)
    img[:] = background
    img[pix[first]] = col[src[first]]
    return img.reshape(h, w, 3)


def font(size: int):
    """A TrueType font (DejaVu Sans from matplotlib or the system) or PIL's default."""
    from PIL import ImageFont

    cands = []
    try:
        import matplotlib
        cands.append(f"{matplotlib.get_data_path()}/fonts/ttf/DejaVuSans-Bold.ttf")
    except ImportError:
        pass
    cands += ["DejaVuSans-Bold.ttf", "/System/Library/Fonts/Helvetica.ttc", "Arial.ttf"]
    for c in cands:
        try:
            return ImageFont.truetype(c, size)
        except OSError:
            continue
    return ImageFont.load_default()


def titled(img: np.ndarray, title: str, size: int = 18, pad: int = 6, bg=(255, 255, 255), fg=(20, 20, 20)):
    """Put `title` above img (centred) -> new image."""
    from PIL import Image, ImageDraw

    f = font(size)
    h = size + 2 * pad
    out = Image.new("RGB", (img.shape[1], img.shape[0] + h), bg)
    out.paste(Image.fromarray(img), (0, h))
    d = ImageDraw.Draw(out)
    tw = d.textlength(title, font=f)
    d.text(((img.shape[1] - tw) / 2, pad - 1), title, fill=fg, font=f)
    return np.asarray(out)
