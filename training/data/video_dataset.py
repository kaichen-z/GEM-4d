"""Generic video dataset for GEM-4D. No dataset is predefined: every clip comes from a manifest
(see manifest.py). Subclass `ManifestVideoDataset` and override `read_frames`, `resize` or
`get_prompt` for custom decoding / cropping / prompts; extra manifest keys are available in `entry`.

Sample:
    video   [F, 3, H, W] float in [-1, 1]  (frame 0 is the conditioning image)
    prompt  str
    path    str
"""
import logging
import random
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .manifest import build_prompt, load_manifest

logger = logging.getLogger(__name__)


def center_crop_resize(frames: List[np.ndarray], size) -> List[np.ndarray]:
    """Center-crop each HxWx3 frame to the target aspect ratio, then resize to (H, W)."""
    th, tw = size
    out = []
    for f in frames:
        h, w = f.shape[:2]
        if w / h > tw / th:
            nw = int(round(h * tw / th)); x0 = (w - nw) // 2
            f = f[:, x0:x0 + nw]
        else:
            nh = int(round(w * th / tw)); y0 = (h - nh) // 2
            f = f[y0:y0 + nh]
        out.append(cv2.resize(f, (tw, th), interpolation=cv2.INTER_LINEAR))
    return out


def read_video(path: Path) -> List[np.ndarray]:
    """All frames of an .mp4/.webm file or a directory of 0.png, 1.png, ... as RGB uint8 arrays."""
    if path.is_dir():
        files = sorted(path.glob("*.png"), key=lambda x: int(x.stem))
        return [np.array(Image.open(f).convert("RGB")) for f in files]
    try:
        import decord
        vr = decord.VideoReader(str(path), num_threads=1)
        return list(vr.get_batch(list(range(len(vr)))).asnumpy())
    except ImportError:
        cap, frames = cv2.VideoCapture(str(path)), []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames


class ManifestVideoDataset(Dataset):
    def __init__(self, manifest: str, data_root: str = None, num_frames: int = 49, height: int = 480,
                 width: int = 640, caption_dropout: float = 0.05, min_frames: int = 1):
        self.samples = load_manifest(manifest, data_root)
        if not self.samples:
            raise ValueError(f"empty manifest: {manifest}")
        self.num_frames, self.size = num_frames, (height, width)
        self.caption_dropout = caption_dropout
        self.min_frames = min_frames

    def __len__(self):
        return len(self.samples)

    # ---- override points ----
    def read_frames(self, entry: Dict) -> List[np.ndarray]:
        return read_video(Path(entry["video"]))

    def resize(self, frames, entry):
        return center_crop_resize(frames, self.size)

    def get_prompt(self, entry) -> str:
        return "" if random.random() < self.caption_dropout else build_prompt(entry).lower()
    # -------------------------

    def fit_length(self, frames: list) -> list:
        """Uniformly subsample long clips; pad short clips by repeating the last frame."""
        n, t = len(frames), self.num_frames
        if n > t:
            return [frames[i] for i in np.linspace(0, n - 1, t, dtype=int)]
        return frames + [frames[-1]] * (t - n)

    def __getitem__(self, idx):
        for _ in range(10):
            entry = self.samples[idx]
            try:
                frames = self.read_frames(entry)
                if len(frames) < self.min_frames:
                    raise ValueError(f"{len(frames)} frames < min_frames {self.min_frames}")
                frames = self.fit_length(self.resize(frames, entry))
                video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 127.5 - 1.0
                return {"video": video, "prompt": self.get_prompt(entry), "path": entry["video"]}
            except Exception as e:  # corrupt / unreadable clip: log and draw another one
                logger.warning(f"skip {entry['video']}: {e}")
                idx = random.randrange(len(self.samples))
        raise RuntimeError("failed to load 10 consecutive samples; check the manifest paths")


def collate(batch):
    return {"video": torch.stack([b["video"] for b in batch]),
            "prompt": [b["prompt"] for b in batch],
            "path": [b["path"] for b in batch]}
