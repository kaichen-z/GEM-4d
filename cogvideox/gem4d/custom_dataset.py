"""Generic video dataset for GEM-4D (CogVideoX backbone).

No dataset is predefined: everything comes from the manifest (see manifest.py).
To adapt to new data you normally only edit the manifest. For special needs
(other video containers, per-dataset cropping, extra augmentation) subclass
`ManifestVideoDataset` and override `read_frames`, `resize` or `get_prompt`.

Output per sample (same contract the trainer expects):
    prompt  str
    image   [1, 9, H, W]  first frame (conditioning), values in [-1, 1]
    video   [F, 9, H, W]  channels 0:3 = RGB; 3:9 are zero placeholders (legacy
                          depth/normal slots, unused by REF15 / RGB_ONLY training —
                          geometry targets are computed on the fly by DA3)
    mask    [rgb, depth, normal] availability flags
    path    source video path
"""
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from manifest import build_prompt, load_manifest

logger = logging.getLogger(__name__)
TRY_TIMES = 10


def crop_and_resize_frames(frames: List[np.ndarray], target_size: Tuple[int, int]) -> List[np.ndarray]:
    """Center-crop each HxWx3 frame to the target aspect ratio, then resize to (H, W)."""
    th, tw = target_size
    out = []
    for f in frames:
        h, w = f.shape[:2]
        if w / h > tw / th:            # too wide -> crop width
            nw = int(round(h * tw / th)); x0 = (w - nw) // 2
            f = f[:, x0:x0 + nw]
        else:                          # too tall -> crop height
            nh = int(round(w * th / tw)); y0 = (h - nh) // 2
            f = f[y0:y0 + nh]
        out.append(cv2.resize(f, (tw, th), interpolation=cv2.INTER_LINEAR))
    return out


class ManifestVideoDataset(Dataset):
    def __init__(
        self,
        dataset_file: str,
        data_root: Optional[str] = None,
        max_num_frames: int = 49,
        height: int = 480,
        width: int = 640,
        id_token: Optional[str] = None,
        caption_dropout: float = 0.05,
        random_flip: Optional[float] = None,
        **unused,
    ) -> None:
        super().__init__()
        self.samples = load_manifest(dataset_file, data_root)
        if not self.samples:
            raise ValueError(f"Empty manifest: {dataset_file}")
        self.max_num_frames = max_num_frames
        self.target_size = (height, width)
        self.id_token = f"{id_token.strip()} " if id_token else ""
        self.caption_dropout = caption_dropout
        self.random_flip = random_flip
        # single bucket: every clip is resized to (max_num_frames, height, width)
        self.resolutions = [(max_num_frames, height, width)]

    def __len__(self) -> int:
        return len(self.samples)

    # ---------------- override points ----------------
    def read_frames(self, path: Path) -> List[np.ndarray]:
        """Return all frames of a clip as a list of HxWx3 uint8 RGB arrays."""
        if path.is_dir():
            fps = sorted(path.glob("*.png"), key=lambda x: int(x.stem))
            return [np.array(Image.open(fp).convert("RGB")) for fp in fps]
        cap = cv2.VideoCapture(str(path))
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames:  # OpenCV's ffmpeg can fail to open h264 (e.g. thread limits); try decord
            import decord
            vr = decord.VideoReader(str(path), num_threads=1)
            frames = list(vr.get_batch(list(range(len(vr)))).asnumpy())
        if not frames:
            raise ValueError(f"No frames decoded from {path}")
        return frames

    def resize(self, frames: List[np.ndarray], entry: Dict[str, str]) -> List[np.ndarray]:
        return crop_and_resize_frames(frames, self.target_size)

    def get_prompt(self, entry: Dict[str, str]) -> str:
        if random.random() < self.caption_dropout:
            return ""
        return self.id_token + build_prompt(entry).lower()
    # --------------------------------------------------

    def adjust_num_frames(self, frames: list) -> list:
        """Pad short clips by repeating the last frame; subsample long clips uniformly."""
        n, t = len(frames), self.max_num_frames
        if n < t:
            frames = frames + [frames[-1]] * (t - n)
        elif n > t:
            frames = [frames[i] for i in np.linspace(0, n - 1, t, dtype=int)]
        return frames

    def __getitem__(self, index):
        if isinstance(index, list):        # BucketSampler hands over pre-loaded items
            return index
        for _ in range(TRY_TIMES):
            try:
                return self.getitem(index)
            except Exception as e:
                logger.error(f"Error loading {self.samples[index]['video']}: {e}")
                index = random.randint(0, len(self.samples) - 1)
        raise RuntimeError(f"Failed to load a sample after {TRY_TIMES} tries")

    def getitem(self, index: int) -> Dict[str, Any]:
        entry = self.samples[index]
        frames = self.adjust_num_frames(self.resize(self.read_frames(Path(entry["video"])), entry))
        rgb = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 127.5 - 1.0   # [F,3,H,W]
        if self.random_flip and random.random() < self.random_flip:
            rgb = rgb.flip(-1)
        video = torch.cat([rgb, torch.zeros_like(rgb), torch.zeros_like(rgb)], dim=1)       # [F,9,H,W]
        return {
            "prompt": self.get_prompt(entry),
            "image": video[:1].clone(),
            "video": video,
            "mask": [True, False, False],
            "path": entry["video"],
            "video_metadata": {"num_frames": video.shape[0], "height": video.shape[2], "width": video.shape[3]},
        }


class BucketSampler(Sampler):
    """Groups samples of equal (frames, height, width) into batches."""

    def __init__(self, data_source: ManifestVideoDataset, batch_size: int = 1, shuffle: bool = True,
                 drop_last: bool = False) -> None:
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __len__(self):
        return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        order = list(range(len(self.data_source)))
        if self.shuffle:
            random.shuffle(order)
        buckets: Dict[tuple, list] = {}
        for i in order:
            data = self.data_source[i]
            m = data["video_metadata"]
            key = (m["num_frames"], m["height"], m["width"])
            buckets.setdefault(key, []).append(data)
            if len(buckets[key]) == self.batch_size:
                yield buckets.pop(key)
        if not self.drop_last:
            for bucket in buckets.values():
                if bucket:
                    yield bucket
