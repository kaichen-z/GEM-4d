"""4D-process: DA3 depth + cameras and RAFT-small optical flow -> scene flow of a generated video,
with a DynPoint-style interactive viewer. Used by `inference/infer.py --process4d` and
`inference/process4d.py`."""
import json
import os

import numpy as np

from .flow_vis import FlowPanel
from .scene_flow import Process4D, Result4D, scene_flow, unproject
from .viewer import summary_frames, viewer_sequence, write_viewer

__all__ = ["FlowPanel", "Process4D", "Result4D", "scene_flow", "unproject", "crop_frames", "run_4d_process",
           "viewer_sequence", "write_viewer"]


def crop_frames(frames: np.ndarray, crop=None) -> np.ndarray:
    """crop = (x0, y0, x1, y1) in pixels of the video, e.g. the main view of a composite video."""
    frames = np.asarray(frames)
    if not crop:
        return frames
    x0, y0, x1, y1 = crop
    return frames[:, y0:y1, x0:x1]


def run_4d_process(proc: Process4D, frames: np.ndarray, out_dir: str, stem: str, crop=None, fps: int = 8,
                   save_npz: bool = True) -> dict:
    """frames [F, H, W, 3] uint8 -> <stem>_4d.html (viewer), <stem>_4d.mp4 (frame | depth & scene-flow
    arrows), <stem>_4d.json (statistics) and <stem>_4d.npz (depth, cameras, flow, scene flow)."""
    from ..utils.video_io import save_mp4

    os.makedirs(out_dir, exist_ok=True)
    r = proc(crop_frames(np.asarray(frames), crop))
    base = os.path.join(out_dir, f"{stem}_4d")
    write_viewer(base + ".html", [viewer_sequence(r, stem)], title=f"4D-process: {stem}")
    save_mp4(summary_frames(r), base + ".mp4", fps=fps)
    stats = r.stats()
    with open(base + ".json", "w") as f:
        json.dump(stats, f, indent=1)
    if save_npz:
        np.savez_compressed(base + ".npz", depth=r.depth.astype(np.float16), conf=r.conf.astype(np.float16),
                            K=r.K, c2w=r.c2w, flow=r.flow.astype(np.float16), scene_flow=r.sf.astype(np.float16),
                            valid=r.valid, dynamic=r.dynamic, gap=r.gap, threshold=r.threshold,
                            align=r.align, shake=r.shake,
                            **({"foreground": r.foreground} if r.foreground is not None else {}))
    return dict(stats=stats, result=r)
