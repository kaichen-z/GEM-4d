"""Task-55: shared helpers for loading the Tesseract-style dataset JSON.

The Tesseract train/val pipeline loads data from a JSON file that is a flat list of
``[caption, absolute_video_path]`` pairs, e.g. ``rl100_train.json`` / ``rl100_val.json``::

    [
      ["pick up the stick ...", "/abs/.../front_rgb_frames/video.mp4"],
      ...
    ]

These helpers let the Wan (VideoX-Fun) training/validation pipelines consume that exact
format directly. Generic: any file following the same schema works without code changes.
"""
import json
import os
from typing import List, Tuple

import cv2


def load_tesseract_pairs(json_path: str) -> List[Tuple[str, str]]:
    """Return (prompt, video_path) pairs from a GEM-4D manifest (see manifest.py).
    The entry's robot description is already appended to the prompt."""
    from .manifest import build_prompt, load_manifest
    return [(build_prompt(e), e["video"]) for e in load_manifest(json_path)]


def expand_validation_from_meta(json_path: str, frame_out_dir: str) -> Tuple[List[str], List[str]]:
    """Build (validation_prompts, validation_paths) from a Tesseract-style JSON.

    The Wan I2V validation loop opens ``validation_paths[i]`` as a PIL image (the start
    frame). The Tesseract JSON stores *video* paths, so for each clip we extract the first
    frame to a PNG under ``frame_out_dir`` and return that as the start image, with the
    matching caption as the prompt. This keeps the rest of the validation loop unchanged.
    """
    os.makedirs(frame_out_dir, exist_ok=True)
    pairs = load_tesseract_pairs(json_path)
    prompts: List[str] = []
    image_paths: List[str] = []
    for i, (caption, video_path) in enumerate(pairs):
        cap = cv2.VideoCapture(video_path)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            print(f"[tesseract_json] WARNING: could not read first frame of {video_path}; skipping")
            continue
        out_png = os.path.join(frame_out_dir, f"val_{i:04d}.png")
        cv2.imwrite(out_png, frame)                         # BGR on disk; PIL re-reads as RGB
        prompts.append(caption)
        image_paths.append(out_png)
    if not prompts:
        raise ValueError(f"{json_path}: no readable validation clips found")
    return prompts, image_paths
