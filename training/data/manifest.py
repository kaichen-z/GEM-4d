"""Dataset manifest parsing shared by training, validation and inference.

There are no predefined datasets. A manifest is a JSON (or JSONL) file that lists
clips; every clip carries its own prompt and (optionally) its own robot description.

Accepted entry forms (they can be mixed in one file):

    {"video": "clips/0001.mp4", "prompt": "pick up the red cube", "robot": "Franka robot arm"}
    ["pick up the red cube", "clips/0001.mp4"]                       # legacy [prompt, video]
    ["pick up the red cube", "clips/0001.mp4", "Franka robot arm"]   # [prompt, video, robot]

`video` may be an .mp4/.webm file, a directory of frames named 0.png, 1.png, ...,
or (inference/validation only) a single image. Relative paths are resolved against
`data_root` if given, else against the manifest's directory.

`robot` is appended to the prompt ("<prompt> <robot>"). Omit it to use the prompt as-is.
A manifest path may also be a directory, in which case every *.json inside is loaded.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Union

VIDEO_KEYS = ("video", "video_path", "path", "rgb", "image")
PROMPT_KEYS = ("prompt", "instruction", "text", "caption")


def _parse_entry(item, base: Path) -> Dict[str, str]:
    if isinstance(item, dict):
        video = next((item[k] for k in VIDEO_KEYS if k in item), None)
        prompt = next((item[k] for k in PROMPT_KEYS if k in item), "")
        robot = item.get("robot", "")
    elif isinstance(item, (list, tuple)) and len(item) >= 2:
        prompt, video = item[0], item[1]
        robot = item[2] if len(item) >= 3 else ""
    else:
        raise ValueError(f"Unrecognised manifest entry: {item!r}")
    if video is None:
        raise ValueError(f"Manifest entry has no video path: {item!r}")
    vp = Path(str(video)).expanduser()
    if not vp.is_absolute():
        vp = base / vp
    return {"video": str(vp), "prompt": str(prompt), "robot": str(robot or "")}


def _read_file(fp: Path) -> list:
    text = fp.read_text(encoding="utf-8")
    if fp.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def load_manifest(path: Union[str, Path], data_root: Optional[str] = None) -> List[Dict[str, str]]:
    p = Path(path).expanduser()
    files = sorted(list(p.glob("*.json")) + list(p.glob("*.jsonl"))) if p.is_dir() else [p]
    entries = []
    for fp in files:
        base = Path(data_root).expanduser() if data_root else fp.parent
        entries.extend(_parse_entry(item, base) for item in _read_file(fp))
    return entries


def build_prompt(entry: Dict[str, str]) -> str:
    prompt = entry["prompt"].strip()
    robot = entry.get("robot", "").strip()
    if robot and robot.lower() not in prompt.lower():
        prompt = f"{prompt} {robot}".strip()
    return prompt
