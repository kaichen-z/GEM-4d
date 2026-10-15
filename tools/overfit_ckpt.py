#!/usr/bin/env python3
"""Task-66/67 — shared checkpoint-directory management for the simplified output layout.

Target layout (identical across Tesseract / Wan / Cosmos):

    <output>/
    ├── vis/                       # val_viz: loss PNGs + Step_NNNNNN/{rgb,geo}_sample*.mp4
    └── checkpoints/
        ├── latest -> step_NNNNNN  # symlink to the newest checkpoint
        ├── step_000500/
        ├── step_001000/
        └── ...

This module ONLY manages the checkpoint directory names, the `latest` symlink, latest-detection
for auto-resume, and top-K ∪ multiples-of-M retention. The actual model / optimizer / scheduler
state is written by each framework into the returned step directory (accelerator.save_state,
torch.save, save_pretrained, ...). Pure stdlib; safe to import anywhere.
"""
import os
import re
import shutil

_STEP_RE = re.compile(r"^step_(\d+)$")


def step_dir(ckpt_root, step):
    """Return (and create) <ckpt_root>/step_NNNNNN for the given step."""
    d = os.path.join(ckpt_root, f"step_{int(step):06d}")
    os.makedirs(d, exist_ok=True)
    return d


def update_latest(ckpt_root, step):
    """Point <ckpt_root>/latest at step_NNNNNN (relative symlink; replaced atomically)."""
    target = f"step_{int(step):06d}"
    link = os.path.join(ckpt_root, "latest")
    tmp = link + ".tmp"
    try:
        if os.path.islink(tmp) or os.path.exists(tmp):
            os.remove(tmp)
        os.symlink(target, tmp)
        os.replace(tmp, link)  # atomic swap, works for symlinks
    except OSError:
        # filesystems without symlink support: write a pointer file instead
        with open(os.path.join(ckpt_root, "latest.txt"), "w") as f:
            f.write(target + "\n")


def list_steps(ckpt_root):
    """Sorted list of (step:int, abspath) for every step_NNNNNN under ckpt_root."""
    if not os.path.isdir(ckpt_root):
        return []
    out = []
    for name in os.listdir(ckpt_root):
        m = _STEP_RE.match(name)
        if m and os.path.isdir(os.path.join(ckpt_root, name)):
            out.append((int(m.group(1)), os.path.join(ckpt_root, name)))
    return sorted(out)


def find_latest(ckpt_root):
    """Return (step:int, abspath) of the newest checkpoint, or None if there are none."""
    steps = list_steps(ckpt_root)
    return steps[-1] if steps else None


def retained(steps, keep_topk=0, keep_interval=0):
    """Return the set of steps to KEEP: the keep_topk most recent ∪ all multiples of keep_interval.

    keep_topk<=0 -> keep all recent (no top-k pruning). keep_interval<=0 -> no periodic retention.
    """
    steps = sorted(set(int(s) for s in steps))
    keep = set()
    if keep_topk and keep_topk > 0:
        keep |= set(steps[-keep_topk:])
    else:
        keep |= set(steps)
    if keep_interval and keep_interval > 0:
        keep |= {s for s in steps if s % keep_interval == 0}
    return keep


def prune(ckpt_root, keep_topk=0, keep_interval=0, protect=("latest",)):
    """Delete step_* dirs not in the retention set. Returns list of removed dir names."""
    existing = list_steps(ckpt_root)
    if not existing:
        return []
    keep = retained([s for s, _ in existing], keep_topk, keep_interval)
    removed = []
    for s, path in existing:
        if s not in keep:
            shutil.rmtree(path, ignore_errors=True)
            removed.append(os.path.basename(path))
    return removed


if __name__ == "__main__":
    import tempfile
    root = tempfile.mkdtemp(prefix="ofckpt_")
    for s in (500, 1000, 1500, 2000, 5000):
        step_dir(root, s)
        update_latest(root, s)
    print("latest ->", os.readlink(os.path.join(root, "latest")))
    print("find_latest:", find_latest(root))
    print("removed:", prune(root, keep_topk=2, keep_interval=5000))
    print("remaining:", [os.path.basename(p) for _, p in list_steps(root)])
    assert os.path.basename(find_latest(root)[1]) == "step_005000"
    print("self-test OK")
