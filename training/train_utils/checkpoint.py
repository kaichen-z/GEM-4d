"""Checkpoints.

    <output_dir>/checkpoints/
        step_000500/transformer/      exported video DiT (diffusers format) -> inference / fine-tuning
        step_000500/geometry.safetensors   Geometry DiT + projector (training only)
        ...
        state/                         full training state of the newest save (accelerate/DeepSpeed:
                                       weights + optimizer + LR scheduler + RNG), used for resume
        latest                         text file with the newest step

Exports are pruned to the newest `keep_last` plus every multiple of `keep_every`. Only one full
state is kept (it is large: fp32 master weights + Adam moments).
"""
import os
import re
import shutil
from typing import Optional

import torch
from safetensors.torch import save_file

STEP_RE = re.compile(r"^step_(\d+)$")


def ckpt_root(output_dir: str) -> str:
    return os.path.join(output_dir, "checkpoints")


def list_steps(root: str):
    if not os.path.isdir(root):
        return []
    return sorted(int(m.group(1)) for d in os.listdir(root) if (m := STEP_RE.match(d)))


def latest_state(output_dir: str) -> Optional[int]:
    """Step of the resumable state, or None."""
    root = ckpt_root(output_dir)
    f = os.path.join(root, "latest")
    if os.path.isfile(f) and os.path.isdir(os.path.join(root, "state")):
        return int(open(f).read().strip())
    return None


def save(accelerator, model, output_dir: str, step: int, keep_last: int = 3, keep_every: int = 2000,
         save_state: bool = True):
    """Call on ALL ranks (DeepSpeed shards the optimizer state across ranks)."""
    root = ckpt_root(output_dir)
    step_dir = os.path.join(root, f"step_{step:06d}")
    if save_state:
        tmp = os.path.join(root, "state.tmp")
        if accelerator.is_main_process and os.path.isdir(tmp):
            shutil.rmtree(tmp)
        accelerator.wait_for_everyone()
        accelerator.save_state(tmp)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        m = accelerator.unwrap_model(model)
        m.transformer.save_pretrained(os.path.join(step_dir, "transformer"), safe_serialization=True,
                                      max_shard_size="5GB")
        geo = {}
        for name in ("geo_dit", "projector"):
            if hasattr(m, name):
                geo.update({f"{name}.{k}": v.detach().cpu().contiguous()
                            for k, v in getattr(m, name).state_dict().items()})
        if geo:
            save_file(geo, os.path.join(step_dir, "geometry.safetensors"))
        if save_state:
            final = os.path.join(root, "state")
            if os.path.isdir(final):
                shutil.rmtree(final)
            os.replace(os.path.join(root, "state.tmp"), final)
        with open(os.path.join(root, "latest"), "w") as f:
            f.write(str(step))
        prune(root, keep_last, keep_every)
    accelerator.wait_for_everyone()
    return step_dir


def prune(root: str, keep_last: int, keep_every: int):
    steps = list_steps(root)
    keep = set(steps[-keep_last:]) if keep_last > 0 else set(steps)
    if keep_every > 0:
        keep |= {s for s in steps if s % keep_every == 0}
    for s in steps:
        if s not in keep:
            shutil.rmtree(os.path.join(root, f"step_{s:06d}"), ignore_errors=True)


def resume(accelerator, output_dir: str) -> int:
    """Restore the full training state if present; returns the global step (0 if fresh)."""
    step = latest_state(output_dir)
    if step is None:
        return 0
    accelerator.load_state(os.path.join(ckpt_root(output_dir), "state"))
    accelerator.print(f"[resume] restored full training state at step {step}")
    return step


def load_geometry(model, path: str):
    """Load geometry.safetensors (Geo-DiT + projector) into an unwrapped GEM4D module."""
    from safetensors.torch import load_file
    sd = load_file(path)
    for name in ("geo_dit", "projector"):
        part = {k[len(name) + 1:]: v for k, v in sd.items() if k.startswith(name + ".")}
        if part and hasattr(model, name):
            getattr(model, name).load_state_dict(part, strict=True)
