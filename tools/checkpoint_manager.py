#!/usr/bin/env python3
"""Task-57: shared, framework-agnostic checkpoint-management utility.

Implements one retention policy used consistently across Tesseract, Cosmos and Wan 1.3B:

    A checkpoint at step N is RETAINED iff
      (a) it is among the best K checkpoints by the evaluation metric
          (top-K; falls back to the K latest steps when no metric is available), OR
      (b) N is a positive multiple of `keep_interval`  (periodic / milestone retention).

All other checkpoints are deleted to save storage. K and keep_interval are both configurable;
either rule can be disabled (K<=0 / keep_interval<=0).

Mirrors Tesseract's native `checkpoint_retention.retained_checkpoints` (which is top-K ∪ multiples-of-M)
and generalizes it with optional metric-based top-K, so every framework gets identical behavior and
the same argument names (CHECKPOINT_TOPK / CHECKPOINT_KEEP_INTERVAL).

CLI:
    python checkpoint_manager.py --dir <ckpt_dir> --topk 3 --keep_interval 5000 \
        [--pattern 'checkpoint-(\\d+)'] [--metrics metrics.json] [--metric-mode min] [--dry-run]

`metrics.json` (optional): {"<step>": <float metric>, ...}. With --metric-mode min, lower is better.
Importable: `from checkpoint_manager import retained_checkpoints, prune_checkpoint_dir`.
"""
import argparse
import json
import os
import re
import shutil
from typing import Dict, Iterable, Optional, Set


def retained_checkpoints(
    step_nums: Iterable[int],
    topk: Optional[int],
    keep_interval: Optional[int],
    metrics: Optional[Dict[int, float]] = None,
    metric_mode: str = "min",
) -> Set[int]:
    """Return the subset of `step_nums` to KEEP under (top-K by metric/step) ∪ (multiples of keep_interval).

    topk<=0/None     -> disable the top-K rule.
    keep_interval<=0/None -> disable the periodic rule.
    metrics          -> {step: metric}; when given, top-K is by metric (metric_mode 'min'|'max').
                        When absent, top-K falls back to the K *latest* steps.
    """
    steps = set(int(s) for s in step_nums)
    if not steps:
        return set()

    top: Set[int] = set()
    if topk is not None and topk > 0:
        if metrics:
            scored = [s for s in steps if s in metrics]
            unscored = [s for s in steps if s not in metrics]
            reverse = (metric_mode == "max")
            scored.sort(key=lambda s: metrics[s], reverse=reverse)
            # best-K by metric; if fewer scored than K, top up with the latest unscored steps
            top = set(scored[:topk])
            if len(top) < topk:
                top |= set(sorted(unscored)[-(topk - len(top)):])
        else:
            top = set(sorted(steps)[-topk:])  # K latest

    multiples = (
        {n for n in steps if keep_interval and n % keep_interval == 0}
        if (keep_interval and keep_interval > 0)
        else set()
    )
    return top | multiples


def _scan_steps(ckpt_dir: str, pattern: str) -> Dict[int, str]:
    """Map step -> checkpoint subdir/file name for entries matching `pattern` (a regex with one group)."""
    re_ckpt = re.compile(pattern)
    out: Dict[int, str] = {}
    if not os.path.isdir(ckpt_dir):
        return out
    for name in os.listdir(ckpt_dir):
        m = re_ckpt.fullmatch(name) or re_ckpt.match(name)
        if m:
            try:
                out[int(m.group(1))] = name
            except (ValueError, IndexError):
                continue
    return out


def prune_checkpoint_dir(
    ckpt_dir: str,
    topk: Optional[int],
    keep_interval: Optional[int],
    pattern: str = r"checkpoint-(\d+)",
    metrics: Optional[Dict[int, float]] = None,
    metric_mode: str = "min",
    dry_run: bool = False,
    protect_substr: str = "permanent",
) -> Dict[str, list]:
    """Delete checkpoints in `ckpt_dir` not retained by `retained_checkpoints`.

    Entries whose name contains `protect_substr` are never deleted. Returns {'kept':[...], 'removed':[...]}.
    No-op when both rules are disabled (topk<=0 and keep_interval<=0), so default behavior is unchanged.
    """
    if (not topk or topk <= 0) and (not keep_interval or keep_interval <= 0):
        return {"kept": [], "removed": []}  # nothing requested -> preserve current behavior

    step_to_name = _scan_steps(ckpt_dir, pattern)
    keep_steps = retained_checkpoints(step_to_name.keys(), topk, keep_interval, metrics, metric_mode)
    kept, removed = [], []
    for step, name in sorted(step_to_name.items()):
        if protect_substr and protect_substr in name:
            kept.append(name); continue
        if step in keep_steps:
            kept.append(name)
        else:
            path = os.path.join(ckpt_dir, name)
            removed.append(name)
            if not dry_run:
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
                elif os.path.exists(path):
                    os.remove(path)
    return {"kept": kept, "removed": removed}


def _load_metrics(path: Optional[str]) -> Optional[Dict[int, float]]:
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        raw = json.load(f)
    return {int(k): float(v) for k, v in raw.items()}


def main():
    ap = argparse.ArgumentParser(description="Task-57 checkpoint retention: top-K ∪ periodic.")
    ap.add_argument("--dir", required=True, help="checkpoint directory")
    ap.add_argument("--topk", type=int, default=0, help="keep best/latest K (0=disable)")
    ap.add_argument("--keep_interval", type=int, default=0, help="always keep steps that are multiples of this (0=disable)")
    ap.add_argument("--pattern", default=r"checkpoint-(\d+)", help="regex with one (step) group")
    ap.add_argument("--metrics", default=None, help="optional metrics.json {step: value} for metric-based top-K")
    ap.add_argument("--metric-mode", default="min", choices=["min", "max"])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    res = prune_checkpoint_dir(
        a.dir, a.topk, a.keep_interval, a.pattern,
        _load_metrics(a.metrics), a.metric_mode, a.dry_run,
    )
    tag = "[dry-run] " if a.dry_run else ""
    print(f"{tag}checkpoint_manager: kept {len(res['kept'])} removed {len(res['removed'])} in {a.dir}")
    if res["removed"]:
        print(f"{tag}removed: {', '.join(res['removed'])}")


if __name__ == "__main__":
    main()
