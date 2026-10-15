"""YAML config with `_base_` inheritance and command-line overrides (`key.sub=value`)."""
import copy
import os
from typing import Any, Dict, List

import yaml


class Cfg(dict):
    """dict with attribute access (cfg.optim.lr)."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k, v):
        self[k] = v


def _to_cfg(d):
    if isinstance(d, dict):
        return Cfg({k: _to_cfg(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_to_cfg(v) for v in d]
    return d


def _merge(base: Dict, new: Dict) -> Dict:
    out = copy.deepcopy(base)
    for k, v in new.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _load(path: str) -> Dict:
    with open(path) as f:
        d = yaml.safe_load(f) or {}
    base = d.pop("_base_", None)
    if base:
        d = _merge(_load(os.path.join(os.path.dirname(path), base)), d)
    return d


def load_config(path: str, overrides: List[str] = ()) -> Cfg:
    d = _load(path)
    for item in overrides:
        key, _, val = item.partition("=")
        node = d
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        v = yaml.safe_load(val)               # "null" -> None, "true" -> bool, "[1, 2]" -> list
        if isinstance(v, str):                # YAML 1.1 reads "5e-6" as a string
            try:
                v = float(v)
            except ValueError:
                pass
        node[leaf] = v
    return _to_cfg(d)


def dump_config(cfg: Any, path: str):
    def plain(x):
        if isinstance(x, dict):
            return {k: plain(v) for k, v in x.items()}
        if isinstance(x, list):
            return [plain(v) for v in x]
        return x
    with open(path, "w") as f:
        yaml.safe_dump(plain(cfg), f, sort_keys=False)
