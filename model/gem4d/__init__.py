"""GEM-4D: Geometry-Enhanced Video World Models for Robot Manipulation."""
from .heads.geo_dit import GeoDiT
from .models.gem4d import GEM4D

BACKBONES = {
    "wan2.2-ti2v-5b": "gem4d.models.wan5b:Wan5BBackbone",      # GEM-4D Wan-5B
    "wan2.1-i2v-14b": "gem4d.models.wan14b:Wan14BBackbone",    # GEM-4D Wan-14B
}


def build_backbone(name: str, **kwargs):
    import importlib
    if name not in BACKBONES:
        raise KeyError(f"unknown backbone {name!r}; choose from {list(BACKBONES)}")
    mod, cls = BACKBONES[name].split(":")
    return getattr(importlib.import_module(mod), cls)(**kwargs)
