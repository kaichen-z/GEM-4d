"""GEM-4D: Geometry-Enhanced Video World Models for Robot Manipulation."""
from .heads.geo_dit import GeoDiT
from .models.gem4d import GEM4D

BACKBONES = {
    "cogvideox": "gem4d.models.cogvideox:CogVideoXBackbone",
    "wan2.2-ti2v-5b": "gem4d.models.wan:WanTI2VBackbone",
}


def build_backbone(name: str, **kwargs):
    import importlib
    if name not in BACKBONES:
        raise KeyError(f"unknown backbone {name!r}; choose from {list(BACKBONES)}")
    mod, cls = BACKBONES[name].split(":")
    return getattr(importlib.import_module(mod), cls)(**kwargs)
