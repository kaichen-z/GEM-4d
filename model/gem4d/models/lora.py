"""LoRA for the video DiT (no extra dependency).

    W x  ->  W x + (alpha / r) * B A x          A [r, in] (Kaiming-uniform), B [out, r] (zeros)

Target modules are matched by name suffix, like peft: "q" matches `blocks.3.self_attn.q` and
`blocks.3.cross_attn.q`, "ffn.0" matches `blocks.3.ffn.0`. The default Wan-14B targets (`WAN14B_TARGETS`)
adapts q, k, v, o of self- and cross-attention and both FFN layers of all 40 blocks: 400 layers.

Adapter files:
    <dir>/lora.safetensors      "<module>.lora_A.weight" [r, in], "<module>.lora_B.weight" [out, r]
    <dir>/lora_config.json      {"r", "alpha", "targets"}
peft-style keys ("<module>.lora_A.default.weight", optionally prefixed "base_model.model.") and the
`.pt` payloads ({"lora": state dict, "lora_config": {...}})
load as well.
"""
import json
import math
import os
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

WAN14B_TARGETS = ["q", "k", "v", "o", "ffn.0", "ffn.2"]                         # wan2.1-i2v-14b
WAN5B_TARGETS = ["to_q", "to_k", "to_v", "to_out.0", "ffn.net.0.proj", "ffn.net.2"]   # diffusers Wan2.2
DEFAULT_TARGETS = {"wan2.1-i2v-14b": WAN14B_TARGETS, "wan2.2-ti2v-5b": WAN5B_TARGETS}


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base, self.r, self.alpha = base, int(r), float(alpha)
        self.scaling = self.alpha / self.r
        w = base.weight
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, device=w.device, dtype=w.dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=w.device, dtype=w.dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))                    # peft's default init
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        base.requires_grad_(False)

    @property
    def weight(self):                      # code that reads `.weight` (dtype / device checks) still works
        return self.base.weight

    def forward(self, x):
        y = self.base(x)
        d = self.dropout(x).to(self.lora_A.dtype) @ self.lora_A.t() @ self.lora_B.t()
        return y + (d * self.scaling).to(y.dtype)

    @torch.no_grad()
    def merged(self) -> nn.Linear:
        """Plain Linear with W' = W + (alpha / r) B A, computed in fp32, stored in W's dtype."""
        lin = self.base
        delta = (self.lora_B.float() @ self.lora_A.float()) * self.scaling
        lin.weight.copy_((lin.weight.float() + delta.to(lin.weight.device)).to(lin.weight.dtype))
        return lin


def _matches(name: str, targets: Iterable[str]) -> bool:
    return any(name == t or name.endswith("." + t) for t in targets)


def _set(root: nn.Module, name: str, module: nn.Module):
    parent, _, child = name.rpartition(".")
    setattr(root.get_submodule(parent) if parent else root, child, module)


def inject_lora(model: nn.Module, targets: Iterable[str], r: int = 128, alpha: float = 128.0,
                dropout: float = 0.0) -> List[str]:
    """Freeze `model` and wrap every nn.Linear whose name matches `targets` in a LoRALinear.
    Returns the wrapped module names (only the LoRA matrices are trainable afterwards)."""
    targets = list(targets)
    model.requires_grad_(False)
    names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear) and _matches(n, targets)]
    if not names:
        raise ValueError(f"no nn.Linear matches LoRA targets {targets}")
    for n in names:
        _set(model, n, LoRALinear(model.get_submodule(n), r, alpha, dropout))
    return names


def lora_modules(model: nn.Module) -> Dict[str, LoRALinear]:
    return {n: m for n, m in model.named_modules() if isinstance(m, LoRALinear)}


def has_lora(model: nn.Module) -> bool:
    return any(isinstance(m, LoRALinear) for m in model.modules())


def lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    sd = {}
    for n, m in lora_modules(model).items():
        sd[f"{n}.lora_A.weight"] = m.lora_A.detach().cpu().contiguous()
        sd[f"{n}.lora_B.weight"] = m.lora_B.detach().cpu().contiguous()
    return sd


def save_lora(model: nn.Module, out_dir: str, targets: Iterable[str]):
    from safetensors.torch import save_file

    mods = lora_modules(model)
    m0 = next(iter(mods.values()))
    os.makedirs(out_dir, exist_ok=True)
    save_file(lora_state_dict(model), os.path.join(out_dir, "lora.safetensors"))
    with open(os.path.join(out_dir, "lora_config.json"), "w") as f:
        json.dump({"r": m0.r, "alpha": m0.alpha, "targets": list(targets), "layers": len(mods)}, f, indent=1)


def read_lora(path: str) -> Tuple[Dict[str, torch.Tensor], dict]:
    """Adapter dir / .safetensors / .pt -> ({"<module>.lora_A.weight": ..., ...}, {"r", "alpha", ...})."""
    if os.path.isdir(path):
        sub = os.path.join(path, "lora")
        if not os.path.isfile(os.path.join(path, "lora.safetensors")) and os.path.isdir(sub):
            path = sub
        cfg = json.load(open(os.path.join(path, "lora_config.json")))
        from safetensors.torch import load_file
        raw = load_file(os.path.join(path, "lora.safetensors"))
    elif path.endswith(".safetensors"):
        from safetensors.torch import load_file
        raw = load_file(path)
        cfg = json.load(open(os.path.join(os.path.dirname(path), "lora_config.json")))
    else:                                                                      # original .pt payload
        payload = torch.load(path, map_location="cpu", weights_only=False)
        raw, cfg = payload["lora"], payload["lora_config"]
    sd = {}
    for k, v in raw.items():
        k = k.replace("base_model.model.", "").replace(".default.weight", ".weight")
        sd[k] = v
    return sd, cfg


def load_lora(model: nn.Module, path: str, scale: float = 1.0, merge: bool = True) -> int:
    """Attach an adapter to `model` (injecting LoRA layers where needed) and, by default, merge it.
    `scale` multiplies the adapter (1 = as trained). Returns the number of adapted layers."""
    sd, cfg = read_lora(path)
    names = sorted({k.split(".lora_A.")[0] for k in sd if ".lora_A." in k})
    present = lora_modules(model)
    for n in names:
        if n not in present:
            base = model.get_submodule(n)
            if not isinstance(base, nn.Linear):
                raise KeyError(f"adapter module {n} is not an nn.Linear in this model")
            _set(model, n, LoRALinear(base, int(cfg["r"]), float(cfg["alpha"])))
        m = model.get_submodule(n)
        m.lora_A.data.copy_(sd[f"{n}.lora_A.weight"].to(m.lora_A))
        m.lora_B.data.copy_((sd[f"{n}.lora_B.weight"] * scale).to(m.lora_B))
    if merge:
        merge_lora(model)
    return len(names)


def merge_lora(model: nn.Module) -> int:
    """Replace every LoRALinear by its merged nn.Linear (in place). Returns the number merged."""
    mods = lora_modules(model)
    for n, m in mods.items():
        _set(model, n, m.merged())
    return len(mods)


def resolve_targets(backbone: str, targets: Optional[Iterable[str]]) -> List[str]:
    if targets:
        return list(targets)
    if backbone not in DEFAULT_TARGETS:
        raise KeyError(f"no default LoRA targets for {backbone!r}; set lora.targets")
    return DEFAULT_TARGETS[backbone]
