"""Load Wan2.1-I2V-14B components from a stock model directory (Wan-AI/Wan2.1-I2V-14B-480P):

    <model_dir>/config.json                                   DiT config
    <model_dir>/diffusion_pytorch_model-*.safetensors (+ index) DiT weights
    <model_dir>/Wan2.1_VAE.pth
    <model_dir>/models_t5_umt5-xxl-enc-bf16.pth
    <model_dir>/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
    <model_dir>/google/umt5-xxl/                              tokenizer

`dit_weights` replaces the DiT weights: a directory in the same layout, or with it under transformer/
(a training export step_N, or the GEM-4D-Wan14B checkpoint), a single .safetensors file or a .pth/.pt state dict.
"""
import glob
import json
import os
from dataclasses import dataclass
from typing import Optional

import torch

from .clip import WanImageEncoderCLIP
from .dit import WanDiT
from .text_encoder import HuggingfaceTokenizer, WanTextEncoder
from .vae import WanVideoVAE

VAE_FILE = "Wan2.1_VAE.pth"
T5_FILE = "models_t5_umt5-xxl-enc-bf16.pth"
CLIP_FILE = "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
TOKENIZER_DIR = os.path.join("google", "umt5-xxl")


@dataclass
class Wan21Components:
    dit: WanDiT
    vae: WanVideoVAE
    text_encoder: Optional[WanTextEncoder]
    tokenizer: Optional[HuggingfaceTokenizer]
    clip: WanImageEncoderCLIP


def _torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except RuntimeError:
        return torch.load(path, map_location="cpu", weights_only=True)


def _unwrap(sd: dict) -> dict:
    for key in ("dit", "state_dict", "module", "model_state"):
        if isinstance(sd.get(key), dict):
            return sd[key]
    return sd


def dit_files(path: str):
    """Weight files of a DiT: a directory (diffusion_pytorch_model*.safetensors, or transformer/ inside) or one file."""
    if os.path.isdir(path):
        if not glob.glob(os.path.join(path, "*.safetensors")) and os.path.isdir(os.path.join(path, "transformer")):
            path = os.path.join(path, "transformer")
        files = sorted(glob.glob(os.path.join(path, "diffusion_pytorch_model*.safetensors")))
        if not files:
            raise FileNotFoundError(f"no diffusion_pytorch_model*.safetensors under {path}")
        return files
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return [path]


def load_dit_state_dict(path: str, dtype: Optional[torch.dtype] = None) -> dict:
    from safetensors.torch import load_file

    sd = {}
    for f in dit_files(path):                 # cast file by file to keep the peak memory low
        part = load_file(f) if f.endswith(".safetensors") else _unwrap(_torch_load(f))
        sd.update({k: v.to(dtype) if dtype is not None else v for k, v in part.items()})
        del part
    return sd


def read_config(model_dir: str) -> dict:
    with open(os.path.join(model_dir, "config.json")) as f:
        return {k: v for k, v in json.load(f).items() if not k.startswith("_")}


def build_dit(model_dir: str, dit_weights: Optional[str] = None, dtype=torch.bfloat16) -> WanDiT:
    """DiT built on 'meta' and filled from the weights (no random init of 14B parameters)."""
    cfg_dir = model_dir
    if dit_weights and os.path.isdir(dit_weights):
        cfg_dir = next((d for d in (dit_weights, os.path.join(dit_weights, "transformer"))
                        if os.path.isfile(os.path.join(d, "config.json"))), model_dir)
    cfg = read_config(cfg_dir)
    with torch.device("meta"):
        dit = WanDiT(**cfg)
    sd = load_dit_state_dict(dit_weights or model_dir, dtype=dtype)
    dit.load_state_dict(sd, strict=True, assign=True)
    dit.init_freqs()
    return dit


def load_wan21_components(model_dir: str, dit_weights: Optional[str] = None, dtype=torch.bfloat16,
                          load_text_encoder: bool = True, max_text_len: int = 512) -> Wan21Components:
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Wan2.1 model directory does not exist: {model_dir}")
    dit = build_dit(model_dir, dit_weights, dtype)

    vae = WanVideoVAE(z_dim=16)
    vsd = _torch_load(os.path.join(model_dir, VAE_FILE))
    vsd = vsd.get("model_state", vsd)
    if not any(k.startswith("model.") for k in vsd):
        vsd = {f"model.{k}": v for k, v in vsd.items()}
    vae.load_state_dict({k: v.float() for k, v in vsd.items()}, strict=True)
    vae.eval().requires_grad_(False)

    text_encoder = tokenizer = None
    if load_text_encoder:
        text_encoder = WanTextEncoder()
        tsd = _unwrap(_torch_load(os.path.join(model_dir, T5_FILE)))
        text_encoder.load_state_dict({k: v.to(dtype) for k, v in tsd.items()}, strict=True)
        text_encoder.to(dtype).eval().requires_grad_(False)
        tokenizer = HuggingfaceTokenizer(name=os.path.join(model_dir, TOKENIZER_DIR), seq_len=max_text_len,
                                         clean="whitespace")

    clip = WanImageEncoderCLIP()
    csd = _torch_load(os.path.join(model_dir, CLIP_FILE))
    csd = {k: v.half() for k, v in csd.items() if k.startswith("visual.") or k == "log_scale"}
    clip.load_state_dict(csd, strict=True)
    clip.half().eval().requires_grad_(False)
    return Wan21Components(dit=dit, vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, clip=clip)
