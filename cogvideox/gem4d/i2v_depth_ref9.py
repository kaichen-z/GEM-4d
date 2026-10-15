import gc
import contextlib
import cv2
import logging
import math
import os
import random
import shutil
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict
import json
import numpy as np
import diffusers
import torch
import transformers
import wandb
from accelerate import Accelerator, DistributedType, init_empty_weights
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    set_seed,)
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,)
from diffusers.training_utils import cast_training_params
from diffusers.utils import export_to_video, load_image
from diffusers.optimization import get_scheduler
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from torch.utils.data import DataLoader
import torch.distributed as dist
from tqdm.auto import tqdm
from transformers import AutoTokenizer, T5EncoderModel
from gem4d.modules.gem4d_model import GEM4DDepthNormal_refine3 as CogVideoXTransformer3DModel
from args import get_args  # isort:skip
from custom_dataset import ManifestVideoDataset, crop_and_resize_frames, BucketSampler
from gem4d.text_encoder import compute_prompt_embeddings  # isort:skip
from utils import (
    get_gradient_norm,
    get_optimizer,
    prepare_rotary_positional_embeddings,
    print_memory,
    reset_memory,
    unwrap_model,
    make_save_model_hook,
    make_load_model_hook,
    CHECKPOINT_MANAGER,
    loss_fn)
import glob
import re
from validation_utils_ref import run_validation
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

# Module-level logger so helpers like compute_validation_loss (which logs the
# val geo-align warning) have `logger` in scope; main() rebinds the same object.
logger = get_logger(__name__)


def _ensure_da3_src_path() -> None:
    """Add Row-4D/depth/src so `depth_anything_3` imports (same tree as gem4d/)."""
    depth_src = Path(__file__).resolve().parents[2] / "depth" / "src"
    if depth_src.is_dir():
        s = str(depth_src)
        if s not in sys.path:
            sys.path.insert(0, s)


def compute_weight_sum_safe(model):
    total = torch.zeros(1, dtype=torch.float64)
    with torch.no_grad():
        for param in model.parameters():
            total += param.detach().to(torch.float64).abs().sum()
    print(f"Weight sum: {total.item()} ================================")
    return total.item()

class DA3_Block(nn.Module):
    """
    Depth Anything 3 branch for diffusion training — same role as ``VGGT_Block`` in ``i2v_depth_ref8.py``:
    load a frozen depth model, run video frames through it, return patch tokens for ``refine3`` (2048-dim).

    Uses the official tensor API from ``Row-4D/depth`` (``depth_anything_3``):
    - Input layout matches ``DepthAnything3Net.forward``: ``(B, N, 3, H, W)`` with ``N`` = time (frames).
    - Features come from ``export_feat_layers`` → ``output.aux[\"feat_layer_{i}\"]`` (same as ``DepthAnything3.forward``
      in ``depth/src/depth_anything_3/api.py``).

    For ``NestedDepthAnything3Net`` (metric variant), we only run the main ``da3`` sub-module so we do not
    double-run the metric branch when we only need backbone features (see ``NestedDepthAnything3Net`` in
    ``depth_anything_3/model/da3.py``).

    Hub id default: ``depth-anything/DA3-LARGE-1.1``. Override with env ``DA3_PRETRAINED`` or a local
    ``from_pretrained`` directory checkpoint path.
    """
    def __init__(
        self,
        DEFAULT_SIZE: int = 336,
        feat_layer: int = 19,
        pretrained_model_name_or_path: str | None = None,
    ):
        super().__init__()
        _ensure_da3_src_path()
        from depth_anything_3.api import DepthAnything3
        pid = pretrained_model_name_or_path or os.environ.get(
            "DA3_PRETRAINED", "depth-anything/DA3-LARGE-1.1")
        self.wrapper = DepthAnything3.from_pretrained(pid)
        root = self.wrapper.model
        self.da3_model = root.da3 if hasattr(root, "da3") else root
        self.DEFAULT_SIZE = int(DEFAULT_SIZE)
        # Task-32: which DA3 (DINOv2 ViT-L, 24 layers) feature layer is the geometry target. ref15
        # now uses the MIDPOINT (DA3_FEAT_LAYER=12) instead of the deep layer 19, symmetric with the
        # RGB midpoint tap (GEO_HOOK_LAYER) and with Cosmos (VGGT-Omega layer 11/24).
        self.feat_layer = int(os.environ.get("DA3_FEAT_LAYER", str(feat_layer)))
        self.export_feat_layers = [self.feat_layer]
        self.patch_size = int(getattr(self.da3_model, "PATCH_SIZE", 14))

    def loading_da3(self) -> None:
        """Mirror ``VGGT_Block.loading_vggt``: eval + no grad (DA3 weights stay fixed during diffusion SFT)."""
        self.da3_model.eval()
        for p in self.da3_model.parameters():
            p.requires_grad = False

    def da3_processor(self, images: Tensor) -> Tensor:
        """``images``: ``[B, T, 3, H, W]`` in ``[0, 1]`` → resize + ImageNet norm, shape ``(B, N, 3, H', W')`` for DA3."""
        b = images.shape[0]
        x = rearrange(images, "b f c h w -> (b f) c h w")
        target = (self.DEFAULT_SIZE, self.DEFAULT_SIZE)
        x = F.interpolate(x, size=target, mode="bilinear", align_corners=False)
        x = rearrange(x, "(b f) c h w -> b f c h w", b=b)
        x = torch.clamp(x, 0.0, 1.0)
        mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
        std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
        x = (x - mean) / std
        H, W = x.shape[-2], x.shape[-1]
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            nh = max(self.patch_size, (H // self.patch_size) * self.patch_size)
            nw = max(self.patch_size, (W // self.patch_size) * self.patch_size)
            x = rearrange(x, "b f c h w -> (b f) c h w")
            x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
            x = rearrange(x, "(b f) c h w -> b f c h w", b=b)
        return x

    def pre_forward_tensor(self, images: Tensor) -> Tensor:
        """Same return layout as ``VGGT_Block.pre_forward_tensor``: ``[1, B, T, P, C]`` (``P = Hp*Wp``)."""
        b, t, _, _, _ = images.shape
        x = self.da3_processor(images)
        x = x.to(dtype=torch.float32)
        x = x.to(next(self.da3_model.parameters()).device)
        with torch.no_grad():
            out = self.da3_model(
                x,
                extrinsics=None,
                intrinsics=None,
                export_feat_layers=self.export_feat_layers,
                infer_gs=False,
                use_ray_pose=False,
                ref_view_strategy=os.environ.get("DA3_REF_VIEW_STRATEGY", "saddle_balanced"),
            )
        aux = out.aux
        key = f"feat_layer_{self.feat_layer}"
        if key not in aux:
            raise KeyError(f"DA3 aux missing {key}; check feat_layer vs backbone depth.")
        feat = aux[key].float()
        C = feat.shape[-1]
        if C != 1024:
            raise RuntimeError(
                f"DA3 backbone channel {C} != 1024; expected DINOv2 ViT-L (DA3-LARGE-1.1). "
                "If you swap the backbone, also update refine3.compress in gem4d_model.py."
            )
        feat = feat.reshape(b, t, -1, C)
        return torch.stack([feat], dim=0)

def _rgb_only_enabled() -> bool:
    """Task-27-10: ``RGB_ONLY=1`` disables the geometry branch entirely.

    No DepthAnything-3 is loaded, no Geo-DiT / depth head is evaluated, and the loss is the plain
    RGB v-prediction objective. Use it to continue fine-tuning a released checkpoint on new data
    without the DA3 geometry-forcing teacher (see ``finetune_rgb_only.sh``).
    """
    return os.environ.get("RGB_ONLY", "0") == "1"


class ZeroGeo_Block(nn.Module):
    """Drop-in stand-in for :class:`DA3_Block` used when ``RGB_ONLY=1``.

    Returns zero tokens with DA3's exact output layout ``[1, B, T, P, 1024]`` so every
    shape-dependent code path downstream keeps working, while loading no geometry weights at all.
    Under ``RGB_ONLY`` the geometry loss is skipped, so these zeros never reach a loss term — they
    only keep the rgb/geo latent bookkeeping (concat-then-split) intact.
    """

    def __init__(self, DEFAULT_SIZE: int = 336, patch_size: int = 14):
        super().__init__()
        self.DEFAULT_SIZE = int(DEFAULT_SIZE)
        self.patch_size = int(patch_size)
        self.num_patches = (self.DEFAULT_SIZE // self.patch_size) ** 2

    def loading_da3(self) -> None:
        return

    def pre_forward_tensor(self, images: Tensor) -> Tensor:
        b, t = images.shape[0], images.shape[1]
        return images.new_zeros((1, b, t, self.num_patches, 1024), dtype=torch.float32)


def _select_trainable_param_probe(model: torch.nn.Module, min_numel: int = 1024):
    """ Pick one trainable parameter to probe for "did weights update?" checks. """
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        if p is None: continue
        if p.numel() >= min_numel: return name, p
    for name, p in model.named_parameters():
        if getattr(p, "requires_grad", False) and p is not None and p.numel() > 0:
            return name, p
    return None, None

class CollateFunction:
    def __init__(self, weight_dtype: torch.dtype, load_tensors: bool) -> None:
        self.weight_dtype = weight_dtype
        self.load_tensors = load_tensors
    def __call__(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        prompts = [x["prompt"] for x in data[0]]
        if self.load_tensors: prompts = torch.stack(prompts).to(dtype=self.weight_dtype, non_blocking=True)
        images = [x["image"] for x in data[0]]
        images = torch.stack(images).to(dtype=self.weight_dtype, non_blocking=True)
        videos = [x["video"] for x in data[0]]
        videos = torch.stack(videos).to(dtype=self.weight_dtype, non_blocking=True)
        masks = [x["mask"] for x in data[0]]
        masks = torch.tensor(masks).to(dtype=torch.bool, non_blocking=True)
        paths = [x["path"] for x in data[0]]
        return { "images": images, "videos": videos, "prompts": prompts, "masks": masks, "paths": paths,}

@torch.no_grad()
def compute_validation_loss(
    args, accelerator, transformer, vae, text_encoder, tokenizer, scheduler,
    val_dataloader, da3_block, weight_dtype, model_config, alphas_cumprod,
    VAE_SCALING_FACTOR, VAE_SCALE_FACTOR_SPATIAL, RoPE_BASE_HEIGHT, RoPE_BASE_WIDTH,
    middle_size, global_step,
):
    """Compute average validation loss over a few batches (no gradients, no backward)."""
    transformer.eval()
    total_loss = 0.0
    total_rgb_loss = 0.0
    total_depth_loss = 0.0
    num_batches = 0
    max_batches = args.validation_loss_num_batches

    for step, batch in enumerate(val_dataloader):
        if step >= max_batches:
            break
        images = batch["images"].to(accelerator.device, non_blocking=True)
        videos = batch["videos"].to(accelerator.device, non_blocking=True)
        prompts = batch["prompts"]

        images = images.permute(0, 2, 1, 3, 4)
        image_noise_sigma = torch.normal(
            mean=-3.0, std=0.5, size=(images.size(0),),
            device=accelerator.device, dtype=weight_dtype)
        image_noise_sigma = torch.exp(image_noise_sigma)
        noisy_images = images + torch.randn_like(images) * image_noise_sigma[:, None, None, None, None]
        B, C, F, H, W = noisy_images.shape
        noisy_images = noisy_images[:, : C // 3, :, :, :]
        image_latent_dist = vae.encode(noisy_images).latent_dist

        videos = videos.permute(0, 2, 1, 3, 4)
        videos = videos[:, : C // 3, :, :, :]
        _, _, frame, _, _ = videos.shape
        latent_dist = vae.encode(videos).latent_dist

        # See training loop for why the .mul(0.5).add(0.5) is needed: videos come
        # from the dataset in VAE-ready [-1, 1] but DA3 wants [0, 1] and its
        # processor otherwise clamps the negative half to zero.
        da3_input = videos.permute(0, 2, 1, 3, 4)[:1].mul(0.5).add(0.5).clamp(0.0, 1.0)
        agg_features = da3_block.pre_forward_tensor(da3_input)
        agg_features = agg_features.permute(1, 2, 3, 4, 0).reshape(
            1, frame, middle_size, middle_size, 1024).permute(0, 4, 1, 2, 3)
        proj_features = transformer.downsample(agg_features, (H // 8, W // 8))
        middle_geo_features = transformer.encoder(proj_features)
        proj_features = proj_features.permute(0, 2, 1, 3, 4)
        middle_geo_features = middle_geo_features.permute(0, 2, 1, 3, 4)
        middle_geo_features = middle_geo_features.to(
            memory_format=torch.contiguous_format, dtype=weight_dtype)
        middle_geo_features = middle_geo_features * VAE_SCALING_FACTOR

        image_latent_dist = image_latent_dist.sample()
        B, C, F, H, W = image_latent_dist.shape
        image_latents = image_latent_dist.reshape(B, C, F, H, W)
        image_latents = image_latents * VAE_SCALING_FACTOR
        image_latents = image_latents.permute(0, 2, 1, 3, 4)
        image_latents = image_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)

        latent_dist = latent_dist.sample()
        latent_dist = latent_dist.reshape(B, C, -1, H, W)
        video_latents = latent_dist * VAE_SCALING_FACTOR
        video_latents = video_latents.permute(0, 2, 1, 3, 4)
        video_latents = video_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)

        padding_shape = (video_latents.shape[0],
                         video_latents.shape[1] - 1, *video_latents.shape[2:])
        latent_padding = image_latents.new_zeros(padding_shape)
        image_latents = torch.cat([image_latents, latent_padding], dim=1)

        if not args.load_tensors:
            prompt_embeds = compute_prompt_embeddings(
                tokenizer, text_encoder, prompts,
                model_config.max_text_seq_length,
                accelerator.device, weight_dtype, requires_grad=False)
        else:
            prompt_embeds = prompts.to(dtype=weight_dtype)

        batch_size, num_frames, num_channels, height, width = video_latents.shape
        # Deterministic validation timesteps (fixed per val batch index) so the
        # validation loss has low variance and is comparable across steps/runs.
        _T = scheduler.config.num_train_timesteps
        _tt = int((step + 0.5) / max(1, max_batches) * _T)
        timesteps = torch.full((batch_size,), _tt, dtype=torch.int64, device=accelerator.device)

        image_rotary_emb = (
            prepare_rotary_positional_embeddings(
                height=height * VAE_SCALE_FACTOR_SPATIAL,
                width=width * VAE_SCALE_FACTOR_SPATIAL,
                num_frames=num_frames,
                vae_scale_factor_spatial=VAE_SCALE_FACTOR_SPATIAL,
                patch_size=model_config.patch_size,
                patch_size_t=(model_config.patch_size_t if hasattr(model_config, "patch_size_t") else None),
                attention_head_dim=model_config.attention_head_dim,
                device=accelerator.device,
                base_height=RoPE_BASE_HEIGHT,
                base_width=RoPE_BASE_WIDTH,
            ) if model_config.use_rotary_positional_embeddings else None)

        # Validation can present a different frame count than training, so the DA3
        # geo features may not match video_latents on (B,F,H,W). Force-align all
        # non-channel dims (channels are the concat dim) before concatenating.
        if middle_geo_features.shape[:2] + middle_geo_features.shape[3:] != \
                video_latents.shape[:2] + video_latents.shape[3:]:
            logger.warning(f"[val geo-align] video_latents={tuple(video_latents.shape)} "
                           f"middle_geo={tuple(middle_geo_features.shape)}")
            _mgf = middle_geo_features.permute(0, 2, 1, 3, 4)  # -> B,C,F,H,W
            if _mgf.shape[0] != video_latents.shape[0]:
                _mgf = _mgf[: video_latents.shape[0]] if _mgf.shape[0] > video_latents.shape[0] \
                    else _mgf.expand(video_latents.shape[0], -1, -1, -1, -1)
            _mgf = torch.nn.functional.interpolate(
                _mgf.float(),
                size=(video_latents.shape[1], video_latents.shape[3], video_latents.shape[4]),
                mode="nearest").to(_mgf.dtype)
            middle_geo_features = _mgf.permute(0, 2, 1, 3, 4).contiguous()
        video_latents = torch.cat([video_latents, middle_geo_features], dim=2)
        noise = torch.randn_like(video_latents)
        if (os.environ.get("REF15", "") == "1" or os.environ.get("REF13", "") == "1") \
                and os.environ.get("GEO_COUPLED_NOISE", "1") == "1":   # Task-48: configurable
            _c0v = video_latents.shape[2] // 2; noise[:, :, _c0v:] = noise[:, :, :_c0v]  # coupled (default)
        noisy_video_latents = scheduler.add_noise(video_latents, noise, timesteps)
        _, _, C_noisy, _, _ = noisy_video_latents.shape

        image_latents_rgb = image_latents.clone()
        noisy_video_latents_rgb = noisy_video_latents[:, :, : C_noisy // 2].clone()
        noisy_video_latents_depth = noisy_video_latents[:, :, C_noisy // 2 : C_noisy // 2 * 2].clone()
        noisy_image_input = torch.cat([noisy_video_latents_rgb, image_latents_rgb], dim=2)

        ofs_embed_dim = ((model_config.ofs_embed_dim if hasattr(model_config, "ofs_embed_dim") else None),)
        ofs_emb = None if ofs_embed_dim is None else noisy_image_input.new_full((1,), fill_value=2.0)

        outputs = transformer(
            hidden_states=noisy_image_input,
            encoder_hidden_states=prompt_embeds,
            timestep=timesteps,
            ofs=ofs_emb,
            image_rotary_emb=image_rotary_emb,
            return_dict=False)
        rgb_output = outputs[0]
        hidden_states = outputs[1]
        rgb_pred = scheduler.get_velocity(rgb_output, noisy_video_latents_rgb, timesteps)
        weights = 1 / (1 - alphas_cumprod[timesteps])
        while len(weights.shape) < len(rgb_pred.shape):
            weights = weights.unsqueeze(-1)

        target = video_latents
        rgb_target, depth_target = target.chunk(2, dim=2)

        # task10 (23-04): mirrors the training change — 16-ch v-pred loss,
        # decoder no longer called.
        rgb_loss = loss_fn(rgb_pred, rgb_target, weights)
        if _rgb_only_enabled():
            # Task-27-10: no geometry branch -> val loss is the RGB term alone.
            depth_loss = rgb_loss.new_zeros(())
        elif os.environ.get("REF15", "") == "1":
            # ref15 val/geo_loss: masked Geo-DiT predicts the full geometry velocity (matches training).
            _acv = alphas_cumprod[timesteps].view(-1, 1, 1, 1, 1).to(rgb_output.dtype)
            _alv = _acv.sqrt(); _sgv = (1 - _acv).sqrt()
            _eps16v = noise[:, :, : video_latents.shape[2] // 2].to(rgb_output.dtype)
            _depth_target_v = _alv * _eps16v - _sgv * middle_geo_features.to(rgb_output.dtype)
            _erhv = (_sgv * noisy_video_latents_rgb + _alv * rgb_pred).detach()
            depth_pred = transformer.geo_dit(hidden_states, timesteps, (num_frames, height, width),
                                             'masked', z_t_geo=noisy_video_latents_depth, eps_rgb_hat=_erhv)
            depth_loss = ((depth_pred - _depth_target_v) ** 2).mean()
        else:
            depth_output = transformer.forward_depth(noisy_video_latents_depth, transformer._depth_feat(hidden_states), rgb_output)[0]
            depth_pred = scheduler.get_velocity(depth_output, noisy_video_latents_depth, timesteps)
            depth_loss = loss_fn(depth_pred, depth_target, weights)
        depth_scale = min(1.0, global_step / float(os.environ.get("GEO_WARMUP", "100")))
        loss = rgb_loss + depth_loss * 0.3 * depth_scale

        total_loss += loss.item()
        total_rgb_loss += rgb_loss.item()
        total_depth_loss += depth_loss.item()
        num_batches += 1

    transformer.train()

    if num_batches == 0:
        return {}
    return {
        "val/loss": total_loss / num_batches,
        "val/rgb_loss": total_rgb_loss / num_batches,
        "val/depth_loss": total_depth_loss / num_batches,
    }


def run_geo_eval(
    args, accelerator, transformer, vae, text_encoder, tokenizer, scheduler,
    val_dataloader, da3_block, weight_dtype, model_config, alphas_cumprod,
    VAE_SCALING_FACTOR, VAE_SCALE_FACTOR_SPATIAL, RoPE_BASE_HEIGHT, RoPE_BASE_WIDTH,
    middle_size, global_step,
):
    """Task-53 geometry evaluation (the CORRECT, trained module).

    The standard ref15 inference visualizes ``transformer.forward_depth`` — a module that gets NO
    gradient in ref15 (REF15 trains ``transformer.geo_dit``). This routine instead evaluates the
    actually-trained ``geo_dit``: it reconstructs the clean geometry feature from the predicted
    geometry VELOCITY and compares it, in the SAME 16-ch representation, against the GT target
    ``z0_geo = encoder(downsample(DA3(gt)))·VAE_scale``.

      target velocity:  v = alpha*eps - sigma*z0_geo     (i2v_depth_ref9.py training, :1034)
      => reconstruction: z0_pred = (alpha*eps - v_pred) / sigma     (eps coupled = eps_rgb)

    Saves per-sample z0_geo / z0_pred tensors (.npy, [F,16,h,w]) + geo_metrics.json (per-sample and
    mean MSE / cosine). Offline geo_render.py turns the .npy into side-by-side videos. Gated by
    GEO_EVAL=1 so other tasks are unaffected. Knobs: GEO_EVAL_T (eval timestep, default 100),
    GEO_EVAL_MASK (token mask ratio, default = training MASK_RATIO).
    """
    import json as _json
    transformer.eval()
    out_dir = os.path.join(args.output_dir, "geo_eval")
    os.makedirs(out_dir, exist_ok=True)
    eval_ts = [int(x) for x in os.environ.get("GEO_EVAL_T", "100,300,500,700").split(",")]
    head_t = eval_ts[0]                       # headline timestep (saved tensors/videos)
    mask_ratio = float(os.environ.get("GEO_EVAL_MASK", os.environ.get("MASK_RATIO", "0.5")))
    results = []
    for step, batch in enumerate(val_dataloader):
        images = batch["images"].to(accelerator.device, non_blocking=True)
        videos = batch["videos"].to(accelerator.device, non_blocking=True)
        prompts = batch["prompts"]
        images = images.permute(0, 2, 1, 3, 4)
        image_noise_sigma = torch.exp(torch.normal(mean=-3.0, std=0.5, size=(images.size(0),),
                                                    device=accelerator.device, dtype=weight_dtype))
        noisy_images = images + torch.randn_like(images) * image_noise_sigma[:, None, None, None, None]
        B, C, F, H, W = noisy_images.shape
        noisy_images = noisy_images[:, : C // 3, :, :, :]
        with torch.no_grad():
            image_latent_dist = vae.encode(noisy_images).latent_dist
            videos = videos.permute(0, 2, 1, 3, 4)
            videos = videos[:, : C // 3, :, :, :]
            _, _, frame, _, _ = videos.shape
            latent_dist = vae.encode(videos).latent_dist
            da3_input = videos.permute(0, 2, 1, 3, 4)[:1].mul(0.5).add(0.5).clamp(0.0, 1.0)
            agg_features = da3_block.pre_forward_tensor(da3_input)
            agg_features = agg_features.permute(1, 2, 3, 4, 0).reshape(
                1, frame, middle_size, middle_size, 1024).permute(0, 4, 1, 2, 3)
            proj_features = transformer.downsample(agg_features, (H // 8, W // 8))
            middle_geo_features = transformer.encoder(proj_features)
            middle_geo_features = middle_geo_features.permute(0, 2, 1, 3, 4).to(
                memory_format=torch.contiguous_format, dtype=weight_dtype) * VAE_SCALING_FACTOR
            image_latent_dist = image_latent_dist.sample()
            B, C, F, H, W = image_latent_dist.shape
            image_latents = (image_latent_dist.reshape(B, C, F, H, W) * VAE_SCALING_FACTOR
                             ).permute(0, 2, 1, 3, 4).to(memory_format=torch.contiguous_format, dtype=weight_dtype)
            latent_dist = latent_dist.sample()
            latent_dist = latent_dist.reshape(B, C, -1, H, W)
            video_latents = (latent_dist * VAE_SCALING_FACTOR).permute(0, 2, 1, 3, 4).to(
                memory_format=torch.contiguous_format, dtype=weight_dtype)
            padding_shape = (video_latents.shape[0], video_latents.shape[1] - 1, *video_latents.shape[2:])
            image_latents = torch.cat([image_latents, image_latents.new_zeros(padding_shape)], dim=1)
            prompt_embeds = (compute_prompt_embeddings(
                tokenizer, text_encoder, prompts, model_config.max_text_seq_length,
                accelerator.device, weight_dtype, requires_grad=False)
                if not args.load_tensors else prompts.to(dtype=weight_dtype))
            batch_size, num_frames, num_channels, height, width = video_latents.shape
            timesteps = torch.full((batch_size,), head_t, dtype=torch.int64, device=accelerator.device)
            image_rotary_emb = (prepare_rotary_positional_embeddings(
                height=height * VAE_SCALE_FACTOR_SPATIAL, width=width * VAE_SCALE_FACTOR_SPATIAL,
                num_frames=num_frames, vae_scale_factor_spatial=VAE_SCALE_FACTOR_SPATIAL,
                patch_size=model_config.patch_size,
                patch_size_t=(model_config.patch_size_t if hasattr(model_config, "patch_size_t") else None),
                attention_head_dim=model_config.attention_head_dim, device=accelerator.device,
                base_height=RoPE_BASE_HEIGHT, base_width=RoPE_BASE_WIDTH,
            ) if model_config.use_rotary_positional_embeddings else None)
            # align geo features to latent grid if frame counts differ (mirrors compute_validation_loss)
            if middle_geo_features.shape[:2] + middle_geo_features.shape[3:] != \
                    video_latents.shape[:2] + video_latents.shape[3:]:
                _mgf = middle_geo_features.permute(0, 2, 1, 3, 4)
                if _mgf.shape[0] != video_latents.shape[0]:
                    _mgf = _mgf[: video_latents.shape[0]] if _mgf.shape[0] > video_latents.shape[0] \
                        else _mgf.expand(video_latents.shape[0], -1, -1, -1, -1)
                _mgf = torch.nn.functional.interpolate(
                    _mgf.float(), size=(video_latents.shape[1], video_latents.shape[3], video_latents.shape[4]),
                    mode="nearest").to(_mgf.dtype)
                middle_geo_features = _mgf.permute(0, 2, 1, 3, 4).contiguous()
            video_latents_cat = torch.cat([video_latents, middle_geo_features], dim=2)
            noise = torch.randn_like(video_latents_cat)
            _c0 = video_latents_cat.shape[2] // 2
            noise[:, :, _c0:] = noise[:, :, :_c0]                      # coupled eps_geo = eps_rgb (match training)
            z0_geo = middle_geo_features.to(weight_dtype)
            by_t = {}; z0_pred_head = None
            # Evaluate the TRAINED geo_dit at several timesteps. At each t we do one denoise step and
            # reconstruct the clean geometry feature with the STABLE v-pred->x0 formula
            #   z0 = alpha*z_t - sigma*v_pred   (NOT (alpha*eps - v)/sigma, which blows up ~1/sigma).
            for _ti, _t in enumerate(eval_ts):
                timesteps = torch.full((batch_size,), _t, dtype=torch.int64, device=accelerator.device)
                noisy = scheduler.add_noise(video_latents_cat, noise, timesteps)
                _, _, C_noisy, _, _ = noisy.shape
                nz_rgb = noisy[:, :, : C_noisy // 2].clone()
                nz_dep = noisy[:, :, C_noisy // 2 : C_noisy // 2 * 2].clone()
                nimg = torch.cat([nz_rgb, image_latents.clone()], dim=2)
                ofs_emb = nimg.new_full((1,), 2.0) if hasattr(model_config, "ofs_embed_dim") else None
                # feed geo_dit the MIDPOINT hook (h_rgb), exactly like training (:1028/:1047).
                if getattr(transformer, "_geo_hook_cache", None) is not None:
                    transformer._geo_hook_cache.clear()
                outputs = transformer(hidden_states=nimg, encoder_hidden_states=prompt_embeds,
                                      timestep=timesteps, ofs=ofs_emb, image_rotary_emb=image_rotary_emb,
                                      return_dict=False)
                rgb_output, hidden_states = outputs[0], outputs[1]
                _hrgb = (transformer._geo_hook_cache[-1]
                         if getattr(transformer, "_geo_hook_cache", None) else hidden_states)
                rgb_pred = scheduler.get_velocity(rgb_output, nz_rgb, timesteps)
                _ac = alphas_cumprod[timesteps].view(-1, 1, 1, 1, 1).to(rgb_output.dtype)
                _al, _sg = _ac.sqrt(), (1 - _ac).sqrt()
                eps_rgb_hat = (_sg * nz_rgb + _al * rgb_pred).detach()
                _saved_mr = getattr(transformer.geo_dit, "mask_ratio", None)
                if _saved_mr is not None:
                    transformer.geo_dit.mask_ratio = mask_ratio
                v_pred = transformer.geo_dit(_hrgb, timesteps, (num_frames, height, width), 'masked',
                                             z_t_geo=nz_dep, eps_rgb_hat=eps_rgb_hat)
                if _saved_mr is not None:
                    transformer.geo_dit.mask_ratio = _saved_mr
                if os.environ.get("GEO_X0", "0") == "1":
                    # x0 mode: geo_dit output IS the geometry feature z0 directly (no velocity).
                    z0_pred = v_pred.to(rgb_output.dtype)
                    g_pred = z0_pred.float().flatten()
                    z0_pred_geom = z0_pred
                else:
                    z0_pred = _al * nz_dep - _sg * v_pred.to(rgb_output.dtype)   # stable v-pred -> x0 (1-step denoise)
                    # LEAKAGE-FREE geometry metric: model's predicted geometry residual
                    #   g_pred = alpha*eps_hat - v_pred  (~ sigma*z0_geo if learned). No z_t, so (esp. at
                    # mask=1.0) it reflects ONLY the model — unlike the 1-step z0 above which is dominated
                    # by alpha*z_t (z0 leakage).
                    g_pred = (_al * eps_rgb_hat - v_pred.to(rgb_output.dtype)).float().flatten()
                    z0_pred_geom = (_al * eps_rgb_hat - v_pred.to(rgb_output.dtype)) / (_sg + 1e-8)
                a = z0_pred.float().flatten(); b = z0_geo.float().flatten()
                cg = z0_pred_geom.float().flatten()
                by_t[_t] = {"timestep": _t,
                            "sigma": float(_sg.flatten()[0].item()),
                            "mse": torch.mean((a - b) ** 2).item(),
                            "cosine": torch.nn.functional.cosine_similarity(a, b, dim=0).item(),
                            "rel_l2": (a - b).norm().item() / (b.norm().item() + 1e-8),
                            # leakage-free geometry-prediction metrics:
                            "cos_geom": torch.nn.functional.cosine_similarity(g_pred, b, dim=0).item(),
                            "mse_geom": torch.mean((cg - b) ** 2).item()}
                if _ti == 0:
                    z0_pred_head = z0_pred
        np.save(os.path.join(out_dir, f"sample{step:02d}_z0geo.npy"), z0_geo[0].detach().float().cpu().numpy())
        np.save(os.path.join(out_dir, f"sample{step:02d}_z0pred.npy"), z0_pred_head[0].detach().float().cpu().numpy())
        # Task-64: standardized Task-62 layout — <VAL_VIZ_DIR>/Step_NNNNNN/geo_sample{step}.mp4
        # (GT geometry feature PCA->RGB | predicted feature PCA->RGB). z0_geo / z0_pred_head are
        # [T_latent, 16, h, w] (channels=16 can exceed latent-time) -> pass layout="TCHW" so the
        # shared helper orients axes correctly. Env-gated (default off); other tasks unaffected.
        _vvd = os.environ.get("VAL_VIZ_DIR", "")
        if _vvd:
            try:
                import sys as _sys
                _tools = os.path.join(os.environ.get("PROJECT_ROOT",
                    str(Path(__file__).resolve().parents[2])), "tools")
                if _tools not in _sys.path:
                    _sys.path.insert(0, _tools)
                import val_viz as _val_viz
                _val_viz.save_geo_sample(
                    _vvd, global_step, step,
                    z0_geo[0].detach().float().cpu().numpy(),
                    z0_pred_head[0].detach().float().cpu().numpy(),
                    layout="TCHW")
            except Exception as _e:
                accelerator.print(f"[val_viz] geo_sample save failed (sample {step}): {_e}")
        rec = {"sample": step, "prompt": (prompts[0][:60] if isinstance(prompts, (list, tuple)) else str(prompts)[:60]),
               "head_t": head_t, "mask_ratio": mask_ratio, "target_std": float(z0_geo.float().std().item()),
               "by_t": by_t,
               "mse": by_t[head_t]["mse"], "cosine": by_t[head_t]["cosine"], "rel_l2": by_t[head_t]["rel_l2"]}
        results.append(rec)
        _tsum = "  ".join(f"t{t}:cosG={by_t[t]['cos_geom']:.3f}/rec={by_t[t]['cosine']:.3f}" for t in eval_ts)
        accelerator.print(f"[GEO_EVAL] sample {step}: {_tsum}  target_std={rec['target_std']:.3f}")
    transformer.train()
    if results:
        # mean metrics per timestep across samples
        mean_by_t = {}
        for t in eval_ts:
            mean_by_t[t] = {"timestep": t,
                            "mean_cos_geom": float(np.mean([r["by_t"][t]["cos_geom"] for r in results])),
                            "mean_mse_geom": float(np.mean([r["by_t"][t]["mse_geom"] for r in results])),
                            "mean_cosine_recon": float(np.mean([r["by_t"][t]["cosine"] for r in results])),
                            "mean_mse": float(np.mean([r["by_t"][t]["mse"] for r in results])),
                            "mean_rel_l2": float(np.mean([r["by_t"][t]["rel_l2"] for r in results])),
                            "sigma": results[0]["by_t"][t]["sigma"]}
        agg = {"n": len(results), "eval_timesteps": eval_ts, "head_t": head_t, "mask_ratio": mask_ratio,
               "mean_cos_geom": mean_by_t[head_t]["mean_cos_geom"],
               "mean_cosine_recon": mean_by_t[head_t]["mean_cosine_recon"],
               "mean_mse": mean_by_t[head_t]["mean_mse"], "mean_rel_l2": mean_by_t[head_t]["mean_rel_l2"],
               "mean_by_timestep": mean_by_t, "per_sample": results}
        with open(os.path.join(out_dir, "geo_metrics.json"), "w") as f:
            _json.dump(agg, f, indent=2)
        accelerator.print("[GEO_EVAL] MEAN cos_geom(leakage-free) by t: " +
                          "  ".join(f"t{t}={mean_by_t[t]['mean_cos_geom']:.4f}" for t in eval_ts))
        accelerator.print("[GEO_EVAL] MEAN cosine_recon(1-step) by t: " +
                          "  ".join(f"t{t}={mean_by_t[t]['mean_cosine_recon']:.4f}(σ={mean_by_t[t]['sigma']:.2f})" for t in eval_ts))
        accelerator.print(f"[GEO_EVAL] -> {out_dir}/geo_metrics.json")
    return results


def main(args):
    if args.seed is not None:
        set_seed(args.seed)
    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        raise ValueError("Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead.")

    logging_dir = Path(args.output_dir, args.logging_dir)
    logging_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,)
    file_handler = logging.FileHandler(logging_dir / "train.log")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    logging.getLogger().addHandler(file_handler)
    logger = get_logger(__name__)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    init_process_group_kwargs = InitProcessGroupKwargs(backend="nccl", timeout=timedelta(seconds=args.nccl_timeout))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs, init_process_group_kwargs],)
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    logger.info(accelerator.state, main_process_only=False)
    # 设置 transformers 和 diffusers 的日志级别
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    # Prepare models and scheduler
    # Base model path for tokenizer/text_encoder/vae/scheduler. Falls back to
    # the HF hub ID when a local CogVideoX-5b-I2V directory is not provided.
    _base_model = os.environ.get(
        "COGVIDEO_BASE_PATH", args.pretrained_model_name_or_path
    )
    # If pretrained_model_name_or_path is a fine-tuned checkpoint (no tokenizer/
    # vae/scheduler subfolders), fall back to the HF hub ID.
    _base_has_subs = os.path.isdir(os.path.join(_base_model, "tokenizer"))
    if not _base_has_subs:
        _base_model = os.environ.get("COGVIDEO_BASE_PATH", "THUDM/CogVideoX-5b-I2V")

    tokenizer = AutoTokenizer.from_pretrained(
        _base_model,
        subfolder="tokenizer", revision=args.revision,)
    text_encoder = T5EncoderModel.from_pretrained(
        _base_model,
        subfolder="text_encoder", revision=args.revision,)
    # CogVideoX-2b weights are stored in float16
    # CogVideoX-5b and CogVideoX-5b-I2V weights are stored in bfloat16
    load_dtype = torch.bfloat16  # CogVideoX-5b weights are bf16 (old heuristic keyed on "5b" in the path)
    _model_has_transformer_sub = os.path.isdir(
        os.path.join(args.pretrained_model_name_or_path, "transformer")
    )
    if _model_has_transformer_sub:
        transformer = CogVideoXTransformer3DModel.from_pretrained_modify(
            args.pretrained_model_name_or_path, subfolder="transformer",
            torch_dtype=load_dtype, revision=args.revision, variant=args.variant,)
    else:
        transformer = CogVideoXTransformer3DModel.from_pretrained_modify(
            args.pretrained_model_name_or_path, subfolder=None,
            torch_dtype=load_dtype, revision=args.revision, variant=args.variant,)
    if args.ignore_learned_positional_embeddings:
        del transformer.patch_embed.pos_embedding
        transformer.patch_embed.use_learned_positional_embeddings = False
        transformer.config.use_learned_positional_embeddings = False
    vae = AutoencoderKLCogVideoX.from_pretrained(
        _base_model,
        subfolder="vae", revision=args.revision, variant=args.variant,)
    scheduler = CogVideoXDPMScheduler.from_pretrained(_base_model, subfolder="scheduler")
    if args.enable_slicing: vae.enable_slicing()
    if args.enable_tiling: vae.enable_tiling()

    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    transformer.requires_grad_(True)
    # task10 (23-04): freeze the geo encoder (self.project, 256→16). The depth
    # loss is now computed in 16-ch v-pred space against `depth_target` directly,
    # so the project head no longer needs to adapt — and joint AE training
    # (project + reconstruct) was the source of the wired/asymmetric supervision.
    for p in transformer.project.parameters():
        p.requires_grad = False
    # TOY_FREEZE: for memory-constrained toy runs, freeze the bulk of the backbone
    # and train only the I/O layers + last-N blocks + depth machinery (symmetric with
    # ref7). Cuts grad + optimizer memory so a 5B model fits a contended GPU.
    if os.environ.get("TOY_FREEZE", "0") == "1":
        _N = int(os.environ.get("TRAIN_LAST_N", "6"))
        _nb = len(transformer.transformer_blocks)
        _keep = ("patch_embed", "norm_out.", "proj_out.", "dn_out", "dn_out_proj",
                 "downsample", "encoder.")
        for _name, _p in transformer.named_parameters():
            _train = _name.startswith(_keep)
            if _name.startswith("transformer_blocks."):
                _idx = int(_name.split(".")[1])
                if _idx >= _nb - _N:
                    _train = True
            _p.requires_grad_(bool(_train))
        for _p in transformer.project.parameters():
            _p.requires_grad_(False)
    # === Task-19: RGB middle-feature taps + per-mode aggregator (AUX_MODE in A/B/C/D) ===
    AUX_MODE = os.environ.get("AUX_MODE", "").upper()
    transformer._tap_cache = []
    if AUX_MODE in ("A", "B", "C", "D"):
        from gem4d.modules.aux_aggregator import AuxAggregator
        _nb = len(transformer.transformer_blocks)
        _taps = [_nb // 2] if AUX_MODE == "A" else sorted({_nb // 4, _nb // 2, (3 * _nb) // 4})
        _dim = transformer.config.num_attention_heads * transformer.config.attention_head_dim
        _dt = next(transformer.parameters()).dtype
        def _mk_hook():
            def _hook(mod, inp, out):
                transformer._tap_cache.append(out[0])
            return _hook
        for _li in _taps:
            transformer.transformer_blocks[_li].register_forward_hook(_mk_hook())
        transformer.aux_aggregator = AuxAggregator(AUX_MODE, _dim, len(_taps), a_index=0).to(
            accelerator.device, dtype=_dt)
        if accelerator.is_main_process:
            print(f"[AUX] mode={AUX_MODE} taps={_taps} dim={_dim} agg_params="
                  f"{sum(p.numel() for p in transformer.aux_aggregator.parameters())}")
    def _depth_feat(hidden_states):
        if AUX_MODE in ("A", "B", "C", "D"):
            n = transformer.aux_aggregator.n_taps           # robust to ckpt-recompute re-firing hooks
            taps = list(transformer._tap_cache)[-n:]
            transformer._tap_cache.clear()
            return transformer.aux_aggregator(taps, hidden_states)
        return hidden_states
    transformer._depth_feat = _depth_feat
    # === Task-26 ref_13 / CN-MIGV: auxiliary Geo-DiT (coupled noise, empty/present mixing) ===
    # === Task-38 ref_15: ref13 + masked geometry latent (70% random token masking) ===
    RGB_ONLY = _rgb_only_enabled()                        # Task-27-10: no geometry branch at all
    REF15 = os.environ.get("REF15", "") == "1" and not RGB_ONLY
    REF13 = (os.environ.get("REF13", "") == "1" or REF15) and not RGB_ONLY   # ref15 builds on ref13 (coupling + geo_dit)
    if REF13:
        from gem4d.modules.geo_dit import GeoDiT
        _dim13 = transformer.config.num_attention_heads * transformer.config.attention_head_dim
        transformer.geo_dit = GeoDiT(_dim13, heads=transformer.config.num_attention_heads,
                                     depth=int(os.environ.get("GEO_DEPTH", "4")), in_ch=16,
                                     patch=transformer.config.patch_size).to(
                                     accelerator.device, dtype=next(transformer.parameters()).dtype)
        if accelerator.is_main_process:
            print(f"[REF13] GeoDiT dim={_dim13} depth={os.environ.get('GEO_DEPTH','4')} "
                  f"params={sum(p.numel() for p in transformer.geo_dit.parameters())} "
                  f"p_empty={os.environ.get('P_EMPTY','0.8')} lambda_geo={os.environ.get('LAMBDA_GEO','0.03')}")
        # DIAG: load TRAINED geo_dit weights from a saved checkpoint dir (for the
        # velocity-sensitivity diagnostic; from_pretrained ignores geo_dit.* keys).
        _ckpt_geo = os.environ.get("CKPT_GEO", "")
        if _ckpt_geo:
            import glob as _glob
            from safetensors.torch import load_file as _load_file
            _sd = {}
            for _f in sorted(_glob.glob(os.path.join(_ckpt_geo, "*.safetensors"))):
                for _k, _v in _load_file(_f).items():
                    if _k.startswith("geo_dit."):
                        _sd[_k[len("geo_dit."):]] = _v
            _miss, _unexp = transformer.geo_dit.load_state_dict(_sd, strict=False)
            transformer.geo_dit.to(accelerator.device, dtype=next(transformer.parameters()).dtype)
            if accelerator.is_main_process:
                print(f"[DIAG] loaded geo_dit from {_ckpt_geo}: {len(_sd)} keys, "
                      f"missing={len(_miss)} unexpected={len(_unexp)}")
    # === Task-32: Geo-DiT cross-attn context h_rgb from a MIDPOINT RGB-DiT block ===
    # Symmetric with Cosmos (geo_feature_block = num_blocks//2). Previously the Geo-DiT read the
    # FINAL trunk hidden state (outputs[1]); GEO_HOOK_LAYER taps an intermediate block instead, so
    # the geometry gradient is injected mid-backbone (blocks <= GEO_HOOK_LAYER). The forward hook
    # captures the block's output in the SAME forward pass (no extra compute); it stays in the graph
    # so gradients flow back (matches the non-detached final hidden_states it replaces). Default =
    # num_blocks//2 (=21 for CogVideoX-5b's 42 blocks). GEO_HOOK_LAYER=-1 -> legacy final-layer.
    transformer._geo_hook_cache = []
    if REF13:
        _geo_nb = len(transformer.transformer_blocks)
        GEO_HOOK_LAYER = int(os.environ.get("GEO_HOOK_LAYER", str(_geo_nb // 2)))
        transformer._geo_hook_layer = GEO_HOOK_LAYER
        if 0 <= GEO_HOOK_LAYER < _geo_nb:
            def _mk_geo_hook():
                def _hook(mod, inp, out):
                    transformer._geo_hook_cache.append(out[0])
                return _hook
            transformer.transformer_blocks[GEO_HOOK_LAYER].register_forward_hook(_mk_geo_hook())
            if accelerator.is_main_process:
                print(f"[Task-32] Geo-DiT h_rgb <- MIDPOINT block {GEO_HOOK_LAYER}/{_geo_nb} "
                      f"(was final-layer outputs[1])")
        elif accelerator.is_main_process:
            print(f"[Task-32] GEO_HOOK_LAYER={GEO_HOOK_LAYER} disabled -> Geo-DiT uses final hidden_states")
    VAE_SCALING_FACTOR = vae.config.scaling_factor
    VAE_SCALE_FACTOR_SPATIAL = 2 ** (len(vae.config.block_out_channels) - 1)
    RoPE_BASE_HEIGHT = transformer.config.sample_height * VAE_SCALE_FACTOR_SPATIAL
    RoPE_BASE_WIDTH = transformer.config.sample_width * VAE_SCALE_FACTOR_SPATIAL
    # --------------- 加在 checkpoint ---------------
    checkpoint_manager = CHECKPOINT_MANAGER(args)
    print("=================================================transformer1=================================================")
    # Task-67: simplified full-state checkpoint dir (<output>/checkpoints) with step_NNNNNN + `latest`
    # and auto-resume. Gated by SIMPLE_CKPT_DIR; when unset, behaviour is exactly as before.
    _simple_ckpt_dir = os.environ.get("SIMPLE_CKPT_DIR", "")
    _simple_resume_state = None
    global_step = 0
    if _simple_ckpt_dir:
        import sys as _sys
        _tools = os.path.join(os.environ.get("PROJECT_ROOT", str(Path(__file__).resolve().parents[2])), "tools")
        if _tools not in _sys.path:
            _sys.path.insert(0, _tools)
        import overfit_ckpt as _ofc
        _latest = _ofc.find_latest(_simple_ckpt_dir)
        if _latest is not None:
            _lstep, _lpath = _latest
            _load_dtype = torch.bfloat16  # CogVideoX-5b weights are bf16 (old heuristic keyed on "5b" in the path)
            accelerator.print(f"[simple-ckpt] auto-resume from {_lpath} (step {_lstep})")
            # Preserve the programmatically-attached geo_dit (ref13/ref15): from_pretrained_modify rebuilds a
            # bare CogVideoX from config and would drop it (and break the optimizer param set). Re-attach the
            # freshly-built module and load its trained weights from the checkpoint safetensors (geo_dit.* keys).
            _geo = getattr(transformer, "geo_dit", None)
            transformer = transformer.from_pretrained_modify(
                os.path.join(_lpath, "transformer"), subfolder=None,
                torch_dtype=_load_dtype, revision=args.revision, variant=args.variant,)
            transformer.patch_embed.use_learned_positional_embeddings = False
            if _geo is not None:
                transformer.geo_dit = _geo
                import glob as _glob
                from safetensors.torch import load_file as _load_file
                _gsd = {}
                for _f in sorted(_glob.glob(os.path.join(_lpath, "transformer", "*.safetensors"))):
                    for _k, _v in _load_file(_f).items():
                        if _k.startswith("geo_dit."):
                            _gsd[_k[len("geo_dit."):]] = _v
                if _gsd:
                    _gm, _gu = transformer.geo_dit.load_state_dict(_gsd, strict=False)
                    accelerator.print(f"[simple-ckpt] geo_dit weights loaded ({len(_gsd)} tensors, "
                                      f"missing={len(_gm)} unexpected={len(_gu)})")
            # Re-apply the SAME freezing as the fresh path (lines ~682-705) so the trainable-param set
            # (and thus the optimizer param group) matches the saved state exactly on resume.
            transformer.requires_grad_(True)
            for _p in transformer.project.parameters():
                _p.requires_grad = False
            if os.environ.get("TOY_FREEZE", "0") == "1":
                _N = int(os.environ.get("TRAIN_LAST_N", "6"))
                _nb = len(transformer.transformer_blocks)
                _keep = ("patch_embed", "norm_out.", "proj_out.", "dn_out", "dn_out_proj", "downsample", "encoder.")
                for _name, _p in transformer.named_parameters():
                    _train = _name.startswith(_keep)
                    if _name.startswith("transformer_blocks."):
                        if int(_name.split(".")[1]) >= _nb - _N:
                            _train = True
                    _p.requires_grad_(bool(_train))
                for _p in transformer.project.parameters():
                    _p.requires_grad_(False)
            global_step = _lstep
            _simple_resume_state = os.path.join(_lpath, "train_state.pt")
        else:
            accelerator.print(f"[simple-ckpt] no checkpoint under {_simple_ckpt_dir}; starting fresh")
    else:
        # compute_weight_sum_safe(transformer)
        global_step, transformer = checkpoint_manager.get_initial_global_step_model(accelerator, transformer)
    resumed_global_step = global_step
    # compute_weight_sum_safe(transformer)
    print("=================================================transformer2=================================================")
    add_in_channels = 16 * 4
    transformer.config.in_channels += add_in_channels
    if global_step > args.lr_warmup_steps:
        args.lr_warmup_steps = 0
    args.lr_warmup_steps = args.lr_warmup_steps * args.gradient_accumulation_steps
    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.state.deepspeed_plugin:
        if ("fp16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["fp16"]["enabled"]):
            weight_dtype = torch.float16
        if ("bf16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["bf16"]["enabled"]):
            weight_dtype = torch.bfloat16
    else:
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        raise ValueError("Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead.")
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    accelerator.register_save_state_pre_hook(make_save_model_hook(accelerator, transformer))

    # task10 (23-04): local load hook with strict=False so checkpoints saved
    # before/after the encoder-freeze + decoder-drop change cross-load cleanly
    # (e.g. resuming from a ckpt that had `reconstruct.*` weights into a model
    # that no longer trains them, or vice-versa). Otherwise mirrors
    # `utils.make_load_model_hook` — kept inline here so utils.py is untouched.
    def _make_load_model_hook_strict_false(accel, t, _args):
        def load_model_hook(models, input_dir):
            transformer_ = None
            init_under_meta = False
            if not accel.distributed_type == DistributedType.DEEPSPEED:
                while len(models) > 0:
                    model = models.pop()
                    if isinstance(unwrap_model(accel, model), type(unwrap_model(accel, t))):
                        transformer_ = unwrap_model(accel, model)
                    else:
                        raise ValueError(
                            f"Unexpected save model: {unwrap_model(accel, model).__class__}"
                        )
            else:
                with init_empty_weights():
                    transformer_ = CogVideoXTransformer3DModel.from_config(
                        _args.pretrained_model_name_or_path, subfolder="transformer")
                    init_under_meta = True
            load_model = CogVideoXTransformer3DModel.from_pretrained_modify(
                os.path.join(input_dir, "transformer"),
                map_dim=False, add_in_channels=0, add_out_channels=0,
            )
            if "I2V" in _args.pretrained_model_name_or_path:
                del load_model.patch_embed.pos_embedding
                load_model.patch_embed.use_learned_positional_embeddings = False
                load_model.config.use_learned_positional_embeddings = False
                if hasattr(transformer_.patch_embed, "pos_embedding"):
                    del transformer_.patch_embed.pos_embedding
                transformer_.patch_embed.use_learned_positional_embeddings = False
                transformer_.config.use_learned_positional_embeddings = False
            transformer_.register_to_config(**load_model.config)
            missing, unexpected = transformer_.load_state_dict(
                load_model.state_dict(), strict=False, assign=init_under_meta,
            )
            if accel.is_main_process:
                if missing:
                    print(f"[load_state_dict strict=False] missing keys ({len(missing)}): "
                          f"{missing[:8]}{'...' if len(missing) > 8 else ''}")
                if unexpected:
                    print(f"[load_state_dict strict=False] unexpected keys ({len(unexpected)}): "
                          f"{unexpected[:8]}{'...' if len(unexpected) > 8 else ''}")
            del load_model
            if _args.mixed_precision == "fp16":
                cast_training_params([transformer_])
        return load_model_hook

    accelerator.register_load_state_pre_hook(
        _make_load_model_hook_strict_false(accelerator, transformer, args))

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * \
                args.train_batch_size * accelerator.num_processes)
    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        cast_training_params([transformer], dtype=torch.float32)
    transformer_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    # Optimization parameters
    params_to_optimize = [{"params": transformer_parameters, "lr": args.learning_rate},]
    num_trainable_parameters = sum(param.numel() for model in params_to_optimize for param in model["params"])
    use_deepspeed_optimizer = (
        accelerator.state.deepspeed_plugin is not None
        and "optimizer" in accelerator.state.deepspeed_plugin.deepspeed_config)
    use_deepspeed_scheduler = (
        accelerator.state.deepspeed_plugin is not None
        and "scheduler" in accelerator.state.deepspeed_plugin.deepspeed_config)
    optimizer = get_optimizer(
        params_to_optimize=params_to_optimize,
        optimizer_name=args.optimizer,
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
        beta3=args.beta3,
        epsilon=args.epsilon,
        weight_decay=args.weight_decay,
        prodigy_decouple=args.prodigy_decouple,
        prodigy_use_bias_correction=args.prodigy_use_bias_correction,
        prodigy_safeguard_warmup=args.prodigy_safeguard_warmup,
        use_8bit=args.use_8bit,
        use_4bit=args.use_4bit,
        use_torchao=args.use_torchao,
        use_deepspeed=use_deepspeed_optimizer,
        use_cpu_offload_optimizer=args.use_cpu_offload_optimizer,
        offload_gradients=args.offload_gradients,)
    # Dataset and DataLoader
    dataset_init_kwargs = {
        "data_root": args.data_root,
        "dataset_file": args.dataset_file,
        "caption_column": args.caption_column,
        "video_column": args.video_column,
        "max_num_frames": args.max_num_frames,
        "id_token": args.id_token,
        "height_buckets": args.height_buckets,
        "width_buckets": args.width_buckets,
        "frame_buckets": args.frame_buckets,
        "load_tensors": args.load_tensors,
        "random_flip": args.random_flip,
        "height": args.height,
        "width": args.width,
        "image_to_video": True,}
    train_dataset = ManifestVideoDataset(**dataset_init_kwargs)
    collate_fn = CollateFunction(weight_dtype, args.load_tensors)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        sampler=BucketSampler(train_dataset, batch_size=args.train_batch_size, shuffle=True),
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
        pin_memory=args.pin_memory,)
    # Validation dataloader (for computing validation loss)
    val_dataloader = None
    if args.validation_dataset_file is not None:
        val_dataset_kwargs = dict(dataset_init_kwargs)
        val_dataset_kwargs["dataset_file"] = args.validation_dataset_file
        val_dataset_kwargs["random_flip"] = None  # no augmentation for val
        val_dataset = ManifestVideoDataset(**val_dataset_kwargs)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=1,
            sampler=BucketSampler(val_dataset, batch_size=args.train_batch_size, shuffle=False),
            collate_fn=collate_fn,
            num_workers=args.dataloader_num_workers,
            pin_memory=args.pin_memory,)
        accelerator.print(f"  Validation dataset: {len(val_dataset)} samples")
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    if args.use_cpu_offload_optimizer:
        lr_scheduler = None
        accelerator.print(
            "CPU Offload Optimizer cannot be used with DeepSpeed or builtin PyTorch LR Schedulers. If "
            "you are training with those settings, they will be ignored.")
    else:
        if use_deepspeed_scheduler:
            from accelerate.utils import DummyScheduler
            lr_scheduler = DummyScheduler(
                name=args.lr_scheduler,
                optimizer=optimizer,
                total_num_steps=args.max_train_steps * accelerator.num_processes,
                num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,)
        else:
            lr_scheduler = get_scheduler(
                args.lr_scheduler,
                optimizer=optimizer,
                num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                num_training_steps=args.max_train_steps * accelerator.num_processes,
                num_cycles=args.lr_num_cycles,
                power=args.lr_power,)
    # Prepare everything with our `accelerator`.
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler)
    # Task-67: restore optimizer + LR-scheduler state for an auto-resumed simple checkpoint
    # (transformer weights were already loaded above; here we recover the training state so a
    # relaunch continues exactly). Best-effort: a fresh-optimizer fallback only warms up a few steps.
    if _simple_resume_state and os.path.isfile(_simple_resume_state):
        try:
            _st = torch.load(_simple_resume_state, map_location="cpu")
            if _st.get("optimizer") is not None:
                optimizer.load_state_dict(_st["optimizer"])
            if _st.get("scheduler") is not None and lr_scheduler is not None:
                lr_scheduler.load_state_dict(_st["scheduler"])
            accelerator.print(f"[simple-ckpt] restored optimizer/scheduler from {_simple_resume_state} "
                              f"(saved step {_st.get('global_step')})")
        except Exception as _e:
            accelerator.print(f"[simple-ckpt] WARNING: could not restore optimizer/scheduler: {_e}")
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    # We need to initialize the trackers we use, and also store our configuration.
    if accelerator.is_main_process:
        tracker_name = args.tracker_name or "cogvideox-sft"
        accelerator.init_trackers(tracker_name, config=vars(args))
        accelerator.print("===== Memory before training =====")
        reset_memory(accelerator.device)
        print_memory(accelerator.device)
    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    accelerator.print("***** Running training *****")
    accelerator.print(f"  Num trainable parameters = {num_trainable_parameters}")
    accelerator.print(f"  Num examples = {len(train_dataset)}")
    accelerator.print(f"  Num batches each epoch = {len(train_dataloader)}")
    accelerator.print(f"  Num epochs = {args.num_train_epochs}")
    accelerator.print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    accelerator.print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    accelerator.print(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    accelerator.print(f"  Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps", disable=not accelerator.is_local_main_process,)
    # For DeepSpeed training
    model_config = transformer.module.config if hasattr(transformer, "module") else transformer.config
    if args.load_tensors:
        del vae, text_encoder
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(accelerator.device)

    alphas_cumprod = scheduler.alphas_cumprod.to(accelerator.device, dtype=torch.float32)
    DEFAULT_SIZE = 336
    middle_size = DEFAULT_SIZE // 14
    # Same pattern as ref8 ``vggt_block`` / ``VGGT_Block``: frozen geo encoder → tokens for downsample/encoder.
    if _rgb_only_enabled():
        # Task-27-10: RGB_ONLY -> never construct DA3; DA3_PRETRAINED is not needed.
        logger.warning("[RGB_ONLY] geometry branch disabled: DA3 is not loaded and no geometry loss is applied.")
        da3_block = ZeroGeo_Block(DEFAULT_SIZE=DEFAULT_SIZE)
    else:
        da3_block = DA3_Block(DEFAULT_SIZE=DEFAULT_SIZE)
    da3_block.to(accelerator.device)  # keep DA3 in float32 — frozen encoder has internal mixed-dtype ops
    da3_block.loading_da3()

    # One-shot pre-training validation pass: exercise the full validation pipeline
    # (rgb/depth side-by-side + log_validation export loop) BEFORE the first optimizer
    # step so pipeline bugs surface immediately at launch.
    if args.run_validation_first and accelerator.is_main_process:
        try:
            run_validation(args, accelerator, transformer, scheduler,
                model_config, weight_dtype, 0, choice="all", da3_block=da3_block)
        except Exception as e:
            logger.warning(f"[run_validation_first] pre-training validation failed: {e}")
    accelerator.wait_for_everyone()

    for epoch in range(args.num_train_epochs):
        transformer.train()
        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer]
            logs = {}
            # last_rank = accelerator.num_processes - 1
            # if global_step == 0 and accelerator.process_index == last_rank:
            #     run_validation(args, accelerator, transformer, scheduler,
            #         model_config, weight_dtype, global_step)
            with accelerator.accumulate(models_to_accumulate):
                if "param_probe_name" not in locals():
                    param_probe_name, param_probe = _select_trainable_param_probe(transformer)
                    if accelerator.is_local_main_process:
                        accelerator.print(f"[debug] param probe: {param_probe_name}")
                probe_before = None
                if accelerator.sync_gradients and param_probe is not None:
                    with torch.no_grad():
                        probe_before = (param_probe.detach().reshape(-1)[:1024].to(dtype=torch.float32, device="cpu").clone())
                images = batch["images"].to(accelerator.device, non_blocking=True)
                videos = batch["videos"].to(accelerator.device, non_blocking=True)
                prompts = batch["prompts"]
                # Encode videos
                images = images.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
                image_noise_sigma = torch.normal(
                    mean=-3.0, std=0.5,
                    size=(images.size(0),),
                    device=accelerator.device,
                    dtype=weight_dtype,)
                image_noise_sigma = torch.exp(image_noise_sigma)
                noisy_images = images + torch.randn_like(images) * image_noise_sigma[:, None, None, None, None]
                B, C, F, H, W = noisy_images.shape
                noisy_images = noisy_images[:, : C//3, :, :, :]
                with torch.no_grad():
                    image_latent_dist = vae.encode(noisy_images).latent_dist
                # ===================== encode video =====================
                videos = videos.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
                videos = videos[:, : C//3, :, :, :]
                with torch.no_grad():
                    # [B, 3, F, H, W]
                    _, _, frame, _, _ = videos.shape
                    latent_dist = vae.encode(videos).latent_dist
                    # [B, 16, F/4, H/8, W/8]
                    # DA3 tokens (ref8: vggt_block.pre_forward_tensor); [:1] = first sample only, full frame count F.
                    # ManifestVideoDataset maps pixels to [-1, 1] (custom_dataset.py) so `videos` is in
                    # [-1, 1] — correct for the VAE but `DA3_Block.da3_processor` expects [0, 1] and clamps
                    # negatives to 0 (line ~127). Convert back to [0, 1] before the DA3 forward; same
                    # convention LVP uses at wan_i2v_da3.py:286.
                    da3_input = videos.permute(0, 2, 1, 3, 4)[:1].mul(0.5).add(0.5).clamp(0.0, 1.0)
                    agg_features = da3_block.pre_forward_tensor(da3_input)
                # [1, B, F, P, 2048] -> grid (same permute/reshape as ref8 VGGT)
                agg_features = agg_features.permute(1,2,3,4,0).reshape(1, frame, middle_size, middle_size, 1024).permute(0,4,1,2,3)
                # Task-27-10: under RGB_ONLY the geo latent never reaches a loss term, so keep the
                # adapter forward out of the autograd graph (it would otherwise be built and discarded).
                with torch.no_grad() if RGB_ONLY else contextlib.nullcontext():
                    proj_features = transformer.downsample(agg_features, (H//8, W//8))
                    middle_geo_features = transformer.encoder(proj_features)
                proj_features = proj_features.permute(0, 2, 1, 3, 4)
                # [1, 16, F/4, H/8, W/8]
                middle_geo_features = middle_geo_features.permute(0, 2, 1, 3, 4)
                # [1, F/4, 16, H/8, W/8]
                middle_geo_features = middle_geo_features.to(memory_format=torch.contiguous_format, dtype=weight_dtype)
                middle_geo_features = middle_geo_features * VAE_SCALING_FACTOR
                # [1, F/4, 16, H/8, W/8]
                # ===================== processing image latent =====================
                image_latent_dist = image_latent_dist.sample()
                B, C, F, H, W = image_latent_dist.shape
                image_latents = image_latent_dist.reshape(B, C, F, H, W)
                image_latents = image_latents * VAE_SCALING_FACTOR
                image_latents = image_latents.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
                image_latents = image_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)
                # ===================== processing video latent =====================
                latent_dist = latent_dist.sample()
                latent_dist = latent_dist.reshape(B, C, -1, H, W)
                video_latents = latent_dist * VAE_SCALING_FACTOR
                video_latents = video_latents.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
                video_latents = video_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)
                # ===================== padding image latent =====================
                padding_shape = (video_latents.shape[0],
                    video_latents.shape[1] - 1, *video_latents.shape[2:],)
                latent_padding = image_latents.new_zeros(padding_shape)
                image_latents = torch.cat([image_latents, latent_padding], dim=1)
                if random.random() < args.noised_image_dropout:
                    image_latents = torch.zeros_like(image_latents)
                # ===================== encode prompts =====================
                if not args.load_tensors:
                    prompt_embeds = compute_prompt_embeddings(
                        tokenizer,
                        text_encoder,
                        prompts,
                        model_config.max_text_seq_length,
                        accelerator.device,
                        weight_dtype,
                        requires_grad=False,)
                else:
                    prompt_embeds = prompts.to(dtype=weight_dtype)
                # Sample noise that will be added to the latents
                batch_size, num_frames, num_channels, height, width = video_latents.shape
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    scheduler.config.num_train_timesteps,
                    (batch_size,),
                    dtype=torch.int64,
                    device=accelerator.device,)
                # Prepare rotary embeds
                image_rotary_emb = (
                    prepare_rotary_positional_embeddings(
                        height=height * VAE_SCALE_FACTOR_SPATIAL,
                        width=width * VAE_SCALE_FACTOR_SPATIAL,
                        num_frames=num_frames,
                        vae_scale_factor_spatial=VAE_SCALE_FACTOR_SPATIAL,
                        patch_size=model_config.patch_size,
                        patch_size_t=(model_config.patch_size_t if hasattr(model_config, "patch_size_t") else None),
                        attention_head_dim=model_config.attention_head_dim,
                        device=accelerator.device,
                        base_height=RoPE_BASE_HEIGHT,
                        base_width=RoPE_BASE_WIDTH,
                    ) if model_config.use_rotary_positional_embeddings else None)
                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                video_latents = torch.cat([
                        video_latents, middle_geo_features], dim=2,)
                # ===== DIAG: velocity-sensitivity (same clip, two noises) =====
                if os.environ.get("DIAG", "") == "1" and REF13:
                    import torch.nn.functional as _F
                    _Cc = video_latents.shape[2]
                    _zgeo = video_latents[:, :, _Cc // 2:]  # geometry z0 (cancels in the target delta)
                    for _tv in [int(x) for x in os.environ.get("DIAG_TS", "100,500,900").split(",")]:
                        _t = torch.full((video_latents.shape[0],), _tv, dtype=torch.int64, device=accelerator.device)
                        _ac = alphas_cumprod[_t].view(-1, 1, 1, 1, 1).to(video_latents.dtype)
                        _al = _ac.sqrt(); _sg = (1 - _ac).sqrt()
                        def _fwd(_eps):
                            _ny = scheduler.add_noise(video_latents, _eps, _t)
                            _rgb = _ny[:, :, :_Cc // 2].clone(); _dep = _ny[:, :, _Cc // 2:].clone()
                            _nii = torch.cat([_rgb, image_latents.clone()], dim=2)
                            _ofs = _nii.new_full((1,), 2.0)
                            _out = transformer(hidden_states=_nii, encoder_hidden_states=prompt_embeds,
                                               timestep=_t, ofs=_ofs, image_rotary_emb=image_rotary_emb,
                                               return_dict=False)
                            _rp = scheduler.get_velocity(_out[0], _rgb, _t)
                            _erh = (_sg * _rgb + _al * _rp).detach()
                            _vg = transformer.geo_dit(_out[1], _t, (num_frames, height, width), 'empty',
                                                      z_t_geo=_dep, eps_rgb_hat=_erh)
                            return _vg, _eps[:, :, :_Cc // 2]
                        torch.manual_seed(1234)
                        _e1 = torch.randn_like(video_latents); _e1[:, :, _Cc // 2:] = _e1[:, :, :_Cc // 2]
                        torch.manual_seed(5678)
                        _e2 = torch.randn_like(video_latents); _e2[:, :, _Cc // 2:] = _e2[:, :, :_Cc // 2]
                        with torch.no_grad():
                            _v1, _eps16_1 = _fwd(_e1); _v2, _eps16_2 = _fwd(_e2)
                        _dpred = (_v1 - _v2).float().flatten()
                        _dtgt = (_al.float() * (_eps16_1.float() - _eps16_2.float())).flatten()
                        _cos = _F.cosine_similarity(_dpred, _dtgt, dim=0).item()
                        _ratio = (_dpred.norm() / (_dtgt.norm() + 1e-8)).item()
                        _rel = ((_v1 - _v2).float().norm() / (_v1.float().norm() + 1e-8)).item()
                        print(f"[DIAG-RESULT] t={_tv} alpha={_al.flatten()[0].item():.4f} sigma={_sg.flatten()[0].item():.4f} "
                              f"cos(dVpred,alpha*dEps)={_cos:.4f} ratio|dVpred|/|alpha*dEps|={_ratio:.4f} "
                              f"relSens|dVpred|/|V1|={_rel:.4f}")
                    sys.exit(0)
                noise = torch.randn_like(video_latents)
                if REF13:
                    _c0 = video_latents.shape[2] // 2
                    if os.environ.get("GEO_COUPLED_NOISE", "1") == "1":   # Task-48: configurable
                        noise[:, :, _c0:] = noise[:, :, :_c0]   # coupled eps_geo = eps_rgb (default)
                noisy_video_latents = scheduler.add_noise(video_latents, noise, timesteps)
                _, _, C, _, _ = noisy_video_latents.shape
                # ===================== processing model input =====================
                image_latents_rgb = image_latents.clone()
                noisy_video_latents_rgb = noisy_video_latents[:, :, : C // 2].clone()
                noisy_video_latents_depth = noisy_video_latents[:, :, C // 2 : C // 2 * 2].clone()
                noisy_image_input = torch.cat(
                    [noisy_video_latents_rgb,
                        image_latents_rgb,], dim=2,)
                (model_config.patch_size_t if hasattr(model_config, "patch_size_t") else None),
                ofs_embed_dim = ((model_config.ofs_embed_dim if hasattr(model_config, "ofs_embed_dim") else None),)
                ofs_emb = None if ofs_embed_dim is None else noisy_image_input.new_full((1,), fill_value=2.0)
                # Task-32: clear the midpoint-tap cache before the forward; the hook re-populates it.
                if getattr(transformer, "_geo_hook_cache", None) is not None:
                    transformer._geo_hook_cache.clear()
                # Predict the noise residual
                outputs = transformer(
                    hidden_states=noisy_image_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timesteps,
                    ofs=ofs_emb,
                    image_rotary_emb=image_rotary_emb,
                    return_dict=False,)
                rgb_output = outputs[0]
                hidden_states = outputs[1]
                # Task-32: Geo-DiT context = midpoint block capture if GEO_HOOK_LAYER is active,
                # else the final trunk hidden state (legacy). Take the last entry (robust to a
                # gradient-checkpointing recompute re-firing the hook).
                _hrgb_geo = transformer._geo_hook_cache[-1] if getattr(transformer, "_geo_hook_cache", None) else hidden_states
                rgb_pred = scheduler.get_velocity(rgb_output, noisy_video_latents_rgb, timesteps)
                if RGB_ONLY:
                    # Task-27-10: no Geo-DiT, no depth head, no DA3 target — RGB objective only.
                    depth_pred = None
                elif REF13:
                    _ac = alphas_cumprod[timesteps].view(-1, 1, 1, 1, 1).to(rgb_output.dtype)
                    _al = _ac.sqrt(); _sg = (1 - _ac).sqrt()
                    _eps16 = noise[:, :, : video_latents.shape[2] // 2].to(rgb_output.dtype)
                    REF13_geo_target = (_al * _eps16 - _sg * middle_geo_features.to(rgb_output.dtype))
                    eps_rgb_hat = (_sg * noisy_video_latents_rgb + _al * rgb_pred).detach()
                    if REF15:
                        # ref15: always masked geometry input; h_rgb NOT detached so L_geo forces the
                        # RGB hidden states to supply the masked-out geometry. Task-32: h_rgb is the
                        # MIDPOINT block capture (_hrgb_geo) rather than the final hidden_states.
                        _hg = _hrgb_geo; _mode = 'masked'; gamma_mode = 1.0
                    else:
                        _use_empty = (torch.rand(()) < float(os.environ.get("P_EMPTY", "0.8"))).item()
                        if _use_empty:
                            _hg = _hrgb_geo; _mode = 'empty'; gamma_mode = 1.0
                        else:
                            _hg = _hrgb_geo.detach(); _mode = 'present'; gamma_mode = 0.25
                    depth_pred = transformer.geo_dit(
                        _hg, timesteps, (num_frames, height, width), _mode,
                        z_t_geo=noisy_video_latents_depth, eps_rgb_hat=eps_rgb_hat)
                else:
                    depth_output = transformer.forward_depth(noisy_video_latents_depth, transformer._depth_feat(hidden_states), rgb_output)[0]
                    depth_pred = scheduler.get_velocity(depth_output, noisy_video_latents_depth, timesteps)
                weights = 1 / (1 - alphas_cumprod[timesteps])
                _weight_ref = rgb_pred if depth_pred is None else depth_pred   # Task-27-10: same shape
                while len(weights.shape) < len(_weight_ref.shape):
                    weights = weights.unsqueeze(-1)
                target = video_latents
                # ==== mask the target and model_pred ====
                C = target.shape[2]
                rgb_target, depth_target = target.chunk(2, dim=2)

                # ===================== Loss Functions =====================
                # loss_fn(pred, target, weight) = mean( weight * (pred - target)^2 )
                #   i.e. SNR-weighted MSE in latent space (v-prediction).
                #
                # task10 (23-04): depth loss is now in 16-ch v-pred space, symmetric
                # with the RGB branch. The encoder (self.project) is frozen and the
                # decoder (self.reconstruct) is no longer called — both for the
                # reasons in design/23-04/tess.md §"Reanalysis".
                #
                # Training loss:
                #   rgb_loss   = loss_fn(rgb_velocity_pred,   rgb_target_latent,   snr_weight)
                #   depth_loss = loss_fn(depth_velocity_pred, depth_target_latent, snr_weight)
                #   total_loss = rgb_loss + 0.3 * depth_scale * depth_loss
                #
                # where depth_scale = min(1.0, global_step / 100) ramps depth supervision
                # over the first 100 steps to avoid destabilising the RGB branch.
                # ============================================================
                rgb_loss = loss_fn(rgb_pred, rgb_target, weights)
                if RGB_ONLY:
                    # Task-27-10: total loss == rgb_loss; depth_loss/depth_scale kept for the logger.
                    depth_loss = rgb_loss.new_zeros(())
                    depth_scale = 0.0
                    loss = rgb_loss
                elif REF13:
                    if os.environ.get("GEO_X0", "0") == "1":
                        # Task-53 FIX: PURE geometry (x0) supervision. The velocity target
                        # v=alpha*eps-sigma*z0_geo is dominated by the coupled-noise alpha*eps that the
                        # model is handed via eps_rgb_hat, so it minimizes the velocity MSE by ECHOING
                        # eps_hat (+copying visible geo tokens) without learning RGB->geometry (verified:
                        # leakage-free cos_geom ~ random). Supervising geo_dit's output DIRECTLY against
                        # z0_geo removes the noise term entirely -> all gradient drives geometry. The
                        # geo_dit output is now interpreted as z0 (not velocity).
                        depth_loss = ((depth_pred - middle_geo_features.to(rgb_output.dtype)) ** 2).mean()
                    else:
                        # full geometry-velocity MSE (no SNR/sigma^2 reweight, per CN-MIGV spec)
                        depth_loss = ((depth_pred - REF13_geo_target) ** 2).mean()
                    depth_scale = min(1.0, global_step / float(os.environ.get("GEO_WARMUP", "100")))            # lambda ramp
                    lam = float(os.environ.get("LAMBDA_GEO", "0.03")) * depth_scale  # Task-41 default 0.1->0.03
                    loss = rgb_loss + lam * gamma_mode * depth_loss
                else:
                    depth_loss = loss_fn(depth_pred, depth_target, weights)
                    depth_scale = min(1.0, global_step / float(os.environ.get("GEO_WARMUP", "100")))
                    loss = rgb_loss + depth_loss * 0.3 * depth_scale
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    # `get_gradient_norm()` can be misleading under ZeRO since many grads are partitioned / None.
                    clip_total_norm = accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                    if clip_total_norm is None:
                        clip_total_norm_value = 0.0
                    elif isinstance(clip_total_norm, torch.Tensor):
                        clip_total_norm_value = float(clip_total_norm.detach().cpu().item())
                    else:
                        clip_total_norm_value = float(clip_total_norm)
                    if accelerator.state.deepspeed_plugin is None:
                        gradient_norm_before_clip = get_gradient_norm(transformer.parameters())
                        logs.update({"gradient_norm_before_clip": gradient_norm_before_clip,
                                "grad_norm_clip_total": clip_total_norm_value,})
                    else:
                        # For DeepSpeed ZeRO, only log the clip norm (reliable); don't log misleading param.grad norms.
                        logs.update({"grad_norm_clip_total": clip_total_norm_value})
                if accelerator.state.deepspeed_plugin is None:
                    optimizer.step()
                    optimizer.zero_grad()
                if not args.use_cpu_offload_optimizer:
                    lr_scheduler.step()
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                # Probe whether weights actually changed after the step.
                if param_probe is not None and probe_before is not None:
                    with torch.no_grad():
                        probe_after = (
                            param_probe.detach().reshape(-1)[:1024].to(dtype=torch.float32, device="cpu"))
                        update_linf = float((probe_after - probe_before).abs().max().item())
                        update_l2 = float((probe_after - probe_before).norm(p=2).item())
                    logs.update({"param_update_linf": update_linf, "param_update_l2": update_l2,})
                # Checkpointing
                # if accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED:
                if accelerator.is_main_process:
                    if _simple_ckpt_dir and args.checkpointing_steps and global_step % args.checkpointing_steps == 0:
                        # Task-67: full-state simple checkpoint -> <SIMPLE_CKPT_DIR>/step_NNNNNN/{transformer, train_state.pt}
                        try:
                            import sys as _sys
                            _tools = os.path.join(os.environ.get("PROJECT_ROOT", str(Path(__file__).resolve().parents[2])), "tools")
                            if _tools not in _sys.path:
                                _sys.path.insert(0, _tools)
                            import overfit_ckpt as _ofc
                            _sd = _ofc.step_dir(_simple_ckpt_dir, global_step)
                            _tf = unwrap_model(accelerator, transformer)
                            _dt = (torch.float16 if args.mixed_precision == "fp16"
                                   else torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32)
                            _tf.to(_dt).save_pretrained(os.path.join(_sd, "transformer"),
                                                        safe_serialization=True, max_shard_size="5GB")
                            torch.save({"optimizer": optimizer.state_dict(),
                                        "scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
                                        "global_step": global_step, "epoch": epoch},
                                       os.path.join(_sd, "train_state.pt"))
                            _ofc.update_latest(_simple_ckpt_dir, global_step)
                            _rm = _ofc.prune(_simple_ckpt_dir,
                                             keep_topk=int(os.environ.get("CKPT_TOPK", "2")),
                                             keep_interval=int(os.environ.get("CKPT_KEEP_INTERVAL", "0")))
                            logger.info(f"[simple-ckpt] saved step_{global_step:06d} (pruned: {_rm})")
                        except Exception as _e:
                            logger.warning(f"[simple-ckpt] save failed at step {global_step}: {_e}")
                    else:
                        checkpoint_manager.save_checkpoint_model(accelerator, transformer,global_step, logger)
                # Post-first-step sanity: after the very first optimizer step, run one full
                # validation pass (viz + checkpoint already saved above) to prove that the
                # train→val→log loop is live before committing to a long run.
                if (args.run_validation_after_first_step
                        and global_step == resumed_global_step + 1
                        and accelerator.is_main_process):
                    try:
                        run_validation(args, accelerator, transformer, scheduler,
                            model_config, weight_dtype, global_step, choice="all", da3_block=da3_block)
                    except Exception as e:
                        logger.warning(f"[run_validation_after_first_step] first-step validation failed: {e}")
                # Validation (video generation). Gated by RUN_VAL_VIDEO env (default off)
                # so toy/loss-only runs skip the expensive + memory-heavy video gen.
                should_run_validation = os.environ.get("RUN_VAL_VIDEO", "0") == "1" and (
                    args.validation_prompt is not None and (
                    args.validation_steps is not None and (global_step) % args.validation_steps == 0))
                if should_run_validation and accelerator.is_main_process:
                    # Wrap: the RGB video pipeline is the most memory-hungry step; on a contended GPU an
                    # OOM here must NOT abort the run before the checkpoint + geo eval are saved (Task-53).
                    try:
                        run_validation(args, accelerator, transformer, scheduler,
                            model_config, weight_dtype, global_step, choice="nodn", da3_block=da3_block)
                    except Exception as e:
                        logger.warning(f"[run_validation] RGB video validation failed at step {global_step}: {e}")
                        import gc as _gc; _gc.collect(); torch.cuda.empty_cache()
                        transformer.train()
                # Validation loss (computed on held-out data)
                should_run_val_loss = (
                    val_dataloader is not None
                    and args.validation_steps is not None
                    and global_step % args.validation_steps == 0)
                if should_run_val_loss and accelerator.is_main_process:
                    try:
                        val_logs = compute_validation_loss(
                            args, accelerator, transformer, vae, text_encoder, tokenizer,
                            scheduler, val_dataloader, da3_block, weight_dtype,
                            model_config, alphas_cumprod,
                            VAE_SCALING_FACTOR, VAE_SCALE_FACTOR_SPATIAL,
                            RoPE_BASE_HEIGHT, RoPE_BASE_WIDTH,
                            middle_size, global_step)
                        logs.update(val_logs)
                        logger.info(f"step={global_step} | validation: {val_logs}")
                    except Exception as e:
                        logger.warning(f"[validation] skipped at step {global_step}: {e}")
                        transformer.train()
            last_lr = lr_scheduler.get_last_lr()[0] if lr_scheduler is not None else args.learning_rate
            logs.update({"loss": loss.detach().item(), "lr": last_lr,})
            logs.update({
                "rgb_loss": rgb_loss.detach().item(),
                "depth_loss": depth_loss.detach().item(),
                "depth_scale": depth_scale,
            })
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            if accelerator.is_main_process:
                logger.info(f"step={global_step} | {logs}")
                # Task-64: machine-parsable metrics line for the shared tools/make_loss_curves.py
                # (same `[geo-metrics]` schema as Cosmos) -> train/val rgb/geo loss PNGs. The module
                # logger sits at WARNING, so write to a dedicated file under VAL_VIZ_DIR rather than
                # relying on logger.info. Env-gated; other tasks unaffected.
                _vvd = os.environ.get("VAL_VIZ_DIR", "")
                if _vvd:
                    try:
                        _line = (f"[geo-metrics] step={global_step} total={logs['loss']:.6f} "
                                 f"rgb_loss={logs['rgb_loss']:.6f} geo_loss={logs['depth_loss']:.6f} "
                                 f"ramp={depth_scale:.4f}")
                        _vl = logs.get("val/loss"); _vr = logs.get("val/rgb_loss"); _vg = logs.get("val/depth_loss")
                        if _vl is not None:
                            _line += (f" val_total={float(_vl):.6f} val_rgb_loss={float(_vr):.6f} "
                                      f"val_geo_loss={float(_vg):.6f}")
                        os.makedirs(_vvd, exist_ok=True)
                        with open(os.path.join(_vvd, "train_metrics.log"), "a") as _mf:
                            _mf.write(_line + "\n")
                    except Exception:
                        pass
            loss = accelerator.reduce(loss, reduction="mean")
            if global_step >= args.max_train_steps:
                break
    accelerator.wait_for_everyone()

    # Task-53: geometry evaluation on the TRAINED geo_dit (correct module), with quantitative
    # MSE/cosine vs the GT geometry target. Gated by GEO_EVAL so other tasks are unaffected.
    if os.environ.get("GEO_EVAL", "0") == "1" and accelerator.is_main_process and val_dataloader is not None:
        try:
            run_geo_eval(args, accelerator, transformer, vae, text_encoder, tokenizer, scheduler,
                         val_dataloader, da3_block, weight_dtype, model_config, alphas_cumprod,
                         VAE_SCALING_FACTOR, VAE_SCALE_FACTOR_SPATIAL, RoPE_BASE_HEIGHT, RoPE_BASE_WIDTH,
                         middle_size, global_step)
        except Exception as e:
            logger.warning(f"[GEO_EVAL] geometry eval failed: {e}")
            import traceback; traceback.print_exc()

    if accelerator.is_main_process:
        checkpoint_manager.save_model(accelerator, transformer)
        # Cleanup trained models to save memory
        if args.load_tensors: del transformer
        else: del transformer, text_encoder, vae
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(accelerator.device)
        accelerator.print("===== Memory before testing =====")
        print_memory(accelerator.device)
        reset_memory(accelerator.device)
    accelerator.end_training()

if __name__ == "__main__":
    args = get_args()
    main(args)
