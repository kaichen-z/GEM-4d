"""Modified from https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
"""
#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import gc
import logging
import math
import os
import pickle
import random
import shutil
import sys

import accelerate
import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms.functional as TF
import transformers
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import DDIMScheduler, FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (EMAModel,
                                      compute_density_for_timestep_sampling,
                                      compute_loss_weighting_for_sd3)
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module
from einops import rearrange
from omegaconf import OmegaConf
from packaging import version
from PIL import Image
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullOptimStateDictConfig, FullStateDictConfig, ShardedOptimStateDictConfig,
    ShardedStateDictConfig)
from torch.utils.data import RandomSampler
from torch.utils.tensorboard import SummaryWriter

# ===================== Task-42: ref15-style geometry branch =====================
# Enabled with env GEO=1. Adds a Geo-DiT (reused verbatim from tesseract/modules/geo_dit.py)
# that predicts the geometry (DA3-depth) velocity from the masked noised geo latent + an
# intermediate RGB-DiT hidden state (coupled noise). Train-only; RGB inference unchanged.
_GEO            = os.environ.get("GEO", "0") == "1"
_GEO_LAMBDA     = float(os.environ.get("LAMBDA_GEO", "0.5"))
_GEO_HOOK_LAYER = int(os.environ.get("GEO_HOOK_LAYER", "15"))    # Video-DiT feature layer feeding the Geo-DiT
_GEO_LATENTS    = os.environ.get("GEO_LATENTS", "")  # legacy, unused
# Task-48: configurable (were hard-coded). DA3 feature layer (-1 => use DA3 final DEPTH, Wan's default
# geometry target); warm-up steps for the lambda ramp; coupled-noise toggle (Geo-DiT shares Video-DiT noise).
_GEO_DA3_FEAT_LAYER = int(os.environ.get("GEO_DA3_FEAT_LAYER", "-1"))
_GEO_WARMUP     = float(os.environ.get("GEO_WARMUP", "100"))
_GEO_COUPLED    = os.environ.get("GEO_COUPLED_NOISE", "1") == "1"
# Task-51: end-of-training geometry-visualization dump (default OFF; no effect on other tasks).
# When GEO_VIZ=1, during the last GEO_VIZ_WINDOW steps we save, once per dataset idx, the geometry
# TARGET latent (z_geo = DA3 depth -> Wan VAE) and the Geo-DiT one-step RECONSTRUCTION
# (z0_hat = noisy_geo - sigma * v_geo) plus the GT RGB clip, into GEO_VIZ_DIR. An offline script
# (task51_geo_viz.py) VAE-decodes these into the "reconstructed geometry visualization video".
_GEO_VIZ        = os.environ.get("GEO_VIZ", "0") == "1"
_GEO_VIZ_DIR    = os.environ.get("GEO_VIZ_DIR", "")
_GEO_VIZ_WINDOW = int(os.environ.get("GEO_VIZ_WINDOW", "60"))
_GEO_VIZ_SIGMA  = float(os.environ.get("GEO_VIZ_SIGMA", "0.3"))   # Task-54: eval geo recon at σ≈this (informative, low-noise)
# Task-54: configurable validation sampler (defaults = original 4.5/25). For a memorized overfit, low
# guidance + more steps reproduce the exact training trajectory far more faithfully than CFG=4.5.
_VAL_GUIDANCE   = float(os.environ.get("VAL_GUIDANCE", "4.5"))
_VAL_NUM_STEPS  = int(os.environ.get("VAL_NUM_STEPS", "25"))
# Task-65: standardized Task-62 validation visualization. When VAL_VIZ_DIR is set the validation
# hook writes Step_NNNNNN/rgb_sample{i}.mp4 (GT-left|pred-right) and geo_sample{i}.mp4 (geometry
# latent PCA->RGB, GT|pred), and at end-of-training the 4 root loss PNGs — the shared layout produced
# identically by tools/val_viz.py across Tesseract/Wan/Cosmos. geo_sample is drawn from the training
# batch (the σ-closest-to-VAL_VIZ geo recon since the last validation), like the Cosmos integration.
_VAL_VIZ_DIR    = os.environ.get("VAL_VIZ_DIR", "")
_valviz         = None
if _VAL_VIZ_DIR:
    _tools_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "tools"))
    if _tools_dir not in sys.path:
        sys.path.insert(0, _tools_dir)
    import val_viz as _valviz
_GEO_VALVIZ_STASH = {}   # {"z_geo":[C,T,h,w], "z0_pred":[C,T,h,w], "sig_close":float} train-batch geo for geo_sample
_VALVIZ_HIST = {"train_rgb": [], "train_geo": [], "val_rgb": [], "val_geo": []}
_VAL_META_PAIRS = []     # [[caption, video_path], ...] from --validation_meta, for rgb_sample GT clips
_geo_state      = {"h": None}   # forward-hook holder for the RGB hidden state
if _GEO:
    # Resolve the tesseract package that holds modules/geo_dit.py. Prefer an env
    # override, else resolve relative to this file (repo: <root>/tesseract/tesseract),
    # else fall back to the original absolute path.
    _geo_tess = os.environ.get("GEO_TESSERACT", "")
    if not _geo_tess:
        _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        # code-git layout: <root>/cogvideox/gem4d/modules/geo_dit.py
        _geo_tess = os.path.join(_repo_root, "cogvideox", "gem4d")
    sys.path.insert(0, _geo_tess)
    from modules.geo_dit import GeoDiT
# ================================================================================
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from transformers.utils import ContextManagers

import datasets

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.data import (ASPECT_RATIO_512, ASPECT_RATIO_RANDOM_CROP_512,
                             ASPECT_RATIO_RANDOM_CROP_PROB,
                             AspectRatioBatchImageVideoSampler,
                             ImageVideoDataset, ImageVideoSampler,
                             RandomSampler, get_closest_ratio, get_random_mask)
from videox_fun.models import (AutoencoderKLWan, CLIPModel, WanT5EncoderModel,
                               WanTransformer3DModel)
from videox_fun.pipeline import WanFunInpaintPipeline, WanFunPipeline
from videox_fun.utils.discrete_sampler import DiscreteSampling
from videox_fun.utils.utils import (calculate_dimensions, get_image_latent,
                                    get_image_to_video_latent,
                                    save_videos_grid)

if is_wandb_available():
    import wandb


def filter_kwargs(cls, kwargs):
    import inspect
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self', 'cls'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return filtered_kwargs

def resize_mask(mask, latent, process_first_frame_only=True):
    latent_size = latent.size()
    batch_size, channels, num_frames, height, width = mask.shape

    if process_first_frame_only:
        target_size = list(latent_size[2:])
        target_size[0] = 1
        first_frame_resized = F.interpolate(
            mask[:, :, 0:1, :, :],
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
        
        target_size = list(latent_size[2:])
        target_size[0] = target_size[0] - 1
        if target_size[0] != 0:
            remaining_frames_resized = F.interpolate(
                mask[:, :, 1:, :, :],
                size=target_size,
                mode='trilinear',
                align_corners=False
            )
            resized_mask = torch.cat([first_frame_resized, remaining_frames_resized], dim=2)
        else:
            resized_mask = first_frame_resized
    else:
        target_size = list(latent_size[2:])
        resized_mask = F.interpolate(
            mask,
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
    return resized_mask

def linear_decay(initial_value, final_value, total_steps, current_step):
    if current_step >= total_steps:
        return final_value
    current_step = max(0, current_step)
    step_size = (final_value - initial_value) / total_steps
    current_value = initial_value + step_size * current_step
    return current_value

def generate_timestep_with_lognorm(low, high, shape, device="cpu", generator=None):
    u = torch.normal(mean=0.0, std=1.0, size=shape, device=device, generator=generator)
    t = 1 / (1 + torch.exp(-u)) * (high - low) + low
    return torch.clip(t.to(torch.int32), low, high - 1)

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.18.0.dev0")

logger = get_logger(__name__, log_level="INFO")

def _valviz_load_gt_clip(path, num_frames, height, width):
    """Task-65: load the GT video clip -> [T,H,W,3] uint8, T=num_frames @ (height,width).

    Used to build rgb_sample{i}.mp4 (GT-left|pred-right). Uses decord if available, else imageio.
    """
    import numpy as _np
    frames = None
    try:
        import decord
        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(path, width=width, height=height)
        n = len(vr)
        idx = list(range(min(num_frames, n)))
        frames = vr.get_batch(idx).asnumpy()              # [t,H,W,3] uint8
    except Exception:
        import imageio.v2 as _iio
        import cv2 as _cv2
        rd = _iio.get_reader(path)
        out = []
        for k, fr in enumerate(rd):
            if k >= num_frames:
                break
            out.append(_cv2.resize(_np.asarray(fr)[..., :3], (width, height)))
        rd.close()
        frames = _np.stack(out, 0) if out else None
    if frames is None or len(frames) == 0:
        return None
    if len(frames) < num_frames:                          # pad by repeating the last frame
        pad = _np.repeat(frames[-1:], num_frames - len(frames), axis=0)
        frames = _np.concatenate([frames, pad], axis=0)
    return frames[:num_frames]


def log_validation(vae, text_encoder, tokenizer, clip_image_encoder, transformer3d, args, config, accelerator, weight_dtype, global_step):
    try:
        is_deepspeed = type(transformer3d).__name__ == 'DeepSpeedEngine'
        if is_deepspeed:
            origin_config = transformer3d.config
            transformer3d.config = accelerator.unwrap_model(transformer3d).config
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=accelerator.device):
            logger.info("Running validation... ")
            scheduler = FlowMatchEulerDiscreteScheduler(
                **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
            )
        
            if args.train_mode != "normal":
                pipeline = WanFunInpaintPipeline(
                    vae=vae, 
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    transformer=accelerator.unwrap_model(transformer3d) if type(transformer3d).__name__ == 'DistributedDataParallel' else transformer3d,
                    scheduler=scheduler,
                    clip_image_encoder=clip_image_encoder,
                )
            else:
                pipeline = WanFunPipeline(
                    vae=vae, 
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    transformer=accelerator.unwrap_model(transformer3d) if type(transformer3d).__name__ == 'DistributedDataParallel' else transformer3d,
                    scheduler=scheduler,
                )
            pipeline = pipeline.to(accelerator.device)

            if args.seed is None:
                generator = None
            else:
                rank_seed = args.seed + accelerator.process_index
                generator = torch.Generator(device=accelerator.device).manual_seed(rank_seed)
                logger.info(f"Rank {accelerator.process_index} using seed: {rank_seed}")

            for i in range(len(args.validation_prompts)):
                if args.train_mode != "normal":
                    start_image = Image.open(args.validation_paths[i])
                    width, height = start_image.width, start_image.height
                    width, height = calculate_dimensions(args.image_sample_size * args.image_sample_size,  width / height)

                    video_length = int((args.video_sample_n_frames - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if args.video_sample_n_frames != 1 else 1
                    input_video, input_video_mask, _ = get_image_to_video_latent(args.validation_paths[i], None, video_length=video_length, sample_size=[height, width])
                    sample = pipeline(
                        args.validation_prompts[i],
                        num_frames = video_length,
                        negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                        height      = height,
                        width       = width,
                        generator   = generator,

                        video        = input_video,
                        mask_video   = input_video_mask,
                        num_inference_steps = _VAL_NUM_STEPS,
                        guidance_scale      = _VAL_GUIDANCE,
                    ).videos

                    _viz_dir = getattr(args, "validation_output_dir", None) or args.output_dir  # Task-57
                    os.makedirs(os.path.join(_viz_dir, "sample"), exist_ok=True)
                    save_videos_grid(
                        sample,
                        os.path.join(
                            _viz_dir,
                            f"sample/sample-{global_step}-rank{accelerator.process_index}-image-{i}.mp4"
                        )
                    )
                    # Task-65: standardized rgb_sample{i}.mp4 (GT-left | pred-right) into VAL_VIZ_DIR.
                    if _VAL_VIZ_DIR and _valviz is not None and accelerator.is_main_process:
                        try:
                            pred_clip = sample[0]                              # [C,T,H,W] in [0,1]
                            _T = int(pred_clip.shape[1])
                            gt_path = _VAL_META_PAIRS[i][1] if i < len(_VAL_META_PAIRS) else None
                            gt_clip = _valviz_load_gt_clip(gt_path, _T, height, width) if gt_path else None
                            if gt_clip is None:                               # fall back to the I2V start frame
                                _f0 = np.array(Image.open(args.validation_paths[i]).convert("RGB").resize((width, height)))
                                gt_clip = np.repeat(_f0[None], _T, axis=0)
                            _valviz.save_rgb_sample(_VAL_VIZ_DIR, global_step, i, gt_clip, pred_clip)
                        except Exception as _e:
                            print(f"[VAL_VIZ] rgb_sample{i} skipped at step {global_step}: {_e}")
                else:
                    sample = pipeline(
                        args.validation_prompts[i],
                        num_frames = args.video_sample_n_frames,
                        negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                        height      = args.video_sample_size,
                        width       = args.video_sample_size,
                        generator   = generator,
                        num_inference_steps = _VAL_NUM_STEPS,
                        guidance_scale      = _VAL_GUIDANCE,
                    ).videos
                    _viz_dir = getattr(args, "validation_output_dir", None) or args.output_dir  # Task-57
                    os.makedirs(os.path.join(_viz_dir, "sample"), exist_ok=True)
                    save_videos_grid(
                        sample,
                        os.path.join(
                            _viz_dir,
                            f"sample/sample-{global_step}-rank{accelerator.process_index}-image-{i}.mp4"
                        )
                    )

            # Task-65: standardized geo_sample0.mp4 (geometry latent PCA->RGB, GT|pred) into the same
            # Step_NNNNNN dir. Source = the σ-closest training-batch geo recon collected since the last
            # validation (the geometry branch only runs on the train batch; documented limitation, same
            # as the Cosmos integration). z_geo/z0_pred are [C=16, T, h, w] -> layout="CTHW".
            if _VAL_VIZ_DIR and _valviz is not None and accelerator.is_main_process and _GEO_VALVIZ_STASH:
                try:
                    _valviz.save_geo_sample(_VAL_VIZ_DIR, global_step, 0,
                                            _GEO_VALVIZ_STASH["z_geo"], _GEO_VALVIZ_STASH["z0_pred"],
                                            layout="CTHW")
                    # log a val-point so val_{rgb,geo}_loss.png are non-empty (overfit: val≈train sample)
                    _VALVIZ_HIST["val_rgb"].append((int(global_step), float(_GEO_VALVIZ_STASH.get("rgb_loss", float("nan")))))
                    _VALVIZ_HIST["val_geo"].append((int(global_step), float(_GEO_VALVIZ_STASH.get("geo_loss", float("nan")))))
                    _GEO_VALVIZ_STASH.clear()
                except Exception as _e:
                    print(f"[VAL_VIZ] geo_sample0 skipped at step {global_step}: {_e}")

            del pipeline
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            vae.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)
            if not args.enable_text_encoder_in_dataloader:
                text_encoder.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)
        if is_deepspeed:
            transformer3d.config = origin_config
    except Exception as e:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print(f"Eval error on rank {accelerator.process_index} with info {e}")
        vae.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)
        if not args.enable_text_encoder_in_dataloader:
            text_encoder.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. "
        ),
    )
    parser.add_argument(
        "--train_data_meta",
        type=str,
        default=None,
        help=(
            "A csv containing the training data. "
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default=None,
        nargs="+",
        help=("A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--validation_paths",
        type=str,
        default=None,
        nargs="+",
        help=("A set of control videos evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--validation_meta",
        type=str,
        default=None,
        help=(
            "Task-55: a Tesseract-style JSON ([[caption, video_path], ...], e.g. rl100_val.json). "
            "If set, --validation_prompts / --validation_paths are populated from it (the first frame "
            "of each clip is used as the I2V start image)."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--validation_output_dir",
        type=str,
        default=None,
        help="Task-57: directory for validation/visualization videos (the sample/ dir). "
             "Defaults to --output_dir when unset.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--use_came",
        action="store_true",
        help="whether to use came",
    )
    parser.add_argument(
        "--multi_stream",
        action="store_true",
        help="whether to use cuda multi-stream",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--vae_mini_batch", type=int, default=32, help="mini batch size for vae."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--report_model_info", action="store_true", help="Whether or not to report more info about model (such as norm, grad)."
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store (top-K, latest). Task-57: CHECKPOINT_TOPK."),
    )
    parser.add_argument(
        "--checkpoint_keep_interval",
        type=int,
        default=0,
        help=("Task-57: always keep checkpoints whose step is a multiple of this value (periodic "
              "retention), independent of --checkpoints_total_limit. 0 = disabled."),
    )
    parser.add_argument(
        "--pretrained_vision_model_path",
        type=str,
        default=None,
        help=("Task-57: path to the pretrained vision foundation model (geometry teacher, e.g. DA3) "
              "for ref15. When set, overrides the GEO_DA3_CKPT env / hard-coded default."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=5,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=2000,
        help="Run validation every X steps.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    
    parser.add_argument(
        "--snr_loss", action="store_true", help="Whether or not to use snr_loss."
    )
    parser.add_argument(
        "--uniform_sampling", action="store_true", help="Whether or not to use uniform_sampling."
    )
    parser.add_argument(
        "--enable_text_encoder_in_dataloader", action="store_true", help="Whether or not to use text encoder in dataloader."
    )
    parser.add_argument(
        "--enable_bucket", action="store_true", help="Whether enable bucket sample in datasets."
    )
    parser.add_argument(
        "--random_ratio_crop", action="store_true", help="Whether enable random ratio crop sample in datasets."
    )
    parser.add_argument(
        "--random_frame_crop", action="store_true", help="Whether enable random frame crop sample in datasets."
    )
    parser.add_argument(
        "--random_hw_adapt", action="store_true", help="Whether enable random adapt height and width in datasets."
    )
    parser.add_argument(
        "--training_with_video_token_length", action="store_true", help="The training stage of the model in training.",
    )
    parser.add_argument(
        "--auto_tile_batch_size", action="store_true", help="Whether to auto tile batch size.",
    )
    parser.add_argument(
        "--motion_sub_loss", action="store_true", help="Whether enable motion sub loss."
    )
    parser.add_argument(
        "--motion_sub_loss_ratio", type=float, default=0.25, help="The ratio of motion sub loss."
    )
    parser.add_argument(
        "--train_sampling_steps",
        type=int,
        default=1000,
        help="Run train_sampling_steps.",
    )
    parser.add_argument(
        "--keep_all_node_same_token_length",
        action="store_true", 
        help="Reference of the length token.",
    )
    parser.add_argument(
        "--token_sample_size",
        type=int,
        default=512,
        help="Sample size of the token.",
    )
    parser.add_argument(
        "--video_sample_size",
        type=int,
        default=512,
        help="Sample size of the video.",
    )
    parser.add_argument(
        "--image_sample_size",
        type=int,
        default=512,
        help="Sample size of the image.",
    )
    parser.add_argument(
        "--fix_sample_size", 
        nargs=2, type=int, default=None,
        help="Fix Sample size [height, width] when using bucket and collate_fn."
    )
    parser.add_argument(
        "--video_sample_stride",
        type=int,
        default=4,
        help="Sample stride of the video.",
    )
    parser.add_argument(
        "--video_sample_n_frames",
        type=int,
        default=17,
        help="Num frame of video.",
    )
    parser.add_argument(
        "--video_repeat",
        type=int,
        default=0,
        help="Num of repeat video.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help=(
            "The config of the model in training."
        ),
    )
    parser.add_argument(
        "--transformer_path",
        type=str,
        default=None,
        help=("If you want to load the weight from other transformers, input its path."),
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        help=("If you want to load the weight from other vaes, input its path."),
    )

    parser.add_argument(
        '--trainable_modules', 
        nargs='+', 
        help='Enter a list of trainable modules'
    )
    parser.add_argument(
        '--trainable_modules_low_learning_rate', 
        nargs='+', 
        default=[],
        help='Enter a list of trainable modules with lower learning rate'
    )
    parser.add_argument(
        '--tokenizer_max_length', 
        type=int,
        default=512,
        help='Max length of tokenizer'
    )
    parser.add_argument(
        "--use_deepspeed", action="store_true", help="Whether or not to use deepspeed."
    )
    parser.add_argument(
        "--use_fsdp", action="store_true", help="Whether or not to use fsdp."
    )
    parser.add_argument(
        "--low_vram", action="store_true", help="Whether enable low_vram mode."
    )
    parser.add_argument(
        "--train_mode",
        type=str,
        default="normal",
        help=(
            'The format of training data. Support `"normal"`'
            ' (default), `"inpaint"`.'
        ),
    )
    parser.add_argument(
        "--abnormal_norm_clip_start",
        type=int,
        default=1000,
        help=(
            'When do we start doing additional processing on abnormal gradients. '
        ),
    )
    parser.add_argument(
        "--initial_grad_norm_ratio",
        type=int,
        default=5,
        help=(
            'The initial gradient is relative to the multiple of the max_grad_norm. '
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def main():
    args = parse_args()

    # Task-55: if a Tesseract-style validation JSON is given, populate validation_prompts/paths from it
    # (extract the first frame of each clip as the I2V start image). Keeps the rest of validation unchanged.
    if getattr(args, "validation_meta", None):
        from videox_fun.data.tesseract_json import expand_validation_from_meta
        _val_frame_dir = os.path.join(args.output_dir or ".", "val_frames")
        args.validation_prompts, args.validation_paths = expand_validation_from_meta(
            args.validation_meta, _val_frame_dir)
        print(f"[Task-55] loaded {len(args.validation_prompts)} validation clips from {args.validation_meta}")
        # Task-65: keep the [caption, video_path] pairs so log_validation can load the GT clip for rgb_sample.
        try:
            from videox_fun.data.tesseract_json import load_tesseract_pairs
            _VAL_META_PAIRS[:] = load_tesseract_pairs(args.validation_meta)
        except Exception as _e:
            print(f"[VAL_VIZ] could not load validation_meta pairs: {_e}")

    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    config = OmegaConf.load(args.config_path)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    deepspeed_plugin = accelerator.state.deepspeed_plugin if hasattr(accelerator.state, "deepspeed_plugin") else None
    fsdp_plugin = accelerator.state.fsdp_plugin if hasattr(accelerator.state, "fsdp_plugin") else None
    if deepspeed_plugin is not None:
        zero_stage = int(deepspeed_plugin.zero_stage)
        fsdp_stage = 0
        print(f"Using DeepSpeed Zero stage: {zero_stage}")

        args.use_deepspeed = True
        if zero_stage == 3:
            print(f"Auto set save_state to True because zero_stage == 3")
            args.save_state = True
    elif fsdp_plugin is not None:
        from torch.distributed.fsdp import ShardingStrategy
        zero_stage = 0
        if fsdp_plugin.sharding_strategy is ShardingStrategy.FULL_SHARD:
            fsdp_stage = 3
        elif fsdp_plugin.sharding_strategy is None: # The fsdp_plugin.sharding_strategy is None in FSDP 2.
            fsdp_stage = 3
        elif fsdp_plugin.sharding_strategy is ShardingStrategy.SHARD_GRAD_OP:
            fsdp_stage = 2
        else:
            fsdp_stage = 0
        print(f"Using FSDP stage: {fsdp_stage}")

        args.use_fsdp = True
        if fsdp_stage == 3:
            print(f"Auto set save_state to True because fsdp_stage == 3")
            args.save_state = True
    else:
        zero_stage = 0
        fsdp_stage = 0
        print("DeepSpeed is not enabled.")

    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir=logging_dir)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        rng = np.random.default_rng(np.random.PCG64(args.seed + accelerator.process_index))
        torch_rng = torch.Generator(accelerator.device).manual_seed(args.seed + accelerator.process_index)
    else:
        rng = None
        torch_rng = None
    index_rng = np.random.default_rng(np.random.PCG64(43))
    print(f"Init rng with seed {args.seed + accelerator.process_index}. Process_index is {accelerator.process_index}")

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora transformer3d) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Load scheduler, tokenizer and models.
    noise_scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
    )

    # Get Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
    )

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        # Get Text encoder
        text_encoder = WanT5EncoderModel.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
            additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        )
        text_encoder = text_encoder.eval()
        # Get Vae
        vae = AutoencoderKLWan.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, config['vae_kwargs'].get('vae_subpath', 'vae')),
            additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
        )
        vae.eval()
        # Get Clip Image Encoder
        if args.train_mode != "normal":
            clip_image_encoder = CLIPModel.from_pretrained(
                os.path.join(args.pretrained_model_name_or_path, config['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')),
            )
            clip_image_encoder = clip_image_encoder.eval()
        else:
            clip_image_encoder = None
            
    # Get Transformer
    transformer3d = WanTransformer3DModel.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
    ).to(weight_dtype)

    # Freeze vae and text_encoder and set transformer3d to trainable
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    transformer3d.requires_grad_(False)
    if args.train_mode != "normal":
        clip_image_encoder.requires_grad_(False)

    # ===================== Task-42: build Geo-DiT + hook + load geo latents =====================
    geo_dit = None
    geo_latents_all = None     # legacy precomputed-file path (unused; kept for compat)
    geo_da3 = None             # on-the-fly DepthAnything-3 model
    geo_cache = {}             # idx -> geo latent [16,Fl,hl,wl] on CPU (computed once per sample)
    if _GEO:
        geo_dit = GeoDiT(
            dim=int(os.environ.get("GEO_DIM", "1536")),
            heads=int(os.environ.get("GEO_HEADS", "12")),
            depth=int(os.environ.get("GEO_DEPTH", "4")),
            in_ch=16, patch=2,
        ).to(accelerator.device, dtype=torch.float32)
        geo_dit.requires_grad_(True)
        # Forward hook: capture the (non-detached) RGB hidden state at block GEO_HOOK_LAYER.
        def _geo_capture(module, inp, out):
            _geo_state["h"] = out[0] if isinstance(out, (tuple, list)) else out
        transformer3d.blocks[_GEO_HOOK_LAYER].register_forward_hook(_geo_capture)
        # On-the-fly geometry: run DepthAnything-3 on the batch RGB during training and encode the
        # depth with the SAME Wan VAE -> 16-ch geo latents. Cached per dataset idx so DA3 runs once
        # per sample (no precomputed geo_latents.pt file -> saves disk). Override ckpt via GEO_DA3_CKPT.
        _repo_root_geo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        _da3_src = os.path.join(_repo_root_geo, "depth", "src")
        if os.path.isdir(_da3_src):
            sys.path.insert(0, _da3_src)
        from depth_anything_3.api import DepthAnything3
        # Task-57: CLI --pretrained_vision_model_path > GEO_DA3_CKPT env > hard-coded default.
        _da3_ckpt = getattr(args, "pretrained_vision_model_path", None) \
            or os.environ.get("GEO_DA3_CKPT", "depth-anything/DA3-LARGE-1.1")
        geo_da3 = DepthAnything3.from_pretrained(_da3_ckpt).to(accelerator.device).eval()
        geo_da3.requires_grad_(False)
        if accelerator.is_main_process:
            print(f"[GEO] Geo-DiT params={sum(p.numel() for p in geo_dit.parameters())} "
                  f"hook=block[{_GEO_HOOK_LAYER}] lambda={_GEO_LAMBDA} "
                  f"mode=on-the-fly-DA3 ckpt={_da3_ckpt}")
    # ===========================================================================================

    if args.transformer_path is not None:
        print(f"From checkpoint: {args.transformer_path}")
        if args.transformer_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(args.transformer_path)
        else:
            state_dict = torch.load(args.transformer_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = transformer3d.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
        assert len(u) == 0

    if args.vae_path is not None:
        print(f"From checkpoint: {args.vae_path}")
        if args.vae_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(args.vae_path)
        else:
            state_dict = torch.load(args.vae_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = vae.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
        assert len(u) == 0
    
    # A good trainable modules is showed below now.
    # For 3D Patch: trainable_modules = ['ff.net', 'pos_embed', 'attn2', 'proj_out', 'timepositionalencoding', 'h_position', 'w_position']
    # For 2D Patch: trainable_modules = ['ff.net', 'attn2', 'timepositionalencoding', 'h_position', 'w_position']
    transformer3d.train()
    if accelerator.is_main_process:
        accelerator.print(
            f"Trainable modules '{args.trainable_modules}'."
        )
    for name, param in transformer3d.named_parameters():
        for trainable_module_name in args.trainable_modules + args.trainable_modules_low_learning_rate:
            if trainable_module_name in name:
                param.requires_grad = True
                break

    # Create EMA for the transformer3d.
    if args.use_ema:
        if zero_stage == 3:
            raise NotImplementedError("FSDP does not support EMA.")

        ema_transformer3d = WanTransformer3DModel.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
            transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        ).to(weight_dtype)

        ema_transformer3d = EMAModel(ema_transformer3d.parameters(), model_cls=WanTransformer3DModel, model_config=ema_transformer3d.config)

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        if fsdp_stage != 0 or zero_stage == 3:
            def save_model_hook(models, weights, output_dir):
                accelerate_state_dict = accelerator.get_state_dict(models[-1], unwrap=True)
                if accelerator.is_main_process:
                    from safetensors.torch import save_file

                    safetensor_save_path = os.path.join(output_dir, f"diffusion_pytorch_model.safetensors")
                    accelerate_state_dict = {k: v.to(dtype=weight_dtype) for k, v in accelerate_state_dict.items()}
                    save_file(accelerate_state_dict, safetensor_save_path, metadata={"format": "pt"})

                    with open(os.path.join(output_dir, "sampler_pos_start.pkl"), 'wb') as file:
                        pickle.dump([batch_sampler.sampler._pos_start, first_epoch], file)

            def load_model_hook(models, input_dir):
                pkl_path = os.path.join(input_dir, "sampler_pos_start.pkl")
                if os.path.exists(pkl_path):
                    with open(pkl_path, 'rb') as file:
                        loaded_number, _ = pickle.load(file)
                        batch_sampler.sampler._pos_start = max(loaded_number - args.dataloader_num_workers * accelerator.num_processes * 2, 0)
                    print(f"Load pkl from {pkl_path}. Get loaded_number = {loaded_number}.")
        else:
            # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
            def save_model_hook(models, weights, output_dir):
                if accelerator.is_main_process:
                    if args.use_ema:
                        ema_transformer3d.save_pretrained(os.path.join(output_dir, "transformer_ema"))

                    models[0].save_pretrained(os.path.join(output_dir, "transformer"))
                    if not args.use_deepspeed:
                        weights.pop()

                    with open(os.path.join(output_dir, "sampler_pos_start.pkl"), 'wb') as file:
                        pickle.dump([batch_sampler.sampler._pos_start, first_epoch], file)

            def load_model_hook(models, input_dir):
                if args.use_ema:
                    ema_path = os.path.join(input_dir, "transformer_ema")
                    _, ema_kwargs = WanTransformer3DModel.load_config(ema_path, return_unused_kwargs=True)
                    load_model = WanTransformer3DModel.from_pretrained(
                        input_dir, subfolder="transformer_ema",
                        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs'])
                    )
                    load_model = EMAModel(load_model.parameters(), model_cls=WanTransformer3DModel, model_config=load_model.config)
                    load_model.load_state_dict(ema_kwargs)

                    ema_transformer3d.load_state_dict(load_model.state_dict())
                    ema_transformer3d.to(accelerator.device)
                    del load_model

                for i in range(len(models)):
                    # pop models so that they are not loaded again
                    model = models.pop()

                    # load diffusers style into model
                    load_model = WanTransformer3DModel.from_pretrained(
                        input_dir, subfolder="transformer"
                    )
                    model.register_to_config(**load_model.config)

                    model.load_state_dict(load_model.state_dict())
                    del load_model

                pkl_path = os.path.join(input_dir, "sampler_pos_start.pkl")
                if os.path.exists(pkl_path):
                    with open(pkl_path, 'rb') as file:
                        loaded_number, _ = pickle.load(file)
                        batch_sampler.sampler._pos_start = max(loaded_number - args.dataloader_num_workers * accelerator.num_processes * 2, 0)
                    print(f"Load pkl from {pkl_path}. Get loaded_number = {loaded_number}.")

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        transformer3d.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    elif args.use_came:
        try:
            from came_pytorch import CAME
        except Exception:
            raise ImportError(
                "Please install came_pytorch to use CAME. You can do so by running `pip install came_pytorch`"
            )

        optimizer_cls = CAME
    else:
        optimizer_cls = torch.optim.AdamW

    trainable_params = list(filter(lambda p: p.requires_grad, transformer3d.parameters()))
    trainable_params_optim = [
        {'params': [], 'lr': args.learning_rate},
        {'params': [], 'lr': args.learning_rate / 2},
    ]
    in_already = []
    for name, param in transformer3d.named_parameters():
        high_lr_flag = False
        if name in in_already:
            continue
        for trainable_module_name in args.trainable_modules:
            if trainable_module_name in name:
                in_already.append(name)
                high_lr_flag = True
                trainable_params_optim[0]['params'].append(param)
                if accelerator.is_main_process:
                    print(f"Set {name} to lr : {args.learning_rate}")
                break
        if high_lr_flag:
            continue
        for trainable_module_name in args.trainable_modules_low_learning_rate:
            if trainable_module_name in name:
                in_already.append(name)
                trainable_params_optim[1]['params'].append(param)
                if accelerator.is_main_process:
                    print(f"Set {name} to lr : {args.learning_rate / 2}")
                break

    # Task-42: include Geo-DiT params in the optimizer (full-lr group) and the clip list.
    if _GEO and geo_dit is not None:
        _geo_params = list(geo_dit.parameters())
        trainable_params_optim[0]['params'].extend(_geo_params)
        trainable_params = trainable_params + _geo_params
        if accelerator.is_main_process:
            print(f"[GEO] added {sum(p.numel() for p in _geo_params)} Geo-DiT params to optimizer")

    if args.use_came:
        optimizer = optimizer_cls(
            trainable_params_optim,
            lr=args.learning_rate,
            # weight_decay=args.adam_weight_decay,
            betas=(0.9, 0.999, 0.9999),
            eps=(1e-30, 1e-16)
        )
    else:
        optimizer = optimizer_cls(
            trainable_params_optim,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    # Get the training dataset
    sample_n_frames_bucket_interval = vae.config.temporal_compression_ratio
    
    if args.fix_sample_size is not None and args.enable_bucket:
        args.video_sample_size = max(max(args.fix_sample_size), args.video_sample_size)
        args.image_sample_size = max(max(args.fix_sample_size), args.image_sample_size)
        args.training_with_video_token_length = False
        args.random_hw_adapt = False

    # Get the dataset
    train_dataset = ImageVideoDataset(
        args.train_data_meta, args.train_data_dir,
        video_sample_size=args.video_sample_size, video_sample_stride=args.video_sample_stride, video_sample_n_frames=args.video_sample_n_frames, 
        video_repeat=args.video_repeat, 
        image_sample_size=args.image_sample_size,
        enable_bucket=args.enable_bucket, enable_inpaint=True if args.train_mode != "normal" else False,
    )

    def worker_init_fn(_seed):
        _seed = _seed * 256
        def _worker_init_fn(worker_id):
            print(f"worker_init_fn with {_seed + worker_id}")
            np.random.seed(_seed + worker_id)
            random.seed(_seed + worker_id)
        return _worker_init_fn
    
    if args.enable_bucket:
        aspect_ratio_sample_size = {key : [x / 512 * args.video_sample_size for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}
        batch_sampler_generator = torch.Generator().manual_seed(args.seed)
        batch_sampler = AspectRatioBatchImageVideoSampler(
            sampler=RandomSampler(train_dataset, generator=batch_sampler_generator), dataset=train_dataset.dataset, 
            batch_size=args.train_batch_size, train_folder = args.train_data_dir, drop_last=True,
            aspect_ratios=aspect_ratio_sample_size,
        )

        def collate_fn(examples):
            def get_length_to_frame_num(token_length):
                if args.image_sample_size > args.video_sample_size:
                    sample_sizes = list(range(args.video_sample_size, args.image_sample_size + 1, 128))

                    if sample_sizes[-1] != args.image_sample_size:
                        sample_sizes.append(args.image_sample_size)
                else:
                    sample_sizes = [args.image_sample_size]
                
                length_to_frame_num = {
                    sample_size: min(token_length / sample_size / sample_size, args.video_sample_n_frames) // sample_n_frames_bucket_interval * sample_n_frames_bucket_interval + 1 for sample_size in sample_sizes
                }

                return length_to_frame_num

            def get_random_downsample_ratio(sample_size, image_ratio=[],
                                            all_choices=False, rng=None):
                def _create_special_list(length):
                    if length == 1:
                        return [1.0]
                    if length >= 2:
                        first_element = 0.90
                        remaining_sum = 1.0 - first_element
                        other_elements_value = remaining_sum / (length - 1)
                        special_list = [first_element] + [other_elements_value] * (length - 1)
                        return special_list
                        
                if sample_size >= 1536:
                    number_list = [1, 1.25, 1.5, 2, 2.5, 3] + image_ratio 
                elif sample_size >= 1024:
                    number_list = [1, 1.25, 1.5, 2] + image_ratio
                elif sample_size >= 768:
                    number_list = [1, 1.25, 1.5] + image_ratio
                elif sample_size >= 512:
                    number_list = [1] + image_ratio
                else:
                    number_list = [1]

                if all_choices:
                    return number_list

                number_list_prob = np.array(_create_special_list(len(number_list)))
                if rng is None:
                    return np.random.choice(number_list, p = number_list_prob)
                else:
                    return rng.choice(number_list, p = number_list_prob)

            # Get token length
            target_token_length = args.video_sample_n_frames * args.token_sample_size * args.token_sample_size
            length_to_frame_num = get_length_to_frame_num(target_token_length)

            # Create new output
            new_examples                 = {}
            new_examples["target_token_length"] = target_token_length
            new_examples["pixel_values"] = []
            new_examples["text"]         = []
            new_examples["idx"]          = []   # Task-42: needed to fetch the matching geo latent
            # Used in Inpaint mode 
            if args.train_mode != "normal":
                new_examples["mask_pixel_values"] = []
                new_examples["mask"] = []
                new_examples["clip_pixel_values"] = []

            # Get downsample ratio in image and videos
            pixel_value     = examples[0]["pixel_values"]
            data_type       = examples[0]["data_type"]
            f, h, w, c      = np.shape(pixel_value)
            if data_type == 'image':
                random_downsample_ratio = 1 if not args.random_hw_adapt else get_random_downsample_ratio(args.image_sample_size, image_ratio=[args.image_sample_size / args.video_sample_size])

                aspect_ratio_sample_size = {key : [x / 512 * args.image_sample_size / random_downsample_ratio for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}
                aspect_ratio_random_crop_sample_size = {key : [x / 512 * args.image_sample_size / random_downsample_ratio for x in ASPECT_RATIO_RANDOM_CROP_512[key]] for key in ASPECT_RATIO_RANDOM_CROP_512.keys()}
                
                batch_video_length = args.video_sample_n_frames + sample_n_frames_bucket_interval
            else:
                if args.random_hw_adapt:
                    if args.training_with_video_token_length:
                        local_min_size = np.min(np.array([np.mean(np.array([np.shape(example["pixel_values"])[1], np.shape(example["pixel_values"])[2]])) for example in examples]))
                        # The video will be resized to a lower resolution than its own.
                        choice_list = [length for length in list(length_to_frame_num.keys()) if length < local_min_size * 1.25]
                        if len(choice_list) == 0:
                            choice_list = list(length_to_frame_num.keys())
                        local_video_sample_size = np.random.choice(choice_list)
                        batch_video_length = length_to_frame_num[local_video_sample_size]
                        random_downsample_ratio = args.video_sample_size / local_video_sample_size
                    else:
                        random_downsample_ratio = get_random_downsample_ratio(args.video_sample_size)
                        batch_video_length = args.video_sample_n_frames + sample_n_frames_bucket_interval
                else:
                    random_downsample_ratio = 1
                    batch_video_length = args.video_sample_n_frames + sample_n_frames_bucket_interval

                aspect_ratio_sample_size = {key : [x / 512 * args.video_sample_size / random_downsample_ratio for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}
                aspect_ratio_random_crop_sample_size = {key : [x / 512 * args.video_sample_size / random_downsample_ratio for x in ASPECT_RATIO_RANDOM_CROP_512[key]] for key in ASPECT_RATIO_RANDOM_CROP_512.keys()}

            if args.fix_sample_size is not None:
                fix_sample_size = [int(x / 16) * 16 for x in args.fix_sample_size]
            elif args.random_ratio_crop:
                if rng is None:
                    random_sample_size = aspect_ratio_random_crop_sample_size[
                        np.random.choice(list(aspect_ratio_random_crop_sample_size.keys()), p = ASPECT_RATIO_RANDOM_CROP_PROB)
                    ]
                else:
                    random_sample_size = aspect_ratio_random_crop_sample_size[
                        rng.choice(list(aspect_ratio_random_crop_sample_size.keys()), p = ASPECT_RATIO_RANDOM_CROP_PROB)
                    ]
                random_sample_size = [int(x / 16) * 16 for x in random_sample_size]
            else:
                closest_size, closest_ratio = get_closest_ratio(h, w, ratios=aspect_ratio_sample_size)
                closest_size = [int(x / 16) * 16 for x in closest_size]

            min_example_length = min(
                [example["pixel_values"].shape[0] for example in examples]
            )
            batch_video_length = int(min(batch_video_length, min_example_length))
            
            # Magvae needs the number of frames to be 4n + 1.
            batch_video_length = (batch_video_length - 1) // sample_n_frames_bucket_interval * sample_n_frames_bucket_interval + 1

            if batch_video_length <= 0:
                batch_video_length = 1

            for example in examples:
                if args.fix_sample_size is not None:
                    # To 0~1
                    pixel_values = torch.from_numpy(example["pixel_values"]).permute(0, 3, 1, 2).contiguous()
                    pixel_values = pixel_values / 255.

                    # Get adapt hw for resize
                    fix_sample_size = list(map(lambda x: int(x), fix_sample_size))
                    transform = transforms.Compose([
                        transforms.Resize(fix_sample_size, interpolation=transforms.InterpolationMode.BILINEAR),  # Image.BICUBIC
                        transforms.CenterCrop(fix_sample_size),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
                    ])
                elif args.random_ratio_crop:
                    # To 0~1
                    pixel_values = torch.from_numpy(example["pixel_values"]).permute(0, 3, 1, 2).contiguous()
                    pixel_values = pixel_values / 255.

                    # Get adapt hw for resize
                    b, c, h, w = pixel_values.size()
                    th, tw = random_sample_size
                    if th / tw > h / w:
                        nh = int(th)
                        nw = int(w / h * nh)
                    else:
                        nw = int(tw)
                        nh = int(h / w * nw)
                    
                    transform = transforms.Compose([
                        transforms.Resize([nh, nw]),
                        transforms.CenterCrop([int(x) for x in random_sample_size]),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
                    ])
                else:
                    # To 0~1
                    pixel_values = torch.from_numpy(example["pixel_values"]).permute(0, 3, 1, 2).contiguous()
                    pixel_values = pixel_values / 255.

                    # Get adapt hw for resize
                    closest_size = list(map(lambda x: int(x), closest_size))
                    if closest_size[0] / h > closest_size[1] / w:
                        resize_size = closest_size[0], int(w * closest_size[0] / h)
                    else:
                        resize_size = int(h * closest_size[1] / w), closest_size[1]
                    
                    transform = transforms.Compose([
                        transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BILINEAR),  # Image.BICUBIC
                        transforms.CenterCrop(closest_size),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
                    ])

                new_examples["pixel_values"].append(transform(pixel_values)[:batch_video_length])
                new_examples["text"].append(example["text"])
                new_examples["idx"].append(int(example["idx"]))   # Task-42

                if args.train_mode != "normal":
                    mask = get_random_mask(new_examples["pixel_values"][-1].size())
                    mask_pixel_values = new_examples["pixel_values"][-1] * (1 - mask) 
                    # Wan 2.1 use 0 for masked pixels
                    # + torch.ones_like(new_examples["pixel_values"][-1]) * -1 * mask
                    new_examples["mask_pixel_values"].append(mask_pixel_values)
                    new_examples["mask"].append(mask)
                    
                    clip_pixel_values = new_examples["pixel_values"][-1][0].permute(1, 2, 0).contiguous()
                    clip_pixel_values = (clip_pixel_values * 0.5 + 0.5) * 255
                    new_examples["clip_pixel_values"].append(clip_pixel_values)

            # Limit the number of frames to the same
            new_examples["pixel_values"] = torch.stack([example for example in new_examples["pixel_values"]])
            if args.train_mode != "normal":
                new_examples["mask_pixel_values"] = torch.stack([example for example in new_examples["mask_pixel_values"]])
                new_examples["mask"] = torch.stack([example for example in new_examples["mask"]])
                new_examples["clip_pixel_values"] = torch.stack([example for example in new_examples["clip_pixel_values"]])

            # Encode prompts when enable_text_encoder_in_dataloader=True
            if args.enable_text_encoder_in_dataloader:
                prompt_ids = tokenizer(
                    new_examples['text'], 
                    max_length=args.tokenizer_max_length, 
                    padding="max_length", 
                    add_special_tokens=True, 
                    truncation=True, 
                    return_tensors="pt"
                )
                encoder_hidden_states = text_encoder(
                    prompt_ids.input_ids
                )[0]
                new_examples['encoder_attention_mask'] = prompt_ids.attention_mask
                new_examples['encoder_hidden_states'] = encoder_hidden_states

            return new_examples
        
        # DataLoaders creation:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            persistent_workers=True if args.dataloader_num_workers != 0 else False,
            num_workers=args.dataloader_num_workers,
            worker_init_fn=worker_init_fn(args.seed + accelerator.process_index)
        )
    else:
        # DataLoaders creation:
        batch_sampler_generator = torch.Generator().manual_seed(args.seed)
        batch_sampler = ImageVideoSampler(RandomSampler(train_dataset, generator=batch_sampler_generator), train_dataset, args.train_batch_size)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=batch_sampler, 
            persistent_workers=True if args.dataloader_num_workers != 0 else False,
            num_workers=args.dataloader_num_workers,
            worker_init_fn=worker_init_fn(args.seed + accelerator.process_index)
        )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    transformer3d, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer3d, optimizer, train_dataloader, lr_scheduler
    )

    if fsdp_stage != 0 or zero_stage != 0:
        from functools import partial

        from videox_fun.dist import set_multi_gpus_devices, shard_model
        shard_fn = partial(shard_model, device_id=accelerator.device, param_dtype=weight_dtype)
        text_encoder = shard_fn(text_encoder)

    if args.use_ema:
        ema_transformer3d.to(accelerator.device)

    # Move text_encode and vae to gpu and cast to weight_dtype
    vae.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)
    if not args.enable_text_encoder_in_dataloader:
        text_encoder.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)
    if args.train_mode != "normal":
        clip_image_encoder.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        keys_to_pop = [k for k, v in tracker_config.items() if isinstance(v, list)]
        for k in keys_to_pop:
            tracker_config.pop(k)
            print(f"Removed tracker_config['{k}']")
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    # Task-67: simplified full-state checkpoint dir (<output>/checkpoints) with step_NNNNNN + `latest`
    # and auto-resume. Gated by SIMPLE_CKPT_DIR; when unset, behaviour is exactly as before. Restores the
    # Geo-DiT weights from geo_dit.pt (load_state only restores transformer3d + optimizer/scheduler).
    _simple_ckpt_dir = os.environ.get("SIMPLE_CKPT_DIR", "")
    if _simple_ckpt_dir:
        import sys as _sys
        from pathlib import Path as _Path
        _sys.path.insert(0, str(_Path(__file__).resolve().parents[3] / "tools"))
        import overfit_ckpt as _ofc
        _latest = _ofc.find_latest(_simple_ckpt_dir)
        if _latest is None:
            accelerator.print(f"[simple-ckpt] no checkpoint under {_simple_ckpt_dir}; starting fresh")
            initial_global_step = 0
        else:
            global_step, _lpath = _latest
            initial_global_step = global_step
            pkl_path = os.path.join(_lpath, "sampler_pos_start.pkl")
            if os.path.exists(pkl_path):
                with open(pkl_path, 'rb') as file:
                    _, first_epoch = pickle.load(file)
            else:
                first_epoch = global_step // num_update_steps_per_epoch
            accelerator.print(f"[simple-ckpt] auto-resume from {_lpath} (step {global_step})")
            accelerator.load_state(_lpath)
            _geo_pt = os.path.join(_lpath, "geo_dit.pt")
            if geo_dit is not None and os.path.isfile(_geo_pt):
                accelerator.unwrap_model(geo_dit).load_state_dict(torch.load(_geo_pt, map_location="cpu"))
                accelerator.print(f"[simple-ckpt] restored Geo-DiT weights from {_geo_pt}")
    elif args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            global_step = int(path.split("-")[1])

            initial_global_step = global_step

            pkl_path = os.path.join(os.path.join(args.output_dir, path), "sampler_pos_start.pkl")
            if os.path.exists(pkl_path):
                with open(pkl_path, 'rb') as file:
                    _, first_epoch = pickle.load(file)
            else:
                first_epoch = global_step // num_update_steps_per_epoch
            print(f"Load pkl from {pkl_path}. Get first_epoch = {first_epoch}.")

            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            # Task-67 fix: load_state does NOT restore Geo-DiT (it is not an accelerate-tracked model).
            _geo_pt = os.path.join(args.output_dir, path, "geo_dit.pt")
            if geo_dit is not None and os.path.isfile(_geo_pt):
                accelerator.unwrap_model(geo_dit).load_state_dict(torch.load(_geo_pt, map_location="cpu"))
                accelerator.print(f"[resume] restored Geo-DiT weights from {_geo_pt}")
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    if args.multi_stream and args.train_mode != "normal":
        # create extra cuda streams to speedup inpaint vae computation
        vae_stream_1 = torch.cuda.Stream()
        vae_stream_2 = torch.cuda.Stream()
    else:
        vae_stream_1 = None
        vae_stream_2 = None

    idx_sampling = DiscreteSampling(args.train_sampling_steps, uniform_sampling=args.uniform_sampling)

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        batch_sampler.sampler.generator = torch.Generator().manual_seed(args.seed + epoch)
        for step, batch in enumerate(train_dataloader):
            # Data batch sanity check
            if epoch == first_epoch and step == 0:
                pixel_values, texts = batch['pixel_values'].cpu(), batch['text']
                pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
                os.makedirs(os.path.join(args.output_dir, "sanity_check"), exist_ok=True)
                for idx, (pixel_value, text) in enumerate(zip(pixel_values, texts)):
                    pixel_value = pixel_value[None, ...]
                    gif_name = '-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{global_step}-{idx}'
                    save_videos_grid(pixel_value, f"{args.output_dir}/sanity_check/{gif_name[:10]}.mp4", rescale=True)
                if args.train_mode != "normal":
                    clip_pixel_values, mask_pixel_values, texts = batch['clip_pixel_values'].cpu(), batch['mask_pixel_values'].cpu(), batch['text']
                    mask_pixel_values = rearrange(mask_pixel_values, "b f c h w -> b c f h w")
                    for idx, (clip_pixel_value, pixel_value, text) in enumerate(zip(clip_pixel_values, mask_pixel_values, texts)):
                        pixel_value = pixel_value[None, ...]
                        Image.fromarray(np.uint8(clip_pixel_value)).save(f"{args.output_dir}/sanity_check/clip_{gif_name[:10] if not text == '' else f'{global_step}-{idx}'}.png")
                        save_videos_grid(pixel_value, f"{args.output_dir}/sanity_check/mask_{gif_name[:10] if not text == '' else f'{global_step}-{idx}'}.mp4", rescale=True)

            with accelerator.accumulate(transformer3d):
                # Convert images to latent space
                pixel_values = batch["pixel_values"].to(weight_dtype)

                # Increase the batch size when the length of the latent sequence of the current sample is small
                if args.auto_tile_batch_size and args.training_with_video_token_length and zero_stage != 3:
                    if args.video_sample_n_frames * args.token_sample_size * args.token_sample_size // 16 >= pixel_values.size()[1] * pixel_values.size()[3] * pixel_values.size()[4]:
                        pixel_values = torch.tile(pixel_values, (4, 1, 1, 1, 1))
                        if args.enable_text_encoder_in_dataloader:
                            batch['encoder_hidden_states'] = torch.tile(batch['encoder_hidden_states'], (4, 1, 1))
                            batch['encoder_attention_mask'] = torch.tile(batch['encoder_attention_mask'], (4, 1))
                        else:
                            batch['text'] = batch['text'] * 4
                    elif args.video_sample_n_frames * args.token_sample_size * args.token_sample_size // 4 >= pixel_values.size()[1] * pixel_values.size()[3] * pixel_values.size()[4]:
                        pixel_values = torch.tile(pixel_values, (2, 1, 1, 1, 1))
                        if args.enable_text_encoder_in_dataloader:
                            batch['encoder_hidden_states'] = torch.tile(batch['encoder_hidden_states'], (2, 1, 1))
                            batch['encoder_attention_mask'] = torch.tile(batch['encoder_attention_mask'], (2, 1))
                        else:
                            batch['text'] = batch['text'] * 2
                
                if args.train_mode != "normal":
                    clip_pixel_values = batch["clip_pixel_values"].to(weight_dtype)
                    mask_pixel_values = batch["mask_pixel_values"].to(weight_dtype)
                    mask = batch["mask"].to(weight_dtype)
                    # Increase the batch size when the length of the latent sequence of the current sample is small
                    if args.auto_tile_batch_size and args.training_with_video_token_length and zero_stage != 3:
                        if args.video_sample_n_frames * args.token_sample_size * args.token_sample_size // 16 >= pixel_values.size()[1] * pixel_values.size()[3] * pixel_values.size()[4]:
                            clip_pixel_values = torch.tile(clip_pixel_values, (4, 1, 1, 1))
                            mask_pixel_values = torch.tile(mask_pixel_values, (4, 1, 1, 1, 1))
                            mask = torch.tile(mask, (4, 1, 1, 1, 1))
                        elif args.video_sample_n_frames * args.token_sample_size * args.token_sample_size // 4 >= pixel_values.size()[1] * pixel_values.size()[3] * pixel_values.size()[4]:
                            clip_pixel_values = torch.tile(clip_pixel_values, (2, 1, 1, 1))
                            mask_pixel_values = torch.tile(mask_pixel_values, (2, 1, 1, 1, 1))
                            mask = torch.tile(mask, (2, 1, 1, 1, 1))

                if args.random_frame_crop:
                    def _create_special_list(length):
                        if length == 1:
                            return [1.0]
                        if length >= 2:
                            last_element = 0.90
                            remaining_sum = 1.0 - last_element
                            other_elements_value = remaining_sum / (length - 1)
                            special_list = [other_elements_value] * (length - 1) + [last_element]
                            return special_list
                    select_frames = [_tmp for _tmp in list(range(sample_n_frames_bucket_interval + 1, args.video_sample_n_frames + sample_n_frames_bucket_interval, sample_n_frames_bucket_interval))]
                    select_frames_prob = np.array(_create_special_list(len(select_frames)))
                    
                    if len(select_frames) != 0:
                        if rng is None:
                            temp_n_frames = np.random.choice(select_frames, p = select_frames_prob)
                        else:
                            temp_n_frames = rng.choice(select_frames, p = select_frames_prob)
                    else:
                        temp_n_frames = 1

                    # Magvae needs the number of frames to be 4n + 1.
                    temp_n_frames = (temp_n_frames - 1) // sample_n_frames_bucket_interval + 1

                    pixel_values = pixel_values[:, :temp_n_frames, :, :]

                    if args.train_mode != "normal":
                        mask_pixel_values = mask_pixel_values[:, :temp_n_frames, :, :]
                        mask = mask[:, :temp_n_frames, :, :]
                    
                # Keep all node same token length to accelerate the traning when resolution grows.
                if args.keep_all_node_same_token_length:
                    if args.token_sample_size > 256:
                        numbers_list = list(range(256, args.token_sample_size + 1, 128))

                        if numbers_list[-1] != args.token_sample_size:
                            numbers_list.append(args.token_sample_size)
                    else:
                        numbers_list = [256]
                    numbers_list = [_number * _number * args.video_sample_n_frames for _number in  numbers_list]
            
                    actual_token_length = index_rng.choice(numbers_list)
                    actual_video_length = (min(
                            actual_token_length / pixel_values.size()[-1] / pixel_values.size()[-2], args.video_sample_n_frames
                    ) - 1) // sample_n_frames_bucket_interval * sample_n_frames_bucket_interval + 1
                    actual_video_length = int(max(actual_video_length, 1))

                    # Magvae needs the number of frames to be 4n + 1.
                    actual_video_length = (actual_video_length - 1) // sample_n_frames_bucket_interval + 1

                    pixel_values = pixel_values[:, :actual_video_length, :, :]
                    if args.train_mode != "normal":
                        mask_pixel_values = mask_pixel_values[:, :actual_video_length, :, :]
                        mask = mask[:, :actual_video_length, :, :]

                # Make the inpaint latents to be zeros.
                if args.train_mode != "normal":
                    t2v_flag = [(_mask == 1).all() for _mask in mask]
                    new_t2v_flag = []
                    for _mask in t2v_flag:
                        if _mask and np.random.rand() < 0.90:
                            new_t2v_flag.append(0)
                        else:
                            new_t2v_flag.append(1)
                    t2v_flag = torch.from_numpy(np.array(new_t2v_flag)).to(accelerator.device, dtype=weight_dtype)

                if args.low_vram:
                    torch.cuda.empty_cache()
                    vae.to(accelerator.device)
                    if args.train_mode != "normal":
                        clip_image_encoder.to(accelerator.device)
                    if not args.enable_text_encoder_in_dataloader:
                        text_encoder.to("cpu")

                with torch.no_grad():
                    # This way is quicker when batch grows up
                    def _batch_encode_vae(pixel_values):
                        pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
                        bs = args.vae_mini_batch
                        new_pixel_values = []
                        for i in range(0, pixel_values.shape[0], bs):
                            pixel_values_bs = pixel_values[i : i + bs]
                            pixel_values_bs = vae.encode(pixel_values_bs)[0]
                            pixel_values_bs = pixel_values_bs.sample()
                            new_pixel_values.append(pixel_values_bs)
                        return torch.cat(new_pixel_values, dim = 0)
                    if vae_stream_1 is not None:
                        vae_stream_1.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(vae_stream_1):
                            latents = _batch_encode_vae(pixel_values)
                    else:
                        latents = _batch_encode_vae(pixel_values)

                    if args.train_mode != "normal":
                        mask = rearrange(mask, "b f c h w -> b c f h w")
                        mask = torch.concat(
                            [
                                torch.repeat_interleave(mask[:, :, 0:1], repeats=4, dim=2), 
                                mask[:, :, 1:]
                            ], dim=2
                        )
                        mask = mask.view(mask.shape[0], mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4])
                        mask = mask.transpose(1, 2)
                        mask = resize_mask(1 - mask, latents)

                        # Encode inpaint latents.
                        mask_latents = _batch_encode_vae(mask_pixel_values)
                        if vae_stream_2 is not None:
                            torch.cuda.current_stream().wait_stream(vae_stream_2) 

                        inpaint_latents = torch.concat([mask, mask_latents], dim=1)
                        inpaint_latents = t2v_flag[:, None, None, None, None] * inpaint_latents

                        clip_context = []
                        for clip_pixel_value in clip_pixel_values:
                            clip_image = Image.fromarray(np.uint8(clip_pixel_value.float().cpu().numpy()))
                            clip_image = TF.to_tensor(clip_image).sub_(0.5).div_(0.5).to(clip_image_encoder.device, weight_dtype)
                            _clip_context = clip_image_encoder([clip_image[:, None, :, :]])

                            if rng is None:
                                zero_init_clip_in = np.random.choice([True, False], p=[0.1, 0.9])
                            else:
                                zero_init_clip_in = rng.choice([True, False], p=[0.1, 0.9])
                            clip_context.append(_clip_context if not zero_init_clip_in else torch.zeros_like(_clip_context))
                            
                        clip_context = torch.cat(clip_context)
                                                
                # wait for latents = vae.encode(pixel_values) to complete
                if vae_stream_1 is not None:
                    torch.cuda.current_stream().wait_stream(vae_stream_1)

                if args.low_vram:
                    vae.to('cpu')
                    if args.train_mode != "normal":
                        clip_image_encoder.to('cpu')
                    torch.cuda.empty_cache()
                    if not args.enable_text_encoder_in_dataloader:
                        text_encoder.to(accelerator.device)

                if args.enable_text_encoder_in_dataloader:
                    prompt_embeds = batch['encoder_hidden_states'].to(device=latents.device)
                else:
                    with torch.no_grad():
                        prompt_ids = tokenizer(
                            batch['text'], 
                            padding="max_length", 
                            max_length=args.tokenizer_max_length, 
                            truncation=True, 
                            add_special_tokens=True, 
                            return_tensors="pt"
                        )
                        text_input_ids = prompt_ids.input_ids
                        prompt_attention_mask = prompt_ids.attention_mask

                        seq_lens = prompt_attention_mask.gt(0).sum(dim=1).long()
                        prompt_embeds = text_encoder(text_input_ids.to(latents.device), attention_mask=prompt_attention_mask.to(latents.device))[0]
                        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]

                if args.low_vram and not args.enable_text_encoder_in_dataloader:
                    text_encoder.to('cpu')
                    torch.cuda.empty_cache()

                bsz, channel, num_frames, height, width = latents.size()
                noise = torch.randn(latents.size(), device=latents.device, generator=torch_rng, dtype=weight_dtype)

                if not args.uniform_sampling:
                    u = compute_density_for_timestep_sampling(
                        weighting_scheme=args.weighting_scheme,
                        batch_size=bsz,
                        logit_mean=args.logit_mean,
                        logit_std=args.logit_std,
                        mode_scale=args.mode_scale,
                    )
                    indices = (u * noise_scheduler.config.num_train_timesteps).long()
                else:
                    # Sample a random timestep for each image
                    # timesteps = generate_timestep_with_lognorm(0, args.train_sampling_steps, (bsz,), device=latents.device, generator=torch_rng)
                    # timesteps = torch.randint(0, args.train_sampling_steps, (bsz,), device=latents.device, generator=torch_rng)
                    indices = idx_sampling(bsz, generator=torch_rng, device=latents.device)
                    indices = indices.long().cpu()
                timesteps = noise_scheduler.timesteps[indices].to(device=latents.device)

                def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
                    sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
                    schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
                    timesteps = timesteps.to(accelerator.device)
                    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

                    sigma = sigmas[step_indices].flatten()
                    while len(sigma.shape) < n_dim:
                        sigma = sigma.unsqueeze(-1)
                    return sigma

                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
                noisy_latents = (1.0 - sigmas) * latents + sigmas * noise

                # Add noise
                target = noise - latents
                
                target_shape = (vae.latent_channels, num_frames, width, height)
                seq_len = math.ceil(
                    (target_shape[2] * target_shape[3]) /
                    (accelerator.unwrap_model(transformer3d).config.patch_size[1] * accelerator.unwrap_model(transformer3d).config.patch_size[2]) *
                    target_shape[1]
                )

                # Predict the noise residual
                with torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=accelerator.device):
                    noise_pred = transformer3d(
                        x=noisy_latents,
                        context=prompt_embeds,
                        t=timesteps,
                        seq_len=seq_len,
                        y=inpaint_latents if args.train_mode != "normal" else None,
                        clip_fea=clip_context if args.train_mode != "normal" else None,
                    )
                
                def custom_mse_loss(noise_pred, target, weighting=None, threshold=50):
                    noise_pred = noise_pred.float()
                    target = target.float()
                    diff = noise_pred - target
                    mse_loss = F.mse_loss(noise_pred, target, reduction='none')
                    mask = (diff.abs() <= threshold).float()
                    masked_loss = mse_loss * mask
                    if weighting is not None:
                        masked_loss = masked_loss * weighting
                    final_loss = masked_loss.mean()
                    return final_loss
                
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                loss = custom_mse_loss(noise_pred.float(), target.float(), weighting.float())
                loss = loss.mean()

                # ===================== Task-42: ref15-style geometry branch =====================
                rgb_loss_val = loss.detach().item()
                geo_loss_val = 0.0
                if _GEO and geo_da3 is not None and _geo_state["h"] is not None:
                    bidx = list(batch["idx"])
                    B = latents.size(0)
                    if len(bidx) < B:                      # safety vs auto-tiling (unused here)
                        bidx = (bidx * ((B // len(bidx)) + 1))[:B]
                    # ---- on-the-fly geo latents: DA3 depth -> Wan VAE, cached per idx (no file) ----
                    _zg = []
                    for _bi in range(B):
                        _key = int(bidx[_bi])
                        _z = geo_cache.get(_key, None)
                        if _z is None:
                            with torch.no_grad():
                                _rgb = pixel_values[_bi]                              # [F,c,h,w] [-1,1] bf16
                                _F = _rgb.shape[0]
                                _rgb01 = (_rgb.float() * 0.5 + 0.5).clamp(0, 1)       # [F,c,h,w]
                                _np = (_rgb01.permute(0, 2, 3, 1) * 255.0).round().byte().cpu().numpy()  # [F,h,w,3]
                                _frames = [_np[k] for k in range(_F)]
                                # Task-48: geometry-model feature layer is configurable. -1 => use DA3
                                # FINAL DEPTH (Wan default, validated). >=0 => use that DA3 feature layer,
                                # reduced to a 1-ch geometry map (guarded; falls back to depth on any issue).
                                _fl = [_GEO_DA3_FEAT_LAYER] if _GEO_DA3_FEAT_LAYER >= 0 else []
                                _pred = geo_da3.inference(_frames, export_feat_layers=_fl)
                                _d = None
                                if _GEO_DA3_FEAT_LAYER >= 0:
                                    try:
                                        _feat = getattr(_pred, "features", None) or getattr(_pred, "feat", None)
                                        if isinstance(_feat, dict):
                                            _feat = _feat[_GEO_DA3_FEAT_LAYER] if _GEO_DA3_FEAT_LAYER in _feat else list(_feat.values())[0]
                                        if isinstance(_feat, (list, tuple)):
                                            _feat = _feat[-1]
                                        _ft = _feat if isinstance(_feat, torch.Tensor) else torch.as_tensor(np.asarray(_feat))
                                        _d = _ft.float().mean(dim=-1)            # collapse channels -> [F,h,w]-ish
                                        _d = _d.reshape(_F, *_d.shape[-2:])
                                    except Exception as _e:
                                        print(f"[GEO] DA3 feat layer {_GEO_DA3_FEAT_LAYER} unavailable ({_e}); using depth")
                                        _d = None
                                if _d is None:
                                    _d = _pred.depth
                                _d = _d.detach().float() if isinstance(_d, torch.Tensor) \
                                     else torch.from_numpy(np.asarray(_d, dtype=np.float32))
                                _d = _d.to(latents.device)                           # [F,h0,w0]
                                _H, _W = _rgb.shape[-2], _rgb.shape[-1]
                                _d = torch.nn.functional.interpolate(
                                    _d.unsqueeze(1), size=(_H, _W), mode="bilinear", align_corners=False).squeeze(1)
                                _dmin, _dmax = _d.amin(), _d.amax()
                                _d01 = (_d - _dmin) / (_dmax - _dmin + 1e-6)          # per-clip [0,1]
                                _pv = _d01.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1, 1)   # [1,3,F,H,W]
                                _pv = (_pv * 2.0 - 1.0).to(latents.device, dtype=pixel_values.dtype)
                                vae.to(latents.device)   # low_vram offloads vae to CPU after RGB encode
                                _zl = vae.encode(_pv)[0].sample()[0]                  # [16,Fl,hl,wl]
                            _z = _zl.detach().cpu()
                            geo_cache[_key] = _z
                        _zg.append(_z)
                    z_geo = torch.stack(_zg, dim=0).to(latents.device, dtype=latents.dtype)  # [B,16,Fl,hl,wl]
                    # Align geo latent's temporal dim to the actual RGB latent frame count (the bucket /
                    # token-length collate can yield fewer frames than the cached compute).
                    if z_geo.shape[2] != latents.shape[2]:
                        z_geo = torch.nn.functional.interpolate(
                            z_geo, size=(latents.shape[2], z_geo.shape[3], z_geo.shape[4]),
                            mode="trilinear", align_corners=False)
                    # Task-48: coupled (default) => geo shares the RGB eps; else an independent draw.
                    _geo_eps = noise if _GEO_COUPLED else torch.randn_like(noise)
                    noisy_geo = (1.0 - sigmas) * z_geo + sigmas * _geo_eps
                    geo_target = _geo_eps - z_geo                         # flow-matching velocity target
                    h_rgb = _geo_state["h"]                               # [B, seq, dim] (non-detached)
                    Fl, Hl, Wl = z_geo.shape[2], z_geo.shape[3], z_geo.shape[4]
                    zt = noisy_geo.permute(0, 2, 1, 3, 4).contiguous()    # [B,Fl,16,Hl,Wl]
                    eps_hat = (noisy_latents + (1.0 - sigmas) * noise_pred).detach().permute(0, 2, 1, 3, 4).contiguous()
                    v_geo = geo_dit(h_rgb.float(), timesteps.float(), (Fl, Hl, Wl), 'masked',
                                    z_t_geo=zt.float(), eps_rgb_hat=eps_hat.float())     # [B,Fl,16,Hl,Wl]
                    v_geo = v_geo.permute(0, 2, 1, 3, 4)                  # [B,16,Fl,Hl,Wl]
                    geo_loss = custom_mse_loss(v_geo.float(), geo_target.float(), weighting.float())
                    geo_ramp = min(1.0, global_step / _GEO_WARMUP)       # Task-48: configurable warm-up
                    geo_loss_val = geo_loss.detach().item()
                    loss = loss + _GEO_LAMBDA * geo_ramp * geo_loss
                    # ---- Task-65: stash the σ-closest-to-VAL_VIZ_SIGMA training-batch geo recon since the
                    # last validation, for the standardized geo_sample0.mp4 written in log_validation. Cheap
                    # (one-step z0 reconstruction, no grad); only when VAL_VIZ_DIR is set.
                    if _VAL_VIZ_DIR and accelerator.is_main_process:
                        try:
                            _sc = abs(float(sigmas.flatten()[0]) - _GEO_VIZ_SIGMA)
                            if (not _GEO_VALVIZ_STASH) or _sc < _GEO_VALVIZ_STASH.get("sig_close", 9e9):
                                with torch.no_grad():
                                    _z0p = (noisy_geo - sigmas * v_geo)[0].detach().float().cpu()  # [16,Fl,hl,wl]
                                _GEO_VALVIZ_STASH.clear()
                                _GEO_VALVIZ_STASH.update({
                                    "z_geo": z_geo[0].detach().float().cpu(), "z0_pred": _z0p,
                                    "sig_close": _sc, "rgb_loss": rgb_loss_val, "geo_loss": geo_loss_val})
                        except Exception as _e:
                            print(f"[VAL_VIZ] geo stash skipped at step {global_step}: {_e}")
                    # ---- Task-51/54: end-of-training geometry eval+viz dump. Per sample idx, keep the
                    # occurrence whose (coupled) sigma is closest to _GEO_VIZ_SIGMA (an informative,
                    # low-noise level) so z0_pred is a clean, in-distribution reconstruction of the
                    # trained geo_dit. Save raw z_geo (GT) + z0_pred + GT RGB; record per-sample MSE+cosine.
                    if _GEO_VIZ and _GEO_VIZ_DIR and accelerator.is_main_process \
                       and global_step >= (args.max_train_steps - _GEO_VIZ_WINDOW):
                        try:
                            os.makedirs(_GEO_VIZ_DIR, exist_ok=True)
                            with torch.no_grad():
                                z0_pred = noisy_geo - sigmas * v_geo          # recover clean geo latent (z0)
                            for _bi in range(z_geo.size(0)):
                                _k = int(bidx[_bi]); _fp = os.path.join(_GEO_VIZ_DIR, f"geo_{_k}.pt")
                                _sig = float(sigmas[_bi].flatten()[0])
                                _closeness = abs(_sig - _GEO_VIZ_SIGMA)
                                if os.path.exists(_fp):
                                    _prev = torch.load(_fp, map_location="cpu").get("sigma_close", 9e9)
                                    if _closeness >= _prev:
                                        continue                              # keep the better (σ-closer) one
                                _zg = z_geo[_bi].detach().float()
                                _zp = z0_pred[_bi].detach().float()
                                _mse = float(((_zp - _zg) ** 2).mean().item())
                                _cos = float(torch.nn.functional.cosine_similarity(
                                    _zp.flatten().unsqueeze(0), _zg.flatten().unsqueeze(0)).item())
                                _rgb_u8 = ((pixel_values[_bi].float() * 0.5 + 0.5).clamp(0, 1)
                                           * 255.0).round().byte().cpu()       # [F,c,h,w] uint8
                                torch.save({"idx": _k, "step": int(global_step), "sigma": _sig,
                                            "sigma_close": _closeness,
                                            "z_geo": _zg.cpu(), "z0_pred": _zp.cpu(),
                                            "mse": _mse, "cosine": _cos, "rgb_u8": _rgb_u8,
                                            "text": batch.get("text", [""])[_bi] if isinstance(batch.get("text", None), list) else ""},
                                           _fp)
                                print(f"[GEO_VIZ] idx={_k} step={global_step} sigma={_sig:.3f} "
                                      f"mse={_mse:.4f} cos={_cos:.4f} -> {_fp}")
                        except Exception as _e:
                            print(f"[GEO_VIZ] dump skipped at step {global_step}: {_e}")
                # ================================================================================

                if args.motion_sub_loss and noise_pred.size()[2] > 2:
                    gt_sub_noise = noise_pred[:, :, 1:].float() - noise_pred[:, :, :-1].float()
                    pre_sub_noise = target[:, :, 1:].float() - target[:, :, :-1].float()
                    sub_loss = F.mse_loss(gt_sub_noise, pre_sub_noise, reduction="mean")
                    loss = loss * (1 - args.motion_sub_loss_ratio) + sub_loss * args.motion_sub_loss_ratio

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if not args.use_deepspeed and not args.use_fsdp:
                        trainable_params_grads = [p.grad for p in trainable_params if p.grad is not None]
                        trainable_params_total_norm = torch.norm(torch.stack([torch.norm(g.detach(), 2) for g in trainable_params_grads]), 2)
                        max_grad_norm = linear_decay(args.max_grad_norm * args.initial_grad_norm_ratio, args.max_grad_norm, args.abnormal_norm_clip_start, global_step)
                        if trainable_params_total_norm / max_grad_norm > 5 and global_step > args.abnormal_norm_clip_start:
                            actual_max_grad_norm = max_grad_norm / min((trainable_params_total_norm / max_grad_norm), 10)
                        else:
                            actual_max_grad_norm = max_grad_norm
                    else:
                        actual_max_grad_norm = args.max_grad_norm

                    if not args.use_deepspeed and not args.use_fsdp and args.report_model_info and accelerator.is_main_process:
                        if trainable_params_total_norm > 1 and global_step > args.abnormal_norm_clip_start:
                            for name, param in transformer3d.named_parameters():
                                if param.requires_grad:
                                    writer.add_scalar(f'gradients/before_clip_norm/{name}', param.grad.norm(), global_step=global_step)

                    norm_sum = accelerator.clip_grad_norm_(trainable_params, actual_max_grad_norm)
                    if not args.use_deepspeed and not args.use_fsdp and args.report_model_info and accelerator.is_main_process:
                        writer.add_scalar(f'gradients/norm_sum', norm_sum, global_step=global_step)
                        writer.add_scalar(f'gradients/actual_max_grad_norm', actual_max_grad_norm, global_step=global_step)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:

                if args.use_ema:
                    ema_transformer3d.step(transformer3d.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if args.use_deepspeed or args.use_fsdp or accelerator.is_main_process:
                        gc.collect()
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                        if _simple_ckpt_dir:
                            # Task-67: full-state checkpoint -> <SIMPLE_CKPT_DIR>/step_NNNNNN + `latest`
                            # (transformer3d + optimizer/scheduler via save_state; Geo-DiT via geo_dit.pt).
                            import sys as _sys
                            from pathlib import Path as _Path
                            _sys.path.insert(0, str(_Path(__file__).resolve().parents[3] / "tools"))
                            import overfit_ckpt as _ofc
                            save_path = _ofc.step_dir(_simple_ckpt_dir, global_step)
                            accelerator.save_state(save_path)
                            if _GEO and geo_dit is not None and accelerator.is_main_process:
                                torch.save(accelerator.unwrap_model(geo_dit).state_dict(),
                                           os.path.join(save_path, "geo_dit.pt"))
                            if accelerator.is_main_process:
                                _ofc.update_latest(_simple_ckpt_dir, global_step)
                                _rm = _ofc.prune(_simple_ckpt_dir,
                                                 keep_topk=int(os.environ.get("CKPT_TOPK", str(args.checkpoints_total_limit or 2))),
                                                 keep_interval=int(os.environ.get("CKPT_KEEP_INTERVAL", str(getattr(args, "checkpoint_keep_interval", 0) or 0))))
                                logger.info(f"[simple-ckpt] saved step_{global_step:06d} (pruned: {_rm})")
                        else:
                            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                            accelerator.save_state(save_path)
                            if _GEO and geo_dit is not None and accelerator.is_main_process:
                                os.makedirs(save_path, exist_ok=True)
                                torch.save(accelerator.unwrap_model(geo_dit).state_dict(),
                                           os.path.join(save_path, "geo_dit.pt"))
                            logger.info(f"Saved state to {save_path}")

                            # Task-57: retain top-K (latest) ∪ periodic (multiples of --checkpoint_keep_interval)
                            # checkpoints via the shared tools/checkpoint_manager; remove the rest. keep_interval
                            # unset/0 -> original latest-K behavior (defaults preserved).
                            _topk = args.checkpoints_total_limit
                            _keep_int = getattr(args, "checkpoint_keep_interval", 0)
                            if (_topk is not None and _topk > 0) or (_keep_int and _keep_int > 0):
                                try:
                                    import sys as _sys
                                    from pathlib import Path as _Path
                                    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3] / "tools"))
                                    from checkpoint_manager import prune_checkpoint_dir
                                    _res = prune_checkpoint_dir(
                                        args.output_dir, topk=_topk, keep_interval=_keep_int,
                                        pattern=r"checkpoint-(\d+)")
                                    if _res["removed"]:
                                        logger.info(f"[Task-57] removed checkpoints: {', '.join(_res['removed'])}")
                                except Exception as _e:
                                    logger.warning(f"[Task-57] checkpoint prune skipped: {_e}")

                if args.validation_prompts is not None and global_step % args.validation_steps == 0:
                    if args.use_ema:
                        # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                        ema_transformer3d.store(transformer3d.parameters())
                        ema_transformer3d.copy_to(transformer3d.parameters())
                    log_validation(
                        vae,
                        text_encoder,
                        tokenizer,
                        clip_image_encoder,
                        transformer3d,
                        args,
                        config,
                        accelerator,
                        weight_dtype,
                        global_step,
                    )
                    if args.use_ema:
                        # Switch back to the original transformer3d parameters.
                        ema_transformer3d.restore(transformer3d.parameters())

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            if _GEO:
                logs["rgb_loss"] = rgb_loss_val
                logs["geo_loss"] = geo_loss_val
                # Task-65: accumulate per-step train losses for the root train_{rgb,geo}_loss.png.
                if _VAL_VIZ_DIR and accelerator.is_main_process:
                    _VALVIZ_HIST["train_rgb"].append((int(global_step), float(rgb_loss_val)))
                    _VALVIZ_HIST["train_geo"].append((int(global_step), float(geo_loss_val)))
                if accelerator.sync_gradients and accelerator.is_main_process:
                    accelerator.log({"rgb_loss": rgb_loss_val, "geo_loss": geo_loss_val,
                                     "total_loss": loss.detach().item()}, step=global_step)
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if args.validation_prompts is not None and epoch % args.validation_epochs == 0:
            if args.use_ema:
                # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                ema_transformer3d.store(transformer3d.parameters())
                ema_transformer3d.copy_to(transformer3d.parameters())
            log_validation(
                vae,
                text_encoder,
                tokenizer,
                clip_image_encoder,
                transformer3d,
                args,
                config,
                accelerator,
                weight_dtype,
                global_step,
            )
            if args.use_ema:
                # Switch back to the original transformer3d parameters.
                ema_transformer3d.restore(transformer3d.parameters())

    # Task-65: write the 4 root loss PNGs (train_{rgb,geo}_loss.png, val_{rgb,geo}_loss.png).
    if _VAL_VIZ_DIR and _valviz is not None and accelerator.is_main_process:
        try:
            _written = _valviz.save_loss_curves(_VAL_VIZ_DIR, _VALVIZ_HIST)
            print(f"[VAL_VIZ] wrote loss curves: {_written} -> {_VAL_VIZ_DIR}")
        except Exception as _e:
            print(f"[VAL_VIZ] loss-curve write skipped: {_e}")

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer3d = unwrap_model(transformer3d)
        if args.use_ema:
            ema_transformer3d.copy_to(transformer3d.parameters())

    if args.use_deepspeed or args.use_fsdp or accelerator.is_main_process:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        accelerator.save_state(save_path)
        logger.info(f"Saved state to {save_path}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
