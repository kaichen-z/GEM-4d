import gc
import cv2
import numpy as np
import os
import re
import shutil
import inspect
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.utils.torch_utils import is_compiled_module
from accelerate import Accelerator, DistributedType, init_empty_weights
from gem4d.modules.gem4d_model import GEM4DDepthNormal as CogVideoXTransformer3DModel
from gem4d.checkpoint_retention import retained_checkpoints
from diffusers.training_utils import cast_training_params

logger = get_logger(__name__)

# Matches `checkpoint-{integer}` dirs written by save_checkpoint_{acc,model}.
# Legacy `checkpoint-permanent-{N}` dirs on disk are intentionally NOT matched —
# the single-prefix policy (plan 17-04) keeps retention rule simple; users who
# need an old permanent checkpoint can pass its path explicitly to
# --resume_from_checkpoint.
_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")

class CHECKPOINT_MANAGER:
    def __init__(self, args):
        self.args = args
    
    def get_initial_global_step_acc(self, accelerator):
        initial_global_step, first_epoch = 0, 0
        if not self.args.resume_from_checkpoint:
            initial_global_step = 0
        else:
            if self.args.resume_from_checkpoint != "latest":
                path = os.path.basename(self.args.resume_from_checkpoint)
                dirs = []
            else:
                dirs = [d for d in os.listdir(self.args.output_dir) if _CKPT_RE.match(d)]
                dirs = sorted(dirs, key=lambda x: int(_CKPT_RE.match(x).group(1)))
                path = dirs[-1] if len(dirs) > 0 else None
            if path is None:
                accelerator.print(f"Checkpoint '{self.args.resume_from_checkpoint}' does not exist. Starting a new training run.")
                self.args.resume_from_checkpoint = None
                initial_global_step = 0
            else:
                try:
                    accelerator.print(f"Resuming from checkpoint {path}")
                    accelerator.load_state(os.path.join(self.args.output_dir, path))
                    initial_global_step = int(_CKPT_RE.match(path).group(1))
                except:
                    shutil.rmtree(os.path.join(self.args.output_dir, dirs[-1]), ignore_errors=True)
                    accelerator.print(f"Resuming from checkpoint {dirs[-2]}")
                    accelerator.load_state(os.path.join(self.args.output_dir, dirs[-2]))
                    initial_global_step = int(_CKPT_RE.match(dirs[-2]).group(1))
        return initial_global_step

    def get_initial_global_step_model(self, accelerator, transformer):
        """Resume from checkpoint saved by save_checkpoint_model (transformer-only)."""
        initial_global_step, first_epoch = 0, 0
        if not self.args.resume_from_checkpoint:
            initial_global_step = 0
        else:
            if self.args.resume_from_checkpoint != "latest":
                path = os.path.basename(self.args.resume_from_checkpoint)
                dirs = []
            else:
                dirs = [d for d in os.listdir(self.args.output_dir) if _CKPT_RE.match(d)]
                dirs = sorted(dirs, key=lambda x: int(_CKPT_RE.match(x).group(1)))
                path = dirs[-1] if len(dirs) > 0 else None
            if path is None:
                accelerator.print(f"Checkpoint '{self.args.resume_from_checkpoint}' does not exist. Starting a new training run.")
                self.args.resume_from_checkpoint = None
                initial_global_step = 0
            else:
                try:
                    load_dtype = torch.bfloat16  # CogVideoX-5b weights are bf16 (old heuristic keyed on "5b" in the path)
                    ckpt_path = os.path.join(self.args.output_dir, path)
                    accelerator.print(f"Resuming from checkpoint {path} (loading transformer only)")
                    transformer = transformer.from_pretrained_modify(
                            ckpt_path, subfolder=None,
                            torch_dtype=load_dtype, revision=self.args.revision, variant=self.args.variant,)
                    initial_global_step = int(_CKPT_RE.match(path).group(1))
                    transformer.patch_embed.use_learned_positional_embeddings = False
                except Exception:
                    shutil.rmtree(os.path.join(self.args.output_dir, dirs[-1]), ignore_errors=True)
                    load_dtype = torch.bfloat16  # CogVideoX-5b weights are bf16 (old heuristic keyed on "5b" in the path)
                    ckpt_path = os.path.join(self.args.output_dir, dirs[-2])
                    accelerator.print(f"Resuming from checkpoint {dirs[-2]} (loading transformer only)")
                    transformer = transformer.from_pretrained_modify(
                            ckpt_path, subfolder=None,
                            torch_dtype=load_dtype, revision=self.args.revision, variant=self.args.variant,)
                    initial_global_step = int(_CKPT_RE.match(dirs[-2]).group(1))
                    transformer.patch_embed.use_learned_positional_embeddings = False
        return initial_global_step, transformer

    # def save_checkpoint_acc(self, accelerator, global_step, logger):
    #     if global_step % self.args.checkpointing_steps == 0:
    #         # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
    #         if self.args.checkpoints_total_limit is not None:
    #             checkpoints = os.listdir(self.args.output_dir)
    #             checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
    #             checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
    #             # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
    #             if len(checkpoints) >= self.args.checkpoints_total_limit:
    #                 num_to_remove = len(checkpoints) - self.args.checkpoints_total_limit + 1
    #                 removing_checkpoints = checkpoints[0:num_to_remove]
    #                 logger.info(f"Removing checkpoints: {', '.join(removing_checkpoints)}")
    #                 for removing_checkpoint in removing_checkpoints:
    #                     shutil.rmtree(os.path.join(self.args.output_dir, removing_checkpoint))
    #         save_path = os.path.join(self.args.output_dir, f"checkpoint-{global_step}")
    #         accelerator.save_state(save_path)
    #         logger.info(f"Saved state to {save_path}")

    def save_checkpoint_acc(self, accelerator, global_step, logger):
        if global_step % self.args.checkpointing_steps != 0:
            return
        self._prune_checkpoints(global_step, logger)
        save_path = os.path.join(self.args.output_dir, f"checkpoint-{global_step}")
        accelerator.save_state(save_path)
        logger.info(f"Saved state to {save_path}")

    def _prune_checkpoints(self, global_step, logger):
        """Apply top-K ∪ multiples-of-M retention to existing checkpoint-{N} dirs.

        Legacy `checkpoint-permanent-*` dirs (if any remain on disk from
        before the 17-04 refactor) are ignored by _CKPT_RE and thus neither
        pruned nor counted — they are treated as manual archives.
        """
        if self.args.checkpoints_total_limit is None:
            return
        K = self.args.checkpoints_total_limit
        M = getattr(self.args, "milestone_checkpointing_steps", 0)
        existing = []
        for name in os.listdir(self.args.output_dir):
            m = _CKPT_RE.match(name)
            if m:
                existing.append((int(m.group(1)), name))
        # Include the about-to-be-saved step so retention matches the post-save state.
        hypothetical_steps = [s for s, _ in existing] + [global_step]
        keep = retained_checkpoints(hypothetical_steps, K=K, M=M)
        removing = [name for step, name in existing if step not in keep]
        if removing:
            logger.info(f"Removing checkpoints: {', '.join(removing)}")
            for name in removing:
                shutil.rmtree(os.path.join(self.args.output_dir, name))

    def save_checkpoint_model(self, accelerator, transformer, global_step, logger):
        if global_step % self.args.checkpointing_steps != 0:
            return
        self._prune_checkpoints(global_step, logger)
        save_path = os.path.join(self.args.output_dir, f"checkpoint-{global_step}")
        transformer = unwrap_model(accelerator, transformer)
        dtype = (torch.float16 if self.args.mixed_precision == "fp16"
            else torch.bfloat16 if self.args.mixed_precision == "bf16" else torch.float32)
        transformer = transformer.to(dtype)
        transformer.save_pretrained(
            save_path, safe_serialization=True, max_shard_size="5GB",)
        logger.info(f"Saved state to {save_path}")

    def save_model(self, accelerator, transformer):
        transformer = unwrap_model(accelerator, transformer)
        dtype = (torch.float16 if self.args.mixed_precision == "fp16"
            else torch.bfloat16 if self.args.mixed_precision == "bf16" else torch.float32)
        transformer = transformer.to(dtype)
        transformer.save_pretrained(
            os.path.join(self.args.output_dir, "transformer"),
            safe_serialization=True, max_shard_size="5GB",)

def loss_fn(pred, target, weight):
    """SNR-weighted MSE loss for flow-matching diffusion training.

    Args:
        pred:   Model prediction (velocity), shape [B, F, C, H, W].
        target: Ground-truth latent, shape [B, F, C, H, W].
        weight: SNR weight = 1/(1 - alpha_cumprod[t]), shape broadcastable to pred.

    Returns:
        Scalar: mean over batch of per-sample weighted MSE.

    Formula: mean_B( mean_elements( weight * (pred - target)^2 ) )
    """
    B = pred.shape[0]
    return torch.mean((weight * (pred - target) ** 2).reshape(B, -1), dim=1).mean()

def make_save_model_hook(accelerator, transformer):
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                if isinstance(unwrap_model(accelerator, model), type(unwrap_model(accelerator, transformer))):
                    model = unwrap_model(accelerator, model)
                    model.save_pretrained(os.path.join(output_dir, "transformer"), safe_serialization=True, max_shard_size="5GB")
                else:
                    raise ValueError(f"Unexpected save model: {model.__class__}")
                if weights:
                    weights.pop()
    return save_model_hook

def make_load_model_hook(accelerator, transformer, args):
    def load_model_hook(models, input_dir):
        transformer_ = None
        init_under_meta = False
        if not accelerator.distributed_type == DistributedType.DEEPSPEED:
            while len(models) > 0:
                model = models.pop()
                if isinstance(unwrap_model(accelerator, model), type(unwrap_model(accelerator, transformer))):
                    transformer_ = unwrap_model(accelerator, model)
                else:
                    raise ValueError(f"Unexpected save model: {unwrap_model(accelerator, model).__class__}")
        else:
            with init_empty_weights():
                transformer_ = CogVideoXTransformer3DModel.from_config(
                    args.pretrained_model_name_or_path, subfolder="transformer")
                init_under_meta = True
        load_model = CogVideoXTransformer3DModel.from_pretrained_modify(
            os.path.join(input_dir, "transformer"), map_dim=False, add_in_channels=0, add_out_channels=0)
        if "I2V" in args.pretrained_model_name_or_path:
            del load_model.patch_embed.pos_embedding
            load_model.patch_embed.use_learned_positional_embeddings = False
            load_model.config.use_learned_positional_embeddings = False
            # transformer_ (e.g. from_config in DeepSpeed) may still have pos_embedding; remove so state_dict matches
            if hasattr(transformer_.patch_embed, "pos_embedding"):
                del transformer_.patch_embed.pos_embedding
            transformer_.patch_embed.use_learned_positional_embeddings = False
            transformer_.config.use_learned_positional_embeddings = False
        transformer_.register_to_config(**load_model.config)
        # in_channels already correct in loaded config when resuming from checkpoint
        transformer_.load_state_dict(load_model.state_dict(), assign=init_under_meta)
        del load_model
        if args.mixed_precision == "fp16":
            cast_training_params([transformer_])
    return load_model_hook

def get_optimizer(
    params_to_optimize,
    optimizer_name: str = "adam",
    learning_rate: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.95,
    beta3: float = 0.98,
    epsilon: float = 1e-8,
    weight_decay: float = 1e-4,
    prodigy_decouple: bool = False,
    prodigy_use_bias_correction: bool = False,
    prodigy_safeguard_warmup: bool = False,
    use_8bit: bool = False,
    use_4bit: bool = False,
    use_torchao: bool = False,
    use_deepspeed: bool = False,
    use_cpu_offload_optimizer: bool = False,
    offload_gradients: bool = False,
) -> torch.optim.Optimizer:
    optimizer_name = optimizer_name.lower()

    # Use DeepSpeed optimzer
    if use_deepspeed:
        from accelerate.utils import DummyOptim

        return DummyOptim(
            params_to_optimize,
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=epsilon,
            weight_decay=weight_decay,
        )

    if use_8bit and use_4bit:
        raise ValueError("Cannot set both `use_8bit` and `use_4bit` to True.")

    if (use_torchao and (use_8bit or use_4bit)) or use_cpu_offload_optimizer:
        try:
            import torchao

            torchao.__version__
        except ImportError:
            raise ImportError(
                "To use optimizers from torchao, please install the torchao library: `USE_CPP=0 pip install torchao`."
            )

    if not use_torchao and use_4bit:
        raise ValueError("4-bit Optimizers are only supported with torchao.")

    # Optimizer creation
    supported_optimizers = ["adam", "adamw", "prodigy", "came"]
    if optimizer_name not in supported_optimizers:
        logger.warning(
            f"Unsupported choice of optimizer: {optimizer_name}. Supported optimizers include {supported_optimizers}. Defaulting to `AdamW`."
        )
        optimizer_name = "adamw"

    if (use_8bit or use_4bit) and optimizer_name not in ["adam", "adamw"]:
        raise ValueError("`use_8bit` and `use_4bit` can only be used with the Adam and AdamW optimizers.")

    if use_8bit:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`.")

    if optimizer_name == "adamw":
        if use_torchao:
            from torchao.prototype.low_bit_optim import AdamW4bit, AdamW8bit

            optimizer_class = AdamW8bit if use_8bit else AdamW4bit if use_4bit else torch.optim.AdamW
        else:
            optimizer_class = bnb.optim.AdamW8bit if use_8bit else torch.optim.AdamW

        init_kwargs = {
            "betas": (beta1, beta2),
            "eps": epsilon,
            "weight_decay": weight_decay,
        }

    elif optimizer_name == "adam":
        if use_torchao:
            from torchao.prototype.low_bit_optim import Adam4bit, Adam8bit

            optimizer_class = Adam8bit if use_8bit else Adam4bit if use_4bit else torch.optim.Adam
        else:
            optimizer_class = bnb.optim.Adam8bit if use_8bit else torch.optim.Adam

        init_kwargs = {
            "betas": (beta1, beta2),
            "eps": epsilon,
            "weight_decay": weight_decay,
        }

    elif optimizer_name == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        init_kwargs = {
            "lr": learning_rate,
            "betas": (beta1, beta2),
            "beta3": beta3,
            "eps": epsilon,
            "weight_decay": weight_decay,
            "decouple": prodigy_decouple,
            "use_bias_correction": prodigy_use_bias_correction,
            "safeguard_warmup": prodigy_safeguard_warmup,
        }

    elif optimizer_name == "came":
        try:
            import came_pytorch
        except ImportError:
            raise ImportError("To use CAME, please install the came-pytorch library: `pip install came-pytorch`")

        optimizer_class = came_pytorch.CAME

        init_kwargs = {
            "lr": learning_rate,
            "eps": (1e-30, 1e-16),
            "betas": (beta1, beta2, beta3),
            "weight_decay": weight_decay,
        }

    if use_cpu_offload_optimizer:
        try:
            from torchao.prototype.low_bit_optim import CPUOffloadOptimizer
        except ImportError:  # torchao >= 0.10 moved this to torchao.optim
            from torchao.optim import CPUOffloadOptimizer

        if "fused" in inspect.signature(optimizer_class.__init__).parameters:
            init_kwargs.update({"fused": True})

        optimizer = CPUOffloadOptimizer(
            params_to_optimize, optimizer_class=optimizer_class, offload_gradients=offload_gradients, **init_kwargs
        )
    else:
        optimizer = optimizer_class(params_to_optimize, **init_kwargs)

    return optimizer


def get_gradient_norm(parameters):
    norm = 0
    for param in parameters:
        if param.grad is None:
            continue
        local_norm = param.grad.detach().data.norm(2)
        norm += local_norm.item() ** 2
    norm = norm**0.5
    return norm


# Similar to diffusers.pipelines.hunyuandit.pipeline_hunyuandit.get_resize_crop_region_for_grid
def get_resize_crop_region_for_grid(src, tgt_width, tgt_height):
    tw = tgt_width
    th = tgt_height
    h, w = src
    r = h / w
    if r > (th / tw):
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)


def prepare_rotary_positional_embeddings(
    height: int,
    width: int,
    num_frames: int,
    vae_scale_factor_spatial: int = 8,
    patch_size: int = 2,
    patch_size_t: int = None,
    attention_head_dim: int = 64,
    device: Optional[torch.device] = None,
    base_height: int = 480,
    base_width: int = 720,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grid_height = height // (vae_scale_factor_spatial * patch_size)
    grid_width = width // (vae_scale_factor_spatial * patch_size)
    base_size_width = base_width // (vae_scale_factor_spatial * patch_size)
    base_size_height = base_height // (vae_scale_factor_spatial * patch_size)

    if patch_size_t is None:
        # CogVideoX 1.0
        grid_crops_coords = get_resize_crop_region_for_grid(
            (grid_height, grid_width), base_size_width, base_size_height
        )
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=attention_head_dim,
            crops_coords=grid_crops_coords,
            grid_size=(grid_height, grid_width),
            temporal_size=num_frames,
        )
    else:
        # CogVideoX 1.5
        base_num_frames = (num_frames + patch_size_t - 1) // patch_size_t

        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=attention_head_dim,
            crops_coords=None,
            grid_size=(grid_height, grid_width),
            temporal_size=base_num_frames,
            grid_type="slice",
            max_size=(base_size_height, base_size_width),
        )

    freqs_cos = freqs_cos.to(device=device)
    freqs_sin = freqs_sin.to(device=device)
    return freqs_cos, freqs_sin


def reset_memory(device: Union[str, torch.device]) -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.reset_accumulated_memory_stats(device)


def print_memory(device: Union[str, torch.device]) -> None:
    memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
    max_memory_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
    max_memory_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
    print(f"{memory_allocated=:.3f} GB")
    print(f"{max_memory_allocated=:.3f} GB")
    print(f"{max_memory_reserved=:.3f} GB")


def unwrap_model(accelerator: Accelerator, model):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def modify_transformer(model, add_in_channels, add_out_channels=0):
    device = model.device
    dtype = model.dtype

    model.config.in_channels += add_in_channels
    old_proj = model.patch_embed.proj
    old_in_channels = old_proj.in_channels
    old_out_channels = old_proj.out_channels

    if type(old_proj) == nn.Conv2d:
        new_proj = nn.Conv2d(
            old_in_channels + add_in_channels,
            old_out_channels,
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
            padding=old_proj.padding,
            bias=old_proj.bias is not None,
        ).to(device, dtype)
    elif type(old_proj) == nn.Linear:
        new_proj = nn.Linear(old_in_channels + add_in_channels, old_out_channels).to(device, dtype)

    with torch.no_grad():
        new_proj.weight.zero_()
        new_proj.weight[:, : old_proj.weight.shape[1]].copy_(old_proj.weight)
        if old_proj.bias is not None:
            new_proj.bias.copy_(old_proj.bias)
    model.patch_embed.proj = new_proj

    if add_out_channels > 0:
        old_proj = model.proj_out
        patch_size = model.config.patch_size
        out_channels = model.config.out_channels
        new_proj = nn.Linear(
            old_proj.in_features, old_proj.out_features + add_out_channels * patch_size * patch_size
        ).to(device, dtype)
        with torch.no_grad():
            new_proj.weight.zero_()
            new_proj.weight[: old_proj.weight.shape[0]].copy_(old_proj.weight)
            if old_proj.bias is not None:
                new_proj.bias[: old_proj.bias.shape[0]].copy_(old_proj.bias)
        model.proj_out = new_proj


def crop_and_resize_frames(frames, target_size, interpolation="bilinear"):
    target_height, target_width = target_size
    original_height, original_width = frames[0].shape[:2]
    if original_height == target_height and original_width == target_width:
        return [frame for frame in frames]

    # ==== interpolation method ====
    if interpolation == "bilinear":
        interpolation = cv2.INTER_LINEAR
    elif interpolation == "nearest":
        interpolation = cv2.INTER_NEAREST
    else:
        interpolation = cv2.INTER_LINEAR
        logger.warning(f"Unsupported interpolation: {interpolation}. Using bilinear instead.")

    processed_frames = []
    for frame in frames:
        original_height, original_width = frame.shape[:2]
        aspect_ratio_target = target_width / target_height
        aspect_ratio_original = original_width / original_height

        if aspect_ratio_original > aspect_ratio_target:
            new_width = int(aspect_ratio_target * original_height)
            start_x = (original_width - new_width) // 2
            cropped_frame = frame[:, start_x : start_x + new_width]
        else:
            new_height = int(original_width / aspect_ratio_target)
            start_y = (original_height - new_height) // 2
            cropped_frame = frame[start_y : start_y + new_height, :]
        resized_frame = cv2.resize(cropped_frame, (target_width, target_height), interpolation=interpolation)
        processed_frames.append(resized_frame)

    return processed_frames

def count_parameters(model):
    total = 0
    trainable = 0

    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()

    return total, trainable