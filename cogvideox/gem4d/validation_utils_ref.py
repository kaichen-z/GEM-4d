import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import CogVideoXImageToVideoPipeline
from PIL import Image
from tqdm import tqdm

from diffusers.utils import export_to_video, load_image
from gem4d.utils import print_memory, reset_memory, unwrap_model
from gem4d.modules.gem4d_pipeline import GEM4DImageToDepthNormalVideoPipeline_refine
import cv2
import gc
from custom_dataset import crop_and_resize_frames


def _feature_to_rgb_frames(feat, target_hw: Tuple[int, int]) -> np.ndarray:
    """Convert a depth/normal feature tensor into uint8 RGB frames for mp4 writing.

    feat: np.ndarray or torch.Tensor, shape [T, H, W] or [T, H, W, C] or [T, C, H, W].
    target_hw: (H, W) to interpolate to.
    Returns: np.ndarray [T, H, W, 3] in uint8, ready for export_to_video.
    """
    if isinstance(feat, np.ndarray):
        feat = torch.from_numpy(feat)
    feat = feat.detach().float().cpu()
    # Normalize to [T, C, H, W] layout.
    if feat.ndim == 3:                                      # [T, H, W]
        feat = feat.unsqueeze(1)                            # [T, 1, H, W]
    elif feat.ndim == 4 and feat.shape[-1] in (1, 3):       # [T, H, W, C]
        feat = feat.permute(0, 3, 1, 2).contiguous()
    # feat is now [T, C, H, W] (C can be anything, e.g. 256).
    feat = feat.mean(dim=1, keepdim=True)                   # reduce to 1 channel
    T, _, H, W = feat.shape
    if (H, W) != tuple(target_hw):
        feat = F.interpolate(feat, size=tuple(target_hw), mode="bilinear", align_corners=False)
    # Global (per-clip) min-max → [0, 1]. Per-frame min-max makes the output
    # mp4 flicker because a static scene's feature magnitudes naturally vary
    # frame-to-frame; rescaling each frame independently amplifies those into
    # luminance swings. Matches LVP's wan_i2v_da3.py:_feature_to_rgb_video.
    mn = feat.amin()
    mx = feat.amax()
    feat = (feat - mn) / (mx - mn + 1e-8)
    feat = feat.expand(T, 3, -1, -1).permute(0, 2, 3, 1).contiguous()  # [T, H, W, 3]
    return (feat.numpy() * 255.0).astype(np.uint8)


def _load_gt_video_frames(video_path: str, target_n_frames: int, target_hw: Tuple[int, int]) -> np.ndarray:
    """Load a ground-truth video/frame-dir and align it to (target_n_frames, target_hw).

    Supports .mp4/.webm/.avi/.mov via cv2.VideoCapture and directories of PNG frames.
    Returns uint8 np.ndarray of shape [target_n_frames, H, W, 3] in RGB.
    Raises ValueError for still images or unsupported paths.
    """
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    frames = []
    if os.path.isdir(video_path):
        import glob as _glob
        png_paths = sorted(_glob.glob(os.path.join(video_path, "*.png")))
        if len(png_paths) == 0:
            raise ValueError(f"No PNG frames found in directory: {video_path}")
        for fp in png_paths:
            bgr = cv2.imread(fp, cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    else:
        lower = video_path.lower()
        if not (lower.endswith(".mp4") or lower.endswith(".webm")
                or lower.endswith(".avi") or lower.endswith(".mov")):
            raise ValueError(f"Unsupported GT video path (not a dir or video file): {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open GT video file: {video_path}")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames decoded from GT path: {video_path}")

    # Resize each frame to target_hw (cv2 expects (W, H)).
    resized = [
        cv2.resize(f, (target_w, target_h), interpolation=cv2.INTER_AREA)
        for f in frames
    ]
    src = np.stack(resized, axis=0).astype(np.uint8)  # [T_src, H, W, 3]
    t_src = src.shape[0]
    if t_src >= target_n_frames:
        idx = np.linspace(0, t_src - 1, target_n_frames).round().astype(int)
        out = src[idx]
    else:
        # Pad by repeating the last frame.
        pad = np.repeat(src[-1:], target_n_frames - t_src, axis=0)
        out = np.concatenate([src, pad], axis=0)
    return out.astype(np.uint8)


def _concat_side_by_side(gt: np.ndarray, gen: np.ndarray) -> np.ndarray:
    """Concat [gt | gen] along width. Both shape [T, H, W, 3] uint8.

    Layout policy (running_logic.md 15-04): ground truth on the LEFT,
    prediction on the RIGHT. If gt has a different T or H than gen, linearly
    resample gt along T and resize along H (preserving aspect) so the output
    is [T_gen, H_gen, W_gt_resized + W_gen, 3]. W may still differ between
    the two panels.
    """
    if gen.ndim != 4 or gt.ndim != 4:
        raise ValueError(
            f"_concat_side_by_side expects 4D [T,H,W,C] arrays, got gen.ndim={gen.ndim} gt.ndim={gt.ndim}"
        )
    if gen.shape[3] != gt.shape[3]:
        raise ValueError(
            f"_concat_side_by_side C mismatch: gen C={gen.shape[3]} vs gt C={gt.shape[3]}"
        )

    t_gen, h_gen, w_gen, _ = gen.shape
    t_gt, h_gt, w_gt, _ = gt.shape

    # Temporal resample gt -> t_gen (nearest via linspace indices; saturates
    # at the ends when upsampling from a shorter gt).
    if t_gt != t_gen:
        if t_gt >= 1:
            idx = np.linspace(0, max(t_gt - 1, 0), t_gen).round().astype(int)
            gt = gt[idx]
        else:
            raise ValueError(f"_concat_side_by_side: gt has T=0, cannot resample")

    # Spatial resize gt -> h_gen, preserving gt's aspect ratio.
    if gt.shape[1] != h_gen:
        scale = h_gen / float(gt.shape[1])
        new_w = max(1, int(round(gt.shape[2] * scale)))
        gt = np.stack(
            [cv2.resize(f, (new_w, h_gen), interpolation=cv2.INTER_AREA) for f in gt],
            axis=0,
        )

    # Width may differ between gt and gen after resize -- that's fine, concat along W.
    # GT comes from disk as uint8 [0,255]; pred comes from the diffusers pipeline
    # as float32 [0,1] (postprocess_video → denormalize → pt_to_numpy). np.concatenate
    # would upcast uint8 → float32, leaving the GT half in [0,255] while pred is in
    # [0,1]; then export_to_video multiplies by 255 again and wraps uint8, garbling
    # the GT panel. Normalize to a single float32 [0,1] scale before concat.
    if gt.dtype == np.uint8 and np.issubdtype(gen.dtype, np.floating):
        gt = gt.astype(np.float32) / 255.0
    elif gen.dtype == np.uint8 and np.issubdtype(gt.dtype, np.floating):
        gen = gen.astype(np.float32) / 255.0
    return np.concatenate([gt, gen], axis=2)


def _da3_feature_rgb_from_gt(
    da3_block,
    gt_frames: np.ndarray,
    target_hw: Tuple[int, int],
    device,
    downsample_fn: Optional[Callable] = None,
    latent_hw: Optional[Tuple[int, int]] = None,
    encoder_fn: Optional[Callable] = None,
) -> np.ndarray:
    """Run the frozen DA3 teacher on a GT video clip and return an RGB
    visualization of its aggregated feature map, aligned to target_hw.

    gt_frames: uint8 np.ndarray [T, H, W, 3] in RGB (as returned by
        `_load_gt_video_frames`).
    target_hw: (H, W) output size for the visualization frames.
    downsample_fn: optional callable matching
        ``GEM4DDepthNormal_refine3.downsample(g, target_hw)`` (a.k.a.
        ``transformer.downsample``) — if provided along with ``latent_hw``,
        the raw DA3 1024-ch token grid is piped through compress + temporal
        + spatial-interp to the 256-ch *downsampled* DA3 feature.
    encoder_fn: optional callable matching
        ``GEM4DDepthNormal_refine3.encoder(x)`` (a.k.a.
        ``transformer.encoder`` / `self.project`, 256→16) — if provided,
        the 256-ch output of ``downsample_fn`` is further projected to
        the 16-ch latent that the task10 (23-04) depth loss actually
        compares against (``depth_target = encoder(downsample(DA3(gt)))``).
        This is the GT side of `[gt_feat | pred_feat]` for the post-task10
        pipeline (whose pred side is the 16-ch ``forward_depth`` output).
    latent_hw: (H_lat, W_lat) target for ``downsample_fn``. Required when
        ``downsample_fn`` is provided. Typically (H // VAE_spatial,
        W // VAE_spatial) = (H // 8, W // 8) for CogVideoX.

    Returns: uint8 np.ndarray [T, H, W, 3].

    Matches the training-time DA3 call in i2v_depth_ref9.py: inputs are in
    [-1, 1] as [B, F, 3, H, W] float32 on `device`.
    """
    if gt_frames.ndim != 4 or gt_frames.shape[-1] != 3:
        raise ValueError(f"_da3_feature_rgb_from_gt: expected [T,H,W,3] uint8, got {gt_frames.shape}")
    frames = torch.from_numpy(gt_frames).float() / 127.5 - 1.0  # [-1, 1]
    # [T, H, W, 3] -> [1, T, 3, H, W]
    frames = frames.permute(0, 3, 1, 2).unsqueeze(0).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        agg_features = da3_block.pre_forward_tensor(frames)
    # Mirror the training-time reshape in i2v_depth_ref9.py:
    #   [1, B, F, P, C] → permute → reshape → [1, C, F, grid, grid]
    DEFAULT_SIZE = 336
    middle_size = DEFAULT_SIZE // 14
    T = gt_frames.shape[0]
    agg_features = agg_features.permute(1, 2, 3, 4, 0).reshape(
        1, T, middle_size, middle_size, 1024
    ).permute(0, 4, 1, 2, 3)  # [1, 1024, T, grid, grid]
    if downsample_fn is not None and latent_hw is not None:
        # [22-04] Apply the transformer's DA3 downsample (compress +
        # temporal + spatial-interp) so GT sits in the same 256-ch latent
        # space as the pred side. Matches wan_i2v_da3.py:300 (geo_proj).
        with torch.no_grad():
            agg_features = downsample_fn(agg_features, latent_hw)
        # agg_features: [1, 256, F_compressed, H_lat, W_lat]
        if encoder_fn is not None:
            # task10 (23-04): drop into the 16-ch latent the loss actually uses
            # (`depth_target = encoder(downsample(DA3(gt)))`). This is what
            # the post-task10 pipeline's `depth_features` is in too.
            with torch.no_grad():
                agg_features = encoder_fn(agg_features)
            # agg_features: [1, 16, F_compressed, H_lat, W_lat]
    # [1, C, F, H, W] → [F, C, H, W] for _feature_to_rgb_frames.
    agg_features = agg_features[0].permute(1, 0, 2, 3)
    return _feature_to_rgb_frames(agg_features, target_hw)


def read_video_first_frame(video_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"No frames available in video: {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def parse_validation_lists(validation_prompt: str, validation_images: str, sep: str = ":::"):
    """Validation set = a manifest file/dir (see manifest.py), or ':::'-separated prompts/images.
    Returned prompts already include the entry's robot description."""
    from manifest import build_prompt, load_manifest
    p = Path(validation_prompt)
    if p.exists() and (p.is_dir() or p.suffix.lower() in (".json", ".jsonl")):
        entries = load_manifest(p)
        return [build_prompt(e) for e in entries], [e["video"] for e in entries]
    return validation_prompt.split(sep), validation_images.split(sep)

def load_image_data(image_path: str, args: Dict[str, Any], weight_dtype: torch.dtype, accelerator: Accelerator, choice: str = "all") -> torch.Tensor:
    # Load RGB image
    if image_path.endswith(".mp4"):
        rgb_image = read_video_first_frame(image_path)
    else:
        rgb_image = load_image(image_path)  # PIL image
    rgb_image = crop_and_resize_frames([np.array(rgb_image)], (args.height, args.width))[0]
    rgb_image = torch.from_numpy(rgb_image).to(dtype=weight_dtype, device=accelerator.device)
    rgb_image = rgb_image / 255.0  # [0, 1]
    image = rgb_image
    return image.permute(2, 0, 1).unsqueeze(0)  # [B, C, H, W]

def log_validation(
    accelerator: Accelerator,
    pipe: CogVideoXImageToVideoPipeline,
    args: Dict[str, Any],
    pipeline_args: Dict[str, Any],
    is_final_validation: bool = False,
    STEP_COUNT: int = 0,
    number: int = 0,
    validation_image: str = None,
    da3_block=None,):
    if not getattr(args, "enable_model_cpu_offload", False):
        pipe = pipe.to(accelerator.device)   
    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
    videos = []
    depth_feats = []
    normal_feats = []
    for _ in range(args.num_validation_videos):
        output = pipe(**pipeline_args, generator=generator, output_type="np")
        videos.append(output.frames[0])
        # Collect depth/normal feature maps if the pipeline exposes them.
        depth_feat = getattr(output, "depth_features", None)
        normal_feat = getattr(output, "normal_features", None)
        depth_feats.append(depth_feat[0] if depth_feat is not None else None)
        normal_feats.append(normal_feat[0] if normal_feat is not None else None)
    video_filenames = []
    for i, video in enumerate(videos):
        prompt = (pipeline_args["prompt"][:40].replace(" ", "_").replace(" ", "_")
            .replace("'", "_").replace('"', "_").replace("/", "_"))
        filename = os.path.join(args.output_dir, f"video_{STEP_COUNT}_{number}_{prompt}.mp4")
        # Derive target frame size from the RGB video so feature videos line up 1:1.
        T = int(video.shape[0])
        target_hw = (int(video.shape[1]), int(video.shape[2]))
        # Try to load a ground-truth video clip for side-by-side comparison.
        gt = None
        if validation_image is not None:
            try:
                gt = _load_gt_video_frames(validation_image, target_n_frames=T, target_hw=target_hw)
            except Exception as e:
                accelerator.print(
                    f"[log_validation] could not load GT video from {validation_image}: {e}. "
                    f"Falling back to generated-only output."
                )
                gt = None
        # RGB save: [gt | pred] if we have gt, else pred alone.
        if gt is not None:
            try:
                video_sbs = _concat_side_by_side(gt, video)
                export_to_video(video_sbs, filename, fps=8)
            except Exception as e:
                accelerator.print(f"[log_validation] side-by-side concat failed: {e}. Writing generated only.")
                export_to_video(video, filename, fps=8)
        else:
            export_to_video(video, filename, fps=8)
        video_filenames.append(filename)
        # Task-64: also emit the standardized Task-62 layout
        #   <VAL_VIZ_DIR>/Step_NNNNNN/rgb_sample{number}.mp4  (GT-left | pred-right)
        # via the shared tools/val_viz.py helper. Env-gated (default off) so other
        # GEM4D tasks are unaffected; needs a GT clip to form the side-by-side.
        _vvd = os.environ.get("VAL_VIZ_DIR", "")
        if _vvd and gt is not None:
            try:
                import sys as _sys
                _tools = os.path.join(os.environ.get("PROJECT_ROOT",
                    "/workspace/code/16_robot/Roman-4d-1906"), "tools")
                if _tools not in _sys.path:
                    _sys.path.insert(0, _tools)
                import val_viz as _val_viz
                _val_viz.save_rgb_sample(_vvd, STEP_COUNT, number, gt, video)
            except Exception as _e:
                accelerator.print(f"[val_viz] rgb_sample save failed at step {STEP_COUNT} sample {number}: {_e}")
        if depth_feats[i] is not None:
            try:
                depth_rgb = _feature_to_rgb_frames(depth_feats[i], target_hw)
                filename_depth = filename.replace(".mp4", "_depth.mp4")
                # For ref9/da3 we want [gt_feat | pred_feat] of the geometry
                # foundation model, so run DA3 on the GT clip when available.
                gt_depth_rgb = None
                if gt is not None and da3_block is not None:
                    try:
                        # [22-04] If the pipeline's transformer exposes
                        # `.downsample` (ref9 GEM4DDepthNormal_refine3),
                        # pipe GT DA3 tokens through compress+temporal+interp
                        # to the 256-ch latent-grid space, then through
                        # `.encoder` (`self.project`, 256→16) so GT lives in
                        # the **16-ch** space the loss uses post-task10. The
                        # pipeline's pred side is also 16-ch now (raw
                        # `forward_depth` output). CogVideoX VAE spatial
                        # factor is 8 (block_out_channels=4 → 2^3).
                        _transformer = getattr(pipe, "transformer", None)
                        _downsample_fn = getattr(_transformer, "downsample", None)
                        _encoder_fn = getattr(_transformer, "encoder", None)
                        _latent_hw = None
                        if _downsample_fn is not None:
                            _latent_hw = (int(args.height) // 8, int(args.width) // 8)
                        gt_depth_rgb = _da3_feature_rgb_from_gt(
                            da3_block, gt, target_hw, accelerator.device,
                            downsample_fn=_downsample_fn,
                            latent_hw=_latent_hw,
                            encoder_fn=_encoder_fn,
                        )
                    except Exception as e_gt:
                        accelerator.print(f"[log_validation] DA3 on GT failed: {e_gt}. Falling back to pred-only depth mp4.")
                        gt_depth_rgb = None
                if gt_depth_rgb is not None:
                    try:
                        depth_sbs = _concat_side_by_side(gt_depth_rgb, depth_rgb)
                        export_to_video(depth_sbs, filename_depth, fps=8)
                    except Exception as e2:
                        accelerator.print(f"[log_validation] depth feat side-by-side concat failed: {e2}. Writing pred depth only.")
                        export_to_video(depth_rgb, filename_depth, fps=8)
                else:
                    export_to_video(depth_rgb, filename_depth, fps=8)
            except Exception as e:
                accelerator.print(f"[log_validation] failed to write depth feature video: {e}")
        if normal_feats[i] is not None:
            try:
                normal_rgb = _feature_to_rgb_frames(normal_feats[i], target_hw)
                filename_normal = filename.replace(".mp4", "_normal.mp4")
                # No DA3 teacher for normals: write pred-only normal mp4.
                export_to_video(normal_rgb, filename_normal, fps=8)
            except Exception as e:
                accelerator.print(f"[log_validation] failed to write normal feature video: {e}")
    return videos

def run_validation(
    args: Dict[str, Any],
    accelerator: Accelerator,
    transformer,
    scheduler,
    model_config: Dict[str, Any],
    weight_dtype: torch.dtype,
    STEP_COUNT: int = 0,
    choice: str = "all",
    da3_block=None,
) -> None:
    accelerator.print("===== Memory before validation =====")
    print_memory(accelerator.device)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(accelerator.device)
    pipe = GEM4DImageToDepthNormalVideoPipeline_refine.from_pretrained(
        os.environ.get("COGVIDEO_BASE_PATH", "THUDM/CogVideoX-5b-I2V"),
        transformer=unwrap_model(accelerator, transformer),
        scheduler=scheduler,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,)
    if args.enable_slicing:
        pipe.vae.enable_slicing()
    if args.enable_tiling:
        pipe.vae.enable_tiling()
    if args.enable_model_cpu_offload:
        pipe.enable_model_cpu_offload(device=accelerator.device)    
    # Process validation images and prompts
    # validation_prompts = args.validation_prompt.split(args.validation_prompt_separator)
    # validation_images = args.validation_images.split(args.validation_prompt_separator)
    validation_prompts, validation_images = parse_validation_lists(args.validation_prompt, args.validation_prompt)
    for number, validation_image, validation_prompt in tqdm(zip(range(len(validation_images)), validation_images, validation_prompts), total=len(validation_images)):
        # Load and process image data
        try:
            image = load_image_data(validation_image, args, weight_dtype, accelerator, choice)
        except Exception as e:
            accelerator.print(f"Error loading image data: {e}")
            continue

        prompt = validation_prompt

        pipeline_args = {
            "image": image,
            "prompt": prompt,
            "guidance_scale": args.guidance_scale,
            "use_dynamic_cfg": args.use_dynamic_cfg,
            "height": args.height,
            "width": args.width,
            "max_sequence_length": model_config.max_text_seq_length,
            "num_frames": args.max_num_frames,}
        # Run validation
        log_validation(
            pipe=pipe, args=args,
            accelerator=accelerator,
            pipeline_args=pipeline_args,
            STEP_COUNT=STEP_COUNT,
            number=number,
            validation_image=validation_image,
            da3_block=da3_block)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(accelerator.device)
    accelerator.print("===== Memory after validation =====")
    print_memory(accelerator.device)
    reset_memory(accelerator.device)
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(accelerator.device)