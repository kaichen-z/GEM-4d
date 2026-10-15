"""GEM-4D (CogVideoX-5b-I2V) inference: first frame + prompt -> 49-frame RGB video.

    PYTHONPATH=<code-git>/cogvideox:<code-git>/depth/src python inference.py \
        --manifest my_test.json \
        --weights_path weights/checkpoint-2 \
        --base_model weights/cogvideox \
        --output_dir outputs/ckpt2

--manifest      manifest file/dir (format: gem4d/manifest.py). `video` may be an .mp4
                (its first frame is used) or an image file.
--weights_path  a fine-tuned transformer dir: either it contains config.json + *.safetensors
                directly, or it has a `transformer/` subfolder (SIMPLE_CKPT step_NNNNNN dirs,
                OUTPUT_DIR/transformer). The Geo-DiT keys inside are ignored.
--base_model    full CogVideoX-5b-I2V pipeline (tokenizer, T5, VAE, scheduler).

Writes <output_dir>/<idx>_<video-stem>.mp4; with --side_by_side also the GT clip | prediction.
"""
import argparse
import gc
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import CogVideoXDPMScheduler
from diffusers.utils import export_to_video, load_image

from gem4d.manifest import build_prompt, load_manifest
from gem4d.modules.gem4d_model import GEM4DDepthNormal_refine3
from gem4d.modules.gem4d_pipeline import GEM4DImageToDepthNormalVideoPipeline_refine
from gem4d.utils import crop_and_resize_frames

torch.set_grad_enabled(False)


def read_frames(path: str, max_frames=None) -> list:
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok or (max_frames and len(frames) >= max_frames):
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"No frame decoded: {path}")
    return frames


def first_frame(path: str, h: int, w: int) -> np.ndarray:
    img = read_frames(path, 1)[0] if Path(path).suffix.lower() in (".mp4", ".webm", ".avi") \
        else np.array(load_image(path).convert("RGB"))
    return crop_and_resize_frames([img.astype(np.float32)], (h, w))[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--weights_path", required=True)
    p.add_argument("--base_model", default="THUDM/CogVideoX-5b-I2V")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--data_root", default=None, help="resolve relative manifest paths against this dir")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--num_frames", type=int, default=49)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=6.0)
    p.add_argument("--use_dynamic_cfg", action="store_true")
    p.add_argument("--num_videos_per_sample", type=int, default=1)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--mixed_precision", default="bf16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--memory_efficient", action="store_true", help="VAE slicing/tiling + CPU offload")
    p.add_argument("--side_by_side", action="store_true", help="also write GT | prediction (needs video input)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.mixed_precision]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wp = Path(args.weights_path)
    if not (wp / "config.json").exists() and (wp / "transformer" / "config.json").exists():
        wp = wp / "transformer"
    pipe = GEM4DImageToDepthNormalVideoPipeline_refine.from_pretrained(args.base_model, torch_dtype=dtype)
    t = GEM4DDepthNormal_refine3.from_pretrained_modify(str(wp)).to(dtype=dtype).eval()
    t.config.in_channels += 16 * 4
    if hasattr(t.patch_embed, "pos_embedding"):
        del t.patch_embed.pos_embedding
    t.patch_embed.use_learned_positional_embeddings = False
    pipe.transformer = t
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config)
    if args.memory_efficient:
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)

    h, w = args.height, args.width
    for idx, entry in enumerate(load_manifest(args.manifest, args.data_root)):
        prompt = build_prompt(entry)
        img = torch.from_numpy(first_frame(entry["video"], h, w)).to(device=device, dtype=dtype) / 255.0
        img = img.permute(2, 0, 1).unsqueeze(0)
        for n in range(args.num_videos_per_sample):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            g = torch.Generator(device=device).manual_seed(args.seed + n)
            frames = pipe(image=img, prompt=prompt, guidance_scale=args.guidance_scale,
                          use_dynamic_cfg=args.use_dynamic_cfg, height=h, width=w,
                          num_inference_steps=args.num_inference_steps, num_frames=args.num_frames,
                          generator=g).frames[0]
            stem = f"{idx:04d}_{Path(entry['video']).stem}" + (f"_{n}" if args.num_videos_per_sample > 1 else "")
            out = os.path.join(args.output_dir, stem + ".mp4")
            export_to_video(frames, out, fps=args.fps)
            if args.side_by_side and Path(entry["video"]).suffix.lower() in (".mp4", ".webm", ".avi"):
                gt = crop_and_resize_frames(read_frames(entry["video"]), (h, w))
                idxs = np.linspace(0, len(gt) - 1, len(frames), dtype=int) if len(gt) > len(frames) else \
                    [min(i, len(gt) - 1) for i in range(len(frames))]
                sbs = [np.concatenate([gt[i].astype(np.uint8), np.array(f)], axis=1) for i, f in zip(idxs, frames)]
                export_to_video([s / 255.0 for s in sbs], out.replace(".mp4", "_gt_pred.mp4"), fps=args.fps)
            print(f"[{idx}] {prompt!r} -> {out}")


if __name__ == "__main__":
    main()
