"""GEM-4D inference: first frame + instruction -> video rollout. The Geometry DiT is not needed.

    python inference/infer.py --backbone cogvideox --base_model weights/cogvideox \
        --weights weights/checkpoint-2 --manifest examples/val_manifest.json --output_dir outputs/demo

--weights: a released checkpoint, a training export (runs/x/checkpoints/step_NNNNNN), or omitted
           (= the base model). Writes <idx>_<clip>.mp4 and, with --side_by_side, <idx>_<clip>_gt_pred.mp4.
"""
import argparse
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "model"), str(ROOT)]

from gem4d import build_backbone                                    # noqa: E402
from gem4d.utils.video_io import first_frame, save_mp4, side_by_side, to_uint8  # noqa: E402
from training.data.manifest import build_prompt, load_manifest      # noqa: E402
from training.data.video_dataset import center_crop_resize, read_video  # noqa: E402

DEFAULTS = {"cogvideox": dict(guidance_scale=6.0, fps=8), "wan2.2-ti2v-5b": dict(guidance_scale=5.0, fps=16)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", required=True, choices=list(DEFAULTS))
    ap.add_argument("--base_model", required=True, help="full diffusers model dir of the backbone")
    ap.add_argument("--weights", default=None, help="fine-tuned transformer (checkpoint dir); default: base")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--num_frames", type=int, default=49)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance_scale", type=float, default=None)
    ap.add_argument("--fps", type=int, default=None)
    ap.add_argument("--num_samples", type=int, default=1, help="videos per manifest entry")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--side_by_side", action="store_true", help="also write [GT | prediction] (video inputs)")
    ap.add_argument("--cpu_offload", action="store_true", help="model CPU offload for <40 GB GPUs")
    args = ap.parse_args()
    d = DEFAULTS[args.backbone]
    gs = args.guidance_scale if args.guidance_scale is not None else d["guidance_scale"]
    fps = args.fps or d["fps"]

    weights = args.weights
    if weights and os.path.isdir(os.path.join(weights, "transformer")):
        weights = os.path.join(weights, "transformer")
    bb = build_backbone(args.backbone, base_model=args.base_model, init_weights=weights)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bb.to(dev, torch.bfloat16)
    pipe = bb.pipeline()
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(dev)

    size = (args.height, args.width)
    for i, entry in enumerate(load_manifest(args.manifest, args.data_root)):
        prompt, stem = build_prompt(entry), f"{i:04d}_{Path(entry['video']).stem}"
        image = first_frame(entry["video"], size)
        for k in range(args.num_samples):
            frames = pipe(image=image, prompt=prompt, height=args.height, width=args.width,
                          num_frames=args.num_frames, num_inference_steps=args.steps, guidance_scale=gs,
                          generator=torch.Generator(device=dev).manual_seed(args.seed + k),
                          output_type="np").frames[0]
            name = stem if args.num_samples == 1 else f"{stem}_{k}"
            out = os.path.join(args.output_dir, name + ".mp4")
            save_mp4(frames, out, fps=fps)
            if args.side_by_side and Path(entry["video"]).suffix.lower() in (".mp4", ".webm", ".avi", ".mov"):
                gt = [center_crop_resize([f], size)[0] for f in read_video(Path(entry["video"]))]
                save_mp4(side_by_side(gt, to_uint8(frames)), out.replace(".mp4", "_gt_pred.mp4"), fps=fps)
            print(f"[{i}] {prompt!r} -> {out}", flush=True)


if __name__ == "__main__":
    main()
