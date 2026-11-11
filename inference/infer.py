"""GEM-4D inference: first frame + instruction -> video rollout. The Geometry DiT is not needed.

    # one image + prompt
    python inference/infer.py --backbone wan2.1-i2v-14b --base_model weights/Wan2.1-I2V-14B-480P \
        --weights weights/gem-4d-wan14b --image examples/first_frame.png --prompt "open the drawer" \
        --output_dir outputs/demo
    # a manifest of clips / images
    python inference/infer.py --backbone wan2.2-ti2v-5b --base_model weights/Wan2.2-TI2V-5B-Diffusers \
        --weights runs/wan5b_gem4d/checkpoints/step_010000 --manifest examples/val_manifest.json --output_dir outputs/demo

--weights: a training export (runs/x/checkpoints/step_NNNNNN), the GEM-4D-Wan14B checkpoint, or
           omitted (= the base model). Writes <idx>_<clip>.mp4 and, with --side_by_side, <idx>_<clip>_gt_pred.mp4.
--compose: a plain photo / single-view video -> the robot composite (main view left, x < 302 of 416).
--process4d: also run 4D-process on every generated video (DA3 + RAFT-small scene flow, see
           inference/process4d.py): <idx>_<clip>_4d.{html,mp4,json,npz}.
Unset sampling options take the backbone's defaults (DEFAULTS below).
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
from inference.process4d import add_args as add_4d_args, parse_crop  # noqa: E402

DEFAULTS = {
    "wan2.2-ti2v-5b": dict(steps=50, guidance_scale=5.0, height=480, width=640, fps=16, seed=42),
    # GEM-4D-Wan14B sampler: 30 steps, shift 3, CFG 4, empty negative prompt, 49 x 240 x 416
    "wan2.1-i2v-14b": dict(steps=30, guidance_scale=4.0, height=240, width=416, fps=8, seed=3402),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", required=True, choices=list(DEFAULTS))
    ap.add_argument("--base_model", required=True, help="full model dir of the backbone")
    ap.add_argument("--weights", default=None, help="fine-tuned transformer (checkpoint dir); default: base")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", help="clips / images with prompts (training/data/manifest.py)")
    src.add_argument("--image", help="one condition image (or video: its first frame); needs --prompt")
    ap.add_argument("--prompt", default=None, help="instruction for --image, used verbatim")
    ap.add_argument("--compose", action="store_true",
                    help="put a single-view image into the 416x240 robot composite GEM-4D-Wan14B was trained on")
    ap.add_argument("--negative_prompt", default="", help="wan2.1-i2v-14b only")
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--num_frames", type=int, default=49)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--guidance_scale", type=float, default=None)
    ap.add_argument("--shift", type=float, default=None, help="wan2.1-i2v-14b sampling shift (default 3)")
    ap.add_argument("--fps", type=int, default=None)
    ap.add_argument("--num_samples", type=int, default=1, help="videos per entry")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--side_by_side", action="store_true", help="also write [GT | prediction] (video inputs)")
    ap.add_argument("--cpu_offload", action="store_true", help="model CPU offload (wan2.2-ti2v-5b only)")
    ap.add_argument("--process4d", action="store_true", help="run 4D-process (scene flow + 3D viewer) on each video")
    add_4d_args(ap, prefix="process4d_")
    args = ap.parse_args()
    if args.image and not args.prompt:
        ap.error("--image needs --prompt")
    d = DEFAULTS[args.backbone]
    opt = {k: getattr(args, k) if getattr(args, k) is not None else v for k, v in d.items()}
    size = (opt["height"], opt["width"])

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
    extra = {}
    if args.backbone == "wan2.1-i2v-14b":
        extra = dict(negative_prompt=args.negative_prompt, shift=args.shift)

    proc4d = None
    if args.process4d:
        from gem4d.process4d import Process4D, run_4d_process
        proc4d = Process4D(da3=args.process4d_da3, gap=args.process4d_gap, process_res=args.process4d_process_res,
                           device=dev, seg=args.process4d_seg, min_flow_px=args.process4d_min_flow_px)

    entries = ([{"video": args.image, "prompt": args.prompt, "robot": ""}] if args.image
               else load_manifest(args.manifest, args.data_root))
    for i, entry in enumerate(entries):
        prompt, stem = build_prompt(entry), f"{i:04d}_{Path(entry['video']).stem}"
        image = first_frame(entry["video"], size, compose=args.compose)
        for k in range(args.num_samples):
            frames = pipe(image=image, prompt=prompt, height=opt["height"], width=opt["width"],
                          num_frames=args.num_frames, num_inference_steps=opt["steps"],
                          guidance_scale=opt["guidance_scale"],
                          generator=torch.Generator(device=dev).manual_seed(opt["seed"] + k),
                          output_type="np", **extra).frames[0]
            name = stem if args.num_samples == 1 else f"{stem}_{k}"
            out = os.path.join(args.output_dir, name + ".mp4")
            save_mp4(frames, out, fps=opt["fps"])
            if args.side_by_side and Path(entry["video"]).suffix.lower() in (".mp4", ".webm", ".avi", ".mov"):
                gt = [center_crop_resize([f], size)[0] for f in read_video(Path(entry["video"]))]
                save_mp4(side_by_side(gt, to_uint8(frames)), out.replace(".mp4", "_gt_pred.mp4"), fps=opt["fps"])
            print(f"[{i}] {prompt!r} -> {out} ({len(frames)} frames)", flush=True)
            if proc4d is not None:
                st = run_4d_process(proc4d, to_uint8(frames), args.output_dir, name,
                                    crop=parse_crop(args.process4d_crop), fps=opt["fps"])["stats"]
                print(f"[{i}] 4D-process -> {out[:-4]}_4d.html  dynamic {100 * st['dynamic_fraction']:.2f}% "
                      f"of pixels, valid {100 * st['valid_fraction']:.1f}%", flush=True)


if __name__ == "__main__":
    main()
