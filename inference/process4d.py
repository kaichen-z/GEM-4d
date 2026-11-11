"""4D-process on existing videos: DA3 depth + cameras, RAFT-small optical flow -> world-space scene
flow, written as an interactive viewer (<stem>_4d.html), a summary video (<stem>_4d.mp4: frame |
depth with scene-flow arrows on the moving foreground), statistics (<stem>_4d.json) and arrays (<stem>_4d.npz).

    python inference/process4d.py --videos outputs/wan14b/*.mp4 --da3 weights/da3 --output_dir outputs/wan14b_4d \
        --crop 0,0,302,240          # main view of the 416x240 composite robot videos

--viewer_all <path>: one extra viewer page with every processed video as a tab.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "model"), str(ROOT)]

from gem4d.process4d import Process4D, run_4d_process, viewer_sequence, write_viewer  # noqa: E402
from training.data.video_dataset import read_video                                    # noqa: E402


def parse_crop(s):
    return tuple(int(v) for v in s.split(",")) if s else None


def add_args(ap, prefix=""):
    ap.add_argument(f"--{prefix}da3", default="depth-anything/DA3-LARGE-1.1", help="DA3 weights (folder or HF id)")
    ap.add_argument(f"--{prefix}crop", default=None, help="x0,y0,x1,y1: process this region (one camera view)")
    ap.add_argument(f"--{prefix}gap", type=int, default=3, help="scene flow from frame t to t+gap")
    ap.add_argument(f"--{prefix}process_res", type=int, default=504, help="DA3 processing resolution")
    ap.add_argument(f"--{prefix}seg", default="CIDAS/clipseg-rd64-refined",
                    help="CLIPSeg weights for the foreground mask (folder or HF id); 'none': no foreground gate")
    ap.add_argument(f"--{prefix}min_flow_px", type=float, default=1.5,
                    help="minimum optical flow (pixels) after removing the camera shake for a dynamic point")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--no_npz", action="store_true")
    ap.add_argument("--viewer_all", default=None)
    add_args(ap)
    args = ap.parse_args()
    proc = Process4D(da3=args.da3, gap=args.gap, process_res=args.process_res, seg=args.seg,
                     min_flow_px=args.min_flow_px)
    seqs = []
    for v in args.videos:
        stem = Path(v).stem
        frames = read_video(Path(v))
        out = run_4d_process(proc, frames, args.output_dir, stem, crop=parse_crop(args.crop), fps=args.fps,
                             save_npz=not args.no_npz)
        print(f"{v} -> {os.path.join(args.output_dir, stem)}_4d.* {json.dumps(out['stats'])}", flush=True)
        if args.viewer_all:
            seqs.append(viewer_sequence(out["result"], stem))
    if args.viewer_all:
        write_viewer(args.viewer_all, seqs)
        print("viewer", args.viewer_all)


if __name__ == "__main__":
    main()
