"""Download the weights GEM-4D trains from.

    da3            DepthAnything-3 geometry teacher (DA3-LARGE-1.1), from ``zhouk777/gem4d-weight``
                   (dataset)                                                (~1.6 GB) -> geometry.teacher
    wan2.2       Wan-AI/Wan2.2-TI2V-5B-Diffusers  (GEM-4D Wan-5B base)       (~34 GB)  -> backbone.base_model
    wan2.1       Wan-AI/Wan2.1-I2V-14B-480P       (GEM-4D Wan-14B base)      (~82 GB)  -> backbone.base_model

Usage
-----
    python tools/download_weights.py wan2.2 da3            # train GEM-4D Wan-5B
    python tools/download_weights.py wan2.1 da3            # train GEM-4D Wan-14B
    python tools/download_weights.py --dest /data/gem4d    # destination root (default <repo>/weights)
"""
import argparse
import os
import sys

from huggingface_hub import snapshot_download

FOLDERS = {  # name -> (HF dataset repo holding <name>/, description)
    "da3": ("zhouk777/gem4d-weight", "DepthAnything-3 geometry teacher (~1.6 GB)"),
}
EXTERNAL = {  # name -> (HF model repo, local folder)
    "wan2.2": ("Wan-AI/Wan2.2-TI2V-5B-Diffusers", "Wan2.2-TI2V-5B-Diffusers"),
    "wan2.1": ("Wan-AI/Wan2.1-I2V-14B-480P", "Wan2.1-I2V-14B-480P"),
}
DEFAULT_DEST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    known = sorted(FOLDERS) + sorted(EXTERNAL)
    ap.add_argument("folders", nargs="+", help="which weights to fetch (" + ", ".join(known) + ")")
    ap.add_argument("--dest", default=DEFAULT_DEST, help=f"destination root (default: {DEFAULT_DEST})")
    args = ap.parse_args()

    unknown = [f for f in args.folders if f not in known]
    if unknown:
        sys.exit(f"unknown folder(s): {', '.join(unknown)}\nknown: {', '.join(known)}")

    os.makedirs(args.dest, exist_ok=True)
    for name in args.folders:
        if name in EXTERNAL:
            repo, local = EXTERNAL[name]
            print(f"[{repo}] -> {os.path.join(args.dest, local)}")
            snapshot_download(repo_id=repo, local_dir=os.path.join(args.dest, local))
            continue
        repo, desc = FOLDERS[name]
        print(f"[{repo}] {name}: {desc}")
        snapshot_download(repo_id=repo, repo_type="dataset", allow_patterns=[f"{name}/*"], local_dir=args.dest)
        got = os.path.join(args.dest, name)
        if not (os.path.isdir(got) and os.listdir(got)):
            sys.exit(f"[{repo}] {name} is not in {repo} (yet)")
        print(f"[{repo}] -> {got}")


if __name__ == "__main__":
    main()
