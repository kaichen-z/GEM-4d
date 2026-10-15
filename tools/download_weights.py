"""Download the GEM4D weights.

Everything S1 needs lives in one Hugging Face dataset repo, ``zhouk777/gem4d-weight``
(~58 GB total), organised as four folders:

    cogvideox/      the exact CogVideoX-5b-I2V S1 was initialised from  (~20 GB)
    da3/            DepthAnything-3 geometry teacher                    (~1.6 GB)
    checkpoint-1/   S1 step 21000  — use this to CONTINUE FINE-TUNING   (~12 GB)
    checkpoint-2/   S1 step 42000  — use this for TESTING / inference   (~21 GB)

Usage
-----
    python download_weights.py                      # everything (~58 GB)
    python download_weights.py cogvideox da3        # only what cogvideox/train.sh needs (GEO=1)
    python download_weights.py checkpoint-1         # only the fine-tuning checkpoint
    python download_weights.py --base               # upstream CogVideoX-5b-I2V from zai-org instead
    python download_weights.py wan                  # Wan2.1-Fun-V1.1-1.3B-InP base (alibaba-pai)
    python download_weights.py --dest /data/gem4d   # choose the destination root

The default destination is ``<repo>/weights``. ``cogvideox/`` is the full base model and can be
used directly as ``MODEL_PATH`` / ``COGVIDEO_BASE_PATH``; the ``cogvideox/CogVideoX-5b-I2V/`` directory
shipped with this repo holds only the configs. There are no released GEM-4D Wan checkpoints;
``wan`` fetches only the upstream base model into ``<dest>/Wan2.1-Fun-V1.1-1.3B-InP``.
"""
import argparse
import os
import sys

from huggingface_hub import snapshot_download

WEIGHT_REPO = "zhouk777/gem4d-weight"
FOLDERS = {
    "cogvideox": "CogVideoX-5b-I2V base weights S1 initialises from (~20 GB)",
    "da3": "DepthAnything-3 geometry teacher, DA3_PRETRAINED (~1.6 GB)",
    "checkpoint-1": "S1 step 21000 — start further fine-tuning from this (~12 GB)",
    "checkpoint-2": "S1 step 42000 — best for testing / inference (~21 GB)",
}
EXTERNAL = {  # name -> (HF model repo, local folder)
    "wan": ("alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP", "Wan2.1-Fun-V1.1-1.3B-InP"),
}
DEFAULT_DEST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="*",
                    help="which folders to fetch (" + ", ".join(sorted(FOLDERS) + sorted(EXTERNAL)) + "); default = all GEM-4D folders")
    ap.add_argument("--dest", default=DEFAULT_DEST, help=f"destination root (default: {DEFAULT_DEST})")
    ap.add_argument("--base", action="store_true",
                    help="instead fetch upstream zai-org/CogVideoX-5b-I2V into gem4d/CogVideoX-5b-I2V/")
    args = ap.parse_args()

    if args.base:
        snapshot_download(repo_id="zai-org/CogVideoX-5b-I2V",
                          local_dir=os.path.join(args.dest, "CogVideoX-5b-I2V"))
        return

    folders = args.folders or sorted(FOLDERS)
    unknown = [f for f in folders if f not in FOLDERS and f not in EXTERNAL]
    if unknown:
        sys.exit(f"unknown folder(s): {', '.join(unknown)}\nknown: {', '.join(sorted(FOLDERS))}")

    os.makedirs(args.dest, exist_ok=True)
    for name in folders:
        if name in EXTERNAL:
            repo, local = EXTERNAL[name]
            print(f"[{repo}] -> {os.path.join(args.dest, local)}")
            snapshot_download(repo_id=repo, local_dir=os.path.join(args.dest, local))
            continue
        print(f"[gem4d-weight] {name}: {FOLDERS[name]}")
        snapshot_download(repo_id=WEIGHT_REPO, repo_type="dataset",
                          allow_patterns=[f"{name}/*"],
                          local_dir=args.dest)
        print(f"[gem4d-weight] -> {os.path.join(args.dest, name)}")


if __name__ == "__main__":
    main()
