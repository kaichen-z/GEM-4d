"""Download GEM-4D weights.

GEM-4D (CogVideoX backbone) weights live in the Hugging Face dataset repo ``zhouk777/gem4d-weight``:

    cogvideox/      CogVideoX-5b-I2V base model GEM-4D starts from     (~20 GB)  -> backbone.base_model
    da3/            DepthAnything-3 geometry teacher                    (~1.6 GB) -> geometry.teacher
    checkpoint-1/   GEM-4D step 21000, for further fine-tuning          (~12 GB)  -> backbone.init_weights
    checkpoint-2/   GEM-4D step 42000, for inference                    (~21 GB)  -> inference --weights

The Wan backbone uses the upstream base model only (no released GEM-4D Wan weights yet):

    wan2.2          Wan-AI/Wan2.2-TI2V-5B-Diffusers                     (~34 GB)  -> backbone.base_model

Usage
-----
    python tools/download_weights.py                       # the four GEM-4D folders (~58 GB)
    python tools/download_weights.py cogvideox da3         # train CogVideoX GEM-4D from the base model
    python tools/download_weights.py cogvideox checkpoint-2   # CogVideoX inference
    python tools/download_weights.py wan2.2 da3            # train Wan2.2-TI2V-5B GEM-4D
    python tools/download_weights.py --base                # upstream zai-org/CogVideoX-5b-I2V instead
    python tools/download_weights.py --dest /data/gem4d    # destination root (default <repo>/weights)
"""
import argparse
import os
import sys

from huggingface_hub import snapshot_download

WEIGHT_REPO = "zhouk777/gem4d-weight"
FOLDERS = {
    "cogvideox": "CogVideoX-5b-I2V base model (~20 GB)",
    "da3": "DepthAnything-3 geometry teacher (~1.6 GB)",
    "checkpoint-1": "GEM-4D step 21000, for fine-tuning (~12 GB)",
    "checkpoint-2": "GEM-4D step 42000, for inference (~21 GB)",
}
EXTERNAL = {  # name -> (HF model repo, local folder)
    "wan2.2": ("Wan-AI/Wan2.2-TI2V-5B-Diffusers", "Wan2.2-TI2V-5B-Diffusers"),
}
DEFAULT_DEST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="*",
                    help="which folders to fetch (" + ", ".join(sorted(FOLDERS) + sorted(EXTERNAL)) + "); default = all GEM-4D folders")
    ap.add_argument("--dest", default=DEFAULT_DEST, help=f"destination root (default: {DEFAULT_DEST})")
    ap.add_argument("--base", action="store_true",
                    help="instead fetch upstream zai-org/CogVideoX-5b-I2V into <dest>/CogVideoX-5b-I2V")
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
