"""Merge a LoRA adapter into its base DiT and export a plain checkpoint (no adapter needed afterwards).

    W' = W + (alpha / r) * B A        (fp32 sum, stored in the DiT's dtype)

    python tools/merge_lora.py --backbone wan2.1-i2v-14b --base_model weights/Wan2.1-I2V-14B-480P \
        --weights weights/gem-4d-wan14b --lora runs/wan14b_gem4d_lora/checkpoints/step_010000 \
        --output weights/gem-4d-wan14b-mytask

The output has the layout of a training export, <output>/transformer/ (Wan-14B: config.json +
diffusion_pytorch_model-*.safetensors, original Wan2.1 names; Wan-5B: diffusers), and loads with `inference/infer.py --weights <output>` or as
`backbone.init_weights` for further training.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "model"), str(ROOT)]

from gem4d import build_backbone                                    # noqa: E402
from gem4d.models.lora import load_lora, read_lora                  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", required=True, choices=["wan2.1-i2v-14b", "wan2.2-ti2v-5b"])
    ap.add_argument("--base_model", required=True, help="full model dir of the backbone")
    ap.add_argument("--weights", default=None, help="DiT the adapter was trained on (default: base)")
    ap.add_argument("--lora", required=True, help="adapter: step_N dir, lora/ dir, lora.safetensors or .pt")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    kw = {"load_text_encoder": False} if args.backbone == "wan2.1-i2v-14b" else {}
    bb = build_backbone(args.backbone, base_model=args.base_model, init_weights=args.weights, **kw)
    n = load_lora(bb.transformer, args.lora, scale=args.scale)
    bb.transformer.save_pretrained(os.path.join(args.output, "transformer"), safe_serialization=True,
                                   max_shard_size="5GB")
    _, cfg = read_lora(args.lora)
    with open(os.path.join(args.output, "merge_info.json"), "w") as f:
        json.dump({"backbone": args.backbone, "weights": args.weights or args.base_model, "lora": args.lora,
                   "lora_config": {k: cfg[k] for k in ("r", "alpha") if k in cfg}, "layers": n, "scale": args.scale,
                   "merge": "W' = W + scale * (alpha / r) * B @ A"}, f, indent=1)
    print(f"merged {n} LoRA layers -> {args.output}")


if __name__ == "__main__":
    main()
