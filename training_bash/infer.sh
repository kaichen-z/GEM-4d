#!/bin/bash
# GEM-4D inference. Examples:
#   bash training_bash/infer.sh --backbone cogvideox --base_model weights/cogvideox \
#        --weights weights/checkpoint-2 --manifest examples/val_manifest.json --output_dir outputs/ckpt2 --side_by_side
#   bash training_bash/infer.sh --backbone wan2.2-ti2v-5b --base_model weights/Wan2.2-TI2V-5B-Diffusers \
#        --weights runs/wan22_ti2v_gem4d/checkpoints/step_010000 --manifest examples/val_manifest.json --output_dir outputs/wan
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/model:${ROOT}:${PYTHONPATH:-}"
exec python "${ROOT}/inference/infer.py" "$@"
