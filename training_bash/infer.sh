#!/bin/bash
# GEM-4D inference. Examples:
#   bash training_bash/infer.sh --backbone wan2.1-i2v-14b --base_model weights/Wan2.1-I2V-14B-480P \
#        --weights weights/gem-4d-wan14b --image first_frame.png --prompt "open the drawer" --output_dir outputs/wan14b
#   bash training_bash/infer.sh --backbone wan2.2-ti2v-5b --base_model weights/Wan2.2-TI2V-5B-Diffusers \
#        --weights runs/wan5b_gem4d/checkpoints/step_010000 --manifest examples/val_manifest.json --output_dir outputs/wan5b
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/model:${ROOT}:${PYTHONPATH:-}"
exec python "${ROOT}/inference/infer.py" "$@"
