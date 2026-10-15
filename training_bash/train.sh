#!/bin/bash
# GEM-4D training launcher (single node, accelerate + DeepSpeed ZeRO-2).
#
#   bash training_bash/train.sh training/config/cogvideox_gem4d.yaml data.train_manifest=/data/train.json
#   NUM_GPUS=4 bash training_bash/train.sh training/config/wan22_ti2v_gem4d.yaml \
#        data.train_manifest=/data/train.json data.val_manifest=/data/val.json train.max_steps=20000
#
# Any config key can be overridden as key.sub=value. Re-running with the same experiment.output_dir
# resumes from <output_dir>/checkpoints/state automatically.
# LSF: bsub -n 1 -gpu "num=8" -W 72:0 "bash training_bash/train.sh <config> <overrides>"
set -euo pipefail
CONFIG="${1:?usage: train.sh <config.yaml> [key=value ...]}"; shift
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}/model:${ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
PORT="${MAIN_PROCESS_PORT:-$((20000 + RANDOM % 20000))}"
# DeepSpeed takes gradient accumulation / clipping from the launcher: read them from the config.
read -r ACCUM CLIP < <(python -c "import sys; from training.train_utils.config import load_config as L; \
c = L(sys.argv[1], sys.argv[2:]); print(c.optim.grad_accum, c.optim.max_grad_norm)" "${CONFIG}" "$@")
echo "[gem4d] config=${CONFIG} gpus=${NUM_GPUS} grad_accum=${ACCUM} clip=${CLIP} overrides=$*"
exec accelerate launch --config_file training/config/accelerate_zero2.yaml --num_processes "${NUM_GPUS}" \
  --gradient_accumulation_steps "${ACCUM}" --gradient_clipping "${CLIP}" \
  --main_process_port "${PORT}" -m training.launch --config "${CONFIG}" "$@"
