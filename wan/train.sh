#!/bin/bash
# =============================================================================
# GEM-4D training launcher — Wan2.1-Fun-V1.1-1.3B-InP backbone (VideoX-Fun).
#
# EXPERIMENTAL: earlier runs of this backbone (lr 1e-4) produced noise-only samples
# from ~step 360 on; the default lr here is lowered to 1e-5. Watch the first
# validation videos (VALIDATION_STEPS) before letting a long run continue.
#
#   GEO=1 (default)  geometry forcing: forward hook on transformer block GEO_HOOK_LAYER
#                    feeds a Geo-DiT that reconstructs the MASKED geometry latent
#                    (DA3 depth computed on the fly, encoded by the Wan VAE); MASK_RATIO of
#                    tokens are replaced by a learned mask token. loss += LAMBDA_GEO * geo_mse.
#   GEO=0            plain RGB fine-tuning (scripts/wan2.1_fun/train.py).
#
# Initialisation:
#   MODEL_PATH       base Wan2.1-Fun-V1.1-1.3B-InP directory (always needed: VAE/T5/CLIP/config)
#   INIT_TRANSFORMER optional saved weights to start from, e.g.
#                    <old OUT>/checkpoint-N/transformer/diffusion_pytorch_model.safetensors
# Resuming: re-launch with the same OUT; the newest OUT/checkpoint-N (weights + optimizer +
# geo_dit.pt) is resumed automatically.
#
# Required: DATASET_FILE (manifest, videox_fun/data/manifest.py), MODEL_PATH,
#           DA3_PRETRAINED (GEO=1)
# Example:
#   DATASET_FILE=data/train.json VALIDATION_DATASET_FILE=data/val.json \
#   MODEL_PATH=weights/Wan2.1-Fun-V1.1-1.3B-InP DA3_PRETRAINED=weights/da3 bash wan/train.sh
# =============================================================================
[ -f ~/.bashrc ] && source ~/.bashrc
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"    # <code-git>/wan
ROOT="$(dirname "${SCRIPT_DIR}")"                             # <code-git>
export PYTHONPATH="${SCRIPT_DIR}:${ROOT}/depth/src:${PYTHONPATH:-}"
export GEO_TESSERACT="${GEO_TESSERACT:-${ROOT}/cogvideox/gem4d}"   # holds modules/geo_dit.py
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | grep -c .)}"

DATASET_FILE="${DATASET_FILE:?set DATASET_FILE to a training manifest}"
VALIDATION_DATASET_FILE="${VALIDATION_DATASET_FILE:-}"
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to Wan2.1-Fun-V1.1-1.3B-InP}"
INIT_TRANSFORMER="${INIT_TRANSFORMER:-}"
CONFIG="${CONFIG:-${SCRIPT_DIR}/config/wan2.1/wan_civitai.yaml}"

GEO="${GEO:-1}"
if [[ "${GEO}" == "1" ]]; then
  TRAIN_PY="${SCRIPT_DIR}/scripts/wan2.1_fun/train_geo.py"
  export GEO=1
  export GEO_DA3_CKPT="${DA3_PRETRAINED:?GEO=1 needs DA3_PRETRAINED (weights/da3)}"
  export MASK_RATIO="${MASK_RATIO:-0.7}"            # fraction of geometry tokens masked (geo_dit.py)
  export LAMBDA_GEO="${LAMBDA_GEO:-0.5}"
  export GEO_HOOK_LAYER="${GEO_HOOK_LAYER:-15}"     # 30 Wan blocks -> midpoint
  export GEO_WARMUP="${GEO_WARMUP:-100}"
  export GEO_COUPLED_NOISE="${GEO_COUPLED_NOISE:-1}"
  export GEO_EPS_HAT="${GEO_EPS_HAT:-0}"            # 1 = also feed stop-grad RGB noise estimate to Geo-DiT
  export GEO_DIM="${GEO_DIM:-1536}" GEO_HEADS="${GEO_HEADS:-12}" GEO_DEPTH="${GEO_DEPTH:-4}"
  VISION_ARG="--pretrained_vision_model_path=${GEO_DA3_CKPT}"
else
  TRAIN_PY="${SCRIPT_DIR}/scripts/wan2.1_fun/train.py"
  unset GEO 2>/dev/null || true
  VISION_ARG=""
fi

NUM_FRAMES="${NUM_FRAMES:-49}"              # must be 4k+1
SAMPLE_SIZE="${SAMPLE_SIZE:-512}"
LEARNING_RATE="${LEARNING_RATE:-1e-05}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-10000}"
CHECKPOINT_SAVE_STEPS="${CHECKPOINT_SAVE_STEPS:-500}"
CHECKPOINT_TOPK="${CHECKPOINT_TOPK:-2}"
CHECKPOINT_KEEP_INTERVAL="${CHECKPOINT_KEEP_INTERVAL:-2000}"
VALIDATION_STEPS="${VALIDATION_STEPS:-500}"
NUM_WORKERS="${NUM_WORKERS:-8}"
OUT="${OUT:-${ROOT}/runs/gem4d-wan1.3b-geo${GEO:-0}}"
mkdir -p "${OUT}"
# the launcher cd's into wan/, so make user paths absolute
_abs() { [[ -z "$1" || "$1" = /* || ! -e "$1" ]] && echo "$1" || echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; }
DATASET_FILE="$(_abs "${DATASET_FILE}")"; VALIDATION_DATASET_FILE="$(_abs "${VALIDATION_DATASET_FILE}")"
MODEL_PATH="$(_abs "${MODEL_PATH}")"; INIT_TRANSFORMER="$(_abs "${INIT_TRANSFORMER}")"; OUT="$(_abs "${OUT}")"
[[ -n "${VISION_ARG}" ]] && { export GEO_DA3_CKPT="$(_abs "${GEO_DA3_CKPT}")"; VISION_ARG="--pretrained_vision_model_path=${GEO_DA3_CKPT}"; }

EXTRA=()
[[ -n "${INIT_TRANSFORMER}" ]] && EXTRA+=(--transformer_path="${INIT_TRANSFORMER}")
[[ -n "${VALIDATION_DATASET_FILE}" ]] && EXTRA+=(--validation_meta="${VALIDATION_DATASET_FILE}")
[[ -n "${VISION_ARG}" ]] && EXTRA+=("${VISION_ARG}")

echo "[gem4d-wan] GEO=${GEO:-0} MASK_RATIO=${MASK_RATIO:-} init=${INIT_TRANSFORMER:-<base>} lr=${LEARNING_RATE} out=${OUT}"
CMD=(accelerate launch --num_processes="${NUM_GPUS}" --num_machines=1 --mixed_precision=bf16 --dynamo_backend=no
  "${TRAIN_PY}"
  --config_path="${CONFIG}" --pretrained_model_name_or_path="${MODEL_PATH}"
  --train_data_dir="" --train_data_meta="${DATASET_FILE}"
  --image_sample_size="${SAMPLE_SIZE}" --video_sample_size="${SAMPLE_SIZE}" --token_sample_size="${SAMPLE_SIZE}"
  --video_sample_stride=1 --video_sample_n_frames="${NUM_FRAMES}"
  --train_batch_size=1 --video_repeat=1 --gradient_accumulation_steps=1
  --dataloader_num_workers="${NUM_WORKERS}" --max_train_steps="${MAX_TRAIN_STEPS}"
  --checkpointing_steps="${CHECKPOINT_SAVE_STEPS}" --checkpoints_total_limit="${CHECKPOINT_TOPK}"
  --checkpoint_keep_interval="${CHECKPOINT_KEEP_INTERVAL}" --resume_from_checkpoint="latest"
  --learning_rate="${LEARNING_RATE}" --lr_scheduler="constant_with_warmup" --lr_warmup_steps=200 --seed=42
  --output_dir="${OUT}" --validation_output_dir="${OUT}" --logging_dir="tensorboard" --report_to=tensorboard
  --gradient_checkpointing --mixed_precision="bf16"
  --adam_weight_decay=3e-2 --adam_epsilon=1e-10 --vae_mini_batch=1 --max_grad_norm=1.0
  --enable_bucket --uniform_sampling --low_vram
  --train_mode="inpaint" --trainable_modules "."
  --validation_steps="${VALIDATION_STEPS}"
  "${EXTRA[@]}")
[[ "${DRY_RUN:-0}" == "1" ]] && { echo "${CMD[*]}"; exit 0; }
cd "${SCRIPT_DIR}" && "${CMD[@]}"
