#!/bin/bash
# =============================================================================
# GEM-4D training launcher — CogVideoX-5b-I2V backbone.
#
#   GEO=1 (default)  geometry forcing ("REF15"): a Geo-DiT reads the hidden state of
#                    block GEO_HOOK_LAYER and must reconstruct MASKED DepthAnything-3
#                    features (MASK_RATIO of tokens replaced by a learned mask token);
#                    loss = rgb_loss + LAMBDA_GEO * geo_loss. Needs DA3_PRETRAINED.
#   GEO=0            RGB-only fine-tuning (RGB_ONLY=1): no DA3, no Geo-DiT.
#
# Initialisation (MODEL_PATH):
#   - base model      MODEL_PATH=weights/cogvideox
#   - saved ckpt      MODEL_PATH=weights/checkpoint-1   (or any checkpoint-N/, step_NNNNNN/,
#                     OUTPUT_DIR/transformer/ — a dir holding config.json + *.safetensors,
#                     directly or under transformer/). Geo-DiT keys in it are ignored and a fresh
#                     Geo-DiT is built.
# Resuming an interrupted run: re-launch with the same OUTPUT_DIR. Full-state checkpoints
# (weights + optimizer + LR scheduler) live in $OUTPUT_DIR/ckpt/step_NNNNNN and the newest one
# is picked up automatically (SIMPLE_CKPT_DIR mechanism).
#
# Required:  DATASET_FILE (manifest, see gem4d/manifest.py), MODEL_PATH,
#            DA3_PRETRAINED (GEO=1 only)
# Example:
#   DATASET_FILE=data/train.json VALIDATION_DATASET_FILE=data/val.json \
#   MODEL_PATH=weights/checkpoint-1 COGVIDEO_BASE_PATH=weights/cogvideox \
#   DA3_PRETRAINED=weights/da3 MAX_TRAIN_STEPS=10000 bash cogvideox/train.sh
# LSF:  bsub -n 1 -gpu "num=8" -W 72:0 < cogvideox/train.sh   (export the variables first)
# =============================================================================
[ -f ~/.bashrc ] && source ~/.bashrc
set -uo pipefail

export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | grep -c .)}"

# ---- paths ------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"    # <code-git>/cogvideox
export PROJECT_ROOT="${PROJECT_ROOT:-$(dirname "${SCRIPT_DIR}")}"   # <code-git> (tools/, depth/)
export PYTHONPATH="${SCRIPT_DIR}:${PROJECT_ROOT}/depth/src:${PYTHONPATH:-}"
TRAIN_PY="${SCRIPT_DIR}/gem4d/i2v_depth_ref9.py"

DATASET_FILE="${DATASET_FILE:?set DATASET_FILE to a training manifest}"
DATA_ROOT="${DATA_ROOT:-$(dirname "${DATASET_FILE}")}"         # base for relative manifest paths
VALIDATION_DATASET_FILE="${VALIDATION_DATASET_FILE:-${DATASET_FILE}}"
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the base model or a saved checkpoint}"
export COGVIDEO_BASE_PATH="${COGVIDEO_BASE_PATH:-${MODEL_PATH}}"   # tokenizer/T5/VAE/scheduler

# ---- geometry forcing -------------------------------------------------------
GEO="${GEO:-1}"
if [[ "${GEO}" == "1" ]]; then
  export REF15=1; unset RGB_ONLY REF13 2>/dev/null || true
  export DA3_PRETRAINED="${DA3_PRETRAINED:?GEO=1 needs DA3_PRETRAINED (weights/da3)}"
  export MASK_RATIO="${MASK_RATIO:-0.5}"            # fraction of geometry tokens masked
  export LAMBDA_GEO="${LAMBDA_GEO:-0.03}"           # weight of the geometry loss
  export GEO_DEPTH="${GEO_DEPTH:-4}"                # Geo-DiT blocks
  export GEO_HOOK_LAYER="${GEO_HOOK_LAYER:-21}"     # CogVideoX block whose output feeds Geo-DiT (42 blocks)
  export DA3_FEAT_LAYER="${DA3_FEAT_LAYER:-12}"     # DA3 ViT-L layer used as geometry target
  export GEO_WARMUP="${GEO_WARMUP:-100}"            # linear ramp of LAMBDA_GEO
  export GEO_COUPLED_NOISE="${GEO_COUPLED_NOISE:-1}" # geometry latent shares the RGB noise
  export GEO_X0="${GEO_X0:-0}"                      # 1 = Geo-DiT predicts z0 instead of velocity
else
  export RGB_ONLY=1; unset REF15 REF13 2>/dev/null || true
fi

# ---- schedule ---------------------------------------------------------------
FRAME_LENGTH="${FRAME_LENGTH:-49}"
HEIGHT="${HEIGHT:-480}"; WIDTH="${WIDTH:-640}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
LR_SCHEDULER="${LR_SCHEDULER:-constant_with_warmup}"
LR_WARMUP="${LR_WARMUP:-100}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-10000}"
VALIDATION_STEPS="${VALIDATION_STEPS:-500}"
CKPT_EVERY_N_STEP="${CKPT_EVERY_N_STEP:-500}"
export CKPT_TOPK="${CKPT_TOPK:-3}"                       # keep newest K full-state ckpts ...
export CKPT_KEEP_INTERVAL="${CKPT_KEEP_INTERVAL:-2000}"  # ... plus every multiple of this step
export RUN_VAL_VIDEO="${RUN_VAL_VIDEO:-1}"               # write GT|pred videos at each validation

OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/gem4d-cogvideox-geo${GEO}__${HEIGHT}x${WIDTH}__f${FRAME_LENGTH}}"
export SIMPLE_CKPT_DIR="${OUTPUT_DIR}/ckpt"
export VAL_VIZ_DIR="${VAL_VIZ_DIR:-${OUTPUT_DIR}/vis}"
mkdir -p "${SIMPLE_CKPT_DIR}" "${VAL_VIZ_DIR}"
export WANDB_DIR="${OUTPUT_DIR}"

echo "[gem4d-cogvideox] GEO=${GEO} MASK_RATIO=${MASK_RATIO:-} LAMBDA_GEO=${LAMBDA_GEO:-} init=${MODEL_PATH}"
echo "[gem4d-cogvideox] data=${DATASET_FILE} gpus=${NUM_GPUS} lr=${LEARNING_RATE} steps=${MAX_TRAIN_STEPS} out=${OUTPUT_DIR}"

ARGS=(
  --pretrained_model_name_or_path "${MODEL_PATH}"
  --dataset_file "${DATASET_FILE}" --data_root "${DATA_ROOT}"
  --dataloader_num_workers "${NUM_WORKERS}"
  --pin_memory --seed 42 --mixed_precision bf16
  --output_dir "${OUTPUT_DIR}"
  --height "${HEIGHT}" --width "${WIDTH}" --guidance_scale 6.0
  --max_num_frames "${FRAME_LENGTH}"
  --train_batch_size 1 --gradient_accumulation_steps "${GRAD_ACCUM}"
  --learning_rate "${LEARNING_RATE}" --lr_scheduler "${LR_SCHEDULER}" --lr_warmup_steps "${LR_WARMUP}"
  --max_train_steps "${MAX_TRAIN_STEPS}"
  --checkpointing_steps "${CKPT_EVERY_N_STEP}"
  --noised_image_dropout 0.05 --gradient_checkpointing --enable_slicing --enable_tiling
  --optimizer adamw --beta1 0.9 --beta2 0.95 --weight_decay 0.001 --max_grad_norm 1.0 --allow_tf32
  --report_to wandb --ignore_learned_positional_embeddings --nccl_timeout 1800
  --num_validation_videos 1 --validation_steps "${VALIDATION_STEPS}"
  --validation_prompt "${VALIDATION_DATASET_FILE}" --validation_prompt_separator :::
  --validation_dataset_file "${VALIDATION_DATASET_FILE}" --validation_loss_num_batches 4
  --height_buckets "${HEIGHT}" --width_buckets "${WIDTH}" --frame_buckets "${FRAME_LENGTH}"
  --run_validation_after_first_step
)
# shellcheck disable=SC2206
[[ -n "${EXTRA_TRAIN_ARGS:-}" ]] && ARGS+=(${EXTRA_TRAIN_ARGS})

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "accelerate launch ${TRAIN_PY} ${ARGS[*]}"; exit 0
fi

ACC_CFG="${OUTPUT_DIR}/accelerate_zero2_${NUM_GPUS}gpu.yaml"
cat > "${ACC_CFG}" <<EOF
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
mixed_precision: bf16
num_machines: 1
num_processes: ${NUM_GPUS}
machine_rank: 0
main_training_function: main
rdzv_backend: static
same_network: true
downcast_bf16: "no"
use_cpu: false
deepspeed_config:
  zero_stage: 2
  gradient_accumulation_steps: ${GRAD_ACCUM}
  offload_optimizer_device: none
  offload_param_device: none
  zero3_init_flag: false
EOF
accelerate launch --config_file "${ACC_CFG}" --main_process_port "${MAIN_PROCESS_PORT:-$((20000 + RANDOM % 20000))}" \
  "${TRAIN_PY}" "${ARGS[@]}"
