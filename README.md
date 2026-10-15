<div align="center">

  <h1>GEM-4D: Geometry-Enhanced Video World Models<br/>for Robot Manipulation</h1>

  <p>
    <b>Kaichen Zhou</b><sup>1,2,*</sup> &nbsp;
    <b>Yuzhen Chen</b><sup>1,*</sup> &nbsp;
    <b>Fangneng Zhan</b><sup>2</sup> &nbsp;
    <b>Hang Hua</b><sup>4</sup> &nbsp;
    <b>Grace Chen</b><sup>1</sup> &nbsp;
    <b>Xinhai Chang</b><sup>2</sup><br/>
    <b>Ao Qu</b><sup>2</sup> &nbsp;
    <b>Yilun Du</b><sup>1</sup> &nbsp;
    <b>Zhuang Liu</b><sup>3</sup> &nbsp;
    <b>Paul Pu Liang</b><sup>2,†</sup> &nbsp;
    <b>Mengyu Wang</b><sup>1,†</sup>
  </p>

  <p>
    <sup>1</sup>Harvard University &nbsp;&nbsp;
    <sup>2</sup>MIT &nbsp;&nbsp;
    <sup>3</sup>Princeton University &nbsp;&nbsp;
    <sup>4</sup>MIT-IBM Watson AI Lab
  </p>

  <p><sup>*</sup>Equal contribution &nbsp;&nbsp; <sup>†</sup>Joint supervision</p>

  <p>
    <a href="https://arxiv.org/abs/2605.22882"><img src="https://img.shields.io/badge/arXiv-2605.22882-F4B8B0?labelColor=FDE6E2" alt="arXiv" /></a>
    &nbsp;
    <a href="https://gem-4d.github.io/"><img src="https://img.shields.io/badge/Project-Page-F6D9A0?labelColor=FFF3DC" alt="Project Page" /></a>
    &nbsp;
    <a href="https://huggingface.co/datasets/zhouk777/gem4d-weight"><img src="https://img.shields.io/badge/Weights-HuggingFace-C9E4CA?labelColor=EAF5EB" alt="Weights" /></a>
  </p>

</div>

![GEM-4D vs. TesserAct: from one observation and an instruction, GEM-4D generates a rollout whose geometry stays consistent enough to act on.](assets/teaser.jpg)

GEM-4D is an image + text → video world model for robot manipulation. During training, a
small **Geo-DiT** branch reads the hidden state of a middle transformer block and must
reconstruct a **masked geometry latent** derived from a frozen DepthAnything-3 (DA3) model.
Most geometry tokens are replaced by a learned mask token, so the only way to fill them is to
read geometry out of the video features — the gradient pushes 3D consistency into the video
backbone. The branch is **dropped at inference**: the deployed model is an ordinary
image-to-video generator with zero extra cost.

```
first frame + prompt ──► video DiT ──────────────► RGB velocity ──► L_rgb
                           │ hidden state @ GEO_HOOK_LAYER
                           ▼
   masked, noised geometry latent ──► Geo-DiT ──► geometry velocity ──► L_geo
   (target: frozen DepthAnything-3)
                       loss = L_rgb + LAMBDA_GEO · L_geo
```

This repository contains the GEM-4D implementation on two backbones:

| Backbone | Folder | Status |
|---|---|---|
| **CogVideoX-5b-I2V** (42 blocks, ~5.7 B) | [`cogvideox/`](cogvideox) | main model, released checkpoints |
| **Wan2.1-Fun-V1.1-1.3B-InP** (30 blocks, 1.3 B) | [`wan/`](wan) | experimental — see [Wan status](#wan-status) |

To use the model, follow the numbered sections: install the environment and download weights →
write a data manifest → fine-tune (optionally from a released checkpoint) → run inference.

- [0. Environment](#0-environment)
- [1. Pretrained weights](#1-pretrained-weights)
- [2. Data: manifest format](#2-data-manifest-format)
- [3. Training / fine-tuning with masked geometry](#3-training--fine-tuning-with-masked-geometry)
- [4. Inference](#4-inference)
- [Wan status](#wan-status) · [Known caveats](#known-caveats) · [Citation](#citation)

## Code layout

| Path | Purpose |
|---|---|
| `cogvideox/train.sh` | CogVideoX training launcher (`GEO=1` geometry forcing / `GEO=0` RGB-only) |
| `cogvideox/inference.py` | CogVideoX checkpoint → videos |
| `cogvideox/gem4d/i2v_depth_ref9.py` | trainer entry point (accelerate + DeepSpeed ZeRO-2) |
| `cogvideox/gem4d/manifest.py` | manifest parser shared by training, validation and inference |
| `cogvideox/gem4d/custom_dataset.py` | generic video dataset — the place to customise data loading |
| `cogvideox/gem4d/modules/` | `gem4d_model.py` (CogVideoX subclass), `geo_dit.py` (Geo-DiT), `gem4d_pipeline.py` |
| `cogvideox/CogVideoX-5b-I2V/` | model configs only (weights are downloaded) |
| `wan/train.sh` | Wan training launcher (`GEO=1` / `GEO=0`) |
| `wan/scripts/wan2.1_fun/` | `train_geo.py` (geometry forcing), `train.py` (RGB-only) |
| `wan/examples/wan2.1_fun/predict_i2v.py` | Wan inference |
| `wan/videox_fun/` | VideoX-Fun package; `data/manifest.py` reads the same manifest format |
| `depth/src/depth_anything_3/` | DepthAnything-3, frozen geometry teacher |
| `tools/` | `download_weights.py`, full-state checkpoint helpers, validation visualisation |
| `examples/` | example training / validation manifests |

## 0. Environment

```bash
conda create -n gem4d python=3.10 -y
conda activate gem4d
pip install -r requirements.txt
```

The launchers set `PYTHONPATH` themselves (`cogvideox/` or `wan/`, plus `depth/src`). Run all
commands from the repository root. Training uses 8 GPUs by default (`CUDA_VISIBLE_DEVICES`);
every launcher accepts `DRY_RUN=1` to print the resolved command.

## 1. Pretrained weights

All CogVideoX weights are in the Hugging Face dataset
[`zhouk777/gem4d-weight`](https://huggingface.co/datasets/zhouk777/gem4d-weight) (~58 GB):

| Folder | Size | Content | Use as |
|---|---|---|---|
| `cogvideox/` | ~20 GB | CogVideoX-5b-I2V that GEM-4D starts from | `MODEL_PATH` (from scratch); always `COGVIDEO_BASE_PATH` / `--base_model` |
| `da3/` | ~1.6 GB | DepthAnything-3 geometry teacher | `DA3_PRETRAINED` (`GEO=1` only) |
| `checkpoint-1/` | ~12 GB | GEM-4D, step 21000 | `MODEL_PATH` to **continue fine-tuning** |
| `checkpoint-2/` | ~21 GB | GEM-4D, step 42000 | `--weights_path` for **inference** |

```bash
python tools/download_weights.py                              # all four -> ./weights/
python tools/download_weights.py cogvideox da3                # train from the base model
python tools/download_weights.py cogvideox da3 checkpoint-1   # fine-tune from GEM-4D
python tools/download_weights.py cogvideox checkpoint-2       # inference only
python tools/download_weights.py wan da3                      # Wan2.1-Fun-V1.1-1.3B-InP base
```

`checkpoint-1/2` contain only the fine-tuned transformer; tokenizer, T5, VAE and scheduler come
from `cogvideox/`. Upstream sources: [`zai-org/CogVideoX-5b-I2V`](https://huggingface.co/zai-org/CogVideoX-5b-I2V),
[`depth-anything/DA3-LARGE-1.1`](https://huggingface.co/depth-anything/DA3-LARGE-1.1) (default
`DA3_PRETRAINED`, fetched from the hub if unset),
[`alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP`](https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP).
There are no released GEM-4D Wan checkpoints.


## 2. Data: manifest format

There are **no predefined datasets** — no dataset names, path patterns or robot tags in the code.
Each clip is listed in a **manifest** (JSON list, or JSONL):

```json
[
  {"video": "rlbench/close_drawer/ep_0001/front.mp4", "prompt": "close the middle drawer", "robot": "Franka robot arm"},
  {"video": "my_lab/frames_0042", "prompt": "push the red block to the left"},
  ["pick orange from white bowl", "rt1/record_00162/49/rgb.mp4", "google robot"]
]
```

| Field | Meaning |
|---|---|
| `video` | `.mp4` / `.webm`, or a folder of frames `0.png, 1.png, …`; for validation and inference also a single image (used as the first frame). Relative paths resolve against `DATA_ROOT` (default: the manifest's folder). Aliases: `video_path`, `path`, `rgb`, `image`. |
| `prompt` | task instruction. Aliases: `instruction`, `text`, `caption`. |
| `robot` | optional embodiment text, appended as `"<prompt> <robot>"`. Use the same string for training and testing; avoid ambiguous words (e.g. "Panda"). |

The list forms `[prompt, video]` and `[prompt, video, robot]` are also accepted. A manifest
argument may be a directory; every `*.json` / `*.jsonl` inside is concatenated. Ready-made
examples: [`examples/train_manifest.json`](examples/train_manifest.json),
[`examples/val_manifest.json`](examples/val_manifest.json).

How a clip is loaded (`cogvideox/gem4d/custom_dataset.py`):

- frames are center-cropped to the target aspect ratio and resized to `HEIGHT × WIDTH` (480 × 640);
- the clip is brought to `FRAME_LENGTH` (49) frames: longer clips are subsampled uniformly,
  **shorter clips are padded by repeating the last frame** — prefer clips with ≥ 49 frames;
- frame 0 is the conditioning image; prompts are lower-cased and dropped to `""` 5 % of the time;
- no depth or normal files are needed — the geometry target is computed on the fly by DA3.

Wan reads the same manifest (`wan/videox_fun/data/manifest.py`); clips are resized to 512 and
sampled to `NUM_FRAMES` (49, must be 4k+1).

**Customising the loader.** For new data, normally only the manifest changes. For custom decoding,
cropping or augmentation, subclass `ManifestVideoDataset` and override `read_frames(path)`,
`resize(frames, entry)` or `get_prompt(entry)` — extra manifest keys are available in `entry` —
then point the two `ManifestVideoDataset(...)` calls in `cogvideox/gem4d/i2v_depth_ref9.py` at
your class. For Wan, the conversion is in `wan/videox_fun/data/dataset_image_video.py`
(search for "GEM-4D manifest").

## 3. Training / fine-tuning with masked geometry

### CogVideoX

Fine-tune from the released GEM-4D checkpoint with the masked-geometry objective:

```bash
DATASET_FILE=/data/my_train.json \
VALIDATION_DATASET_FILE=/data/my_val.json \
MODEL_PATH=weights/checkpoint-1 \
COGVIDEO_BASE_PATH=weights/cogvideox \
DA3_PRETRAINED=weights/da3 \
GEO=1 MAX_TRAIN_STEPS=10000 OUTPUT_DIR=runs/my_geo_ft \
bash cogvideox/train.sh
```

On an LSF cluster, export the same variables and submit
`bsub -n 1 -gpu "num=8" -W 72:0 < cogvideox/train.sh`.

- **Starting point.** `MODEL_PATH` is the base model (`weights/cogvideox`) or any saved
  transformer: the released `checkpoint-1`, an earlier `checkpoint-N/`, a full-state
  `ckpt/step_NNNNNN/`, or `OUTPUT_DIR/transformer/`. Geo-DiT weights stored in a checkpoint are
  not reused; a fresh Geo-DiT is built.
- **RGB-only.** `GEO=0` trains without DA3 and Geo-DiT, with the same command.
- **Geometry-forcing settings** (defaults in `cogvideox/train.sh`):

  | Variable | Default | Meaning |
  |---|---|---|
  | `MASK_RATIO` | 0.5 | fraction of geometry tokens replaced by the mask token |
  | `LAMBDA_GEO` | 0.03 | geometry-loss weight, ramped in over `GEO_WARMUP` = 100 steps |
  | `GEO_HOOK_LAYER` | 21 | CogVideoX block (of 42) whose output feeds the Geo-DiT |
  | `DA3_FEAT_LAYER` | 12 | DA3 ViT-L layer used as the target (→ frozen 1×1 projections → 16-ch latent) |
  | `GEO_DEPTH` | 4 | number of Geo-DiT blocks |
  | `GEO_COUPLED_NOISE` | 1 | geometry latent is noised with the same ε as RGB |
  | `GEO_X0` | 0 | 1 = Geo-DiT regresses z₀ instead of the velocity |

- **Schedule.** `LEARNING_RATE` 1e-5, `LR_SCHEDULER` constant_with_warmup, batch 1 per GPU,
  49 × 480 × 640, bf16.
- **Checkpoints.** Every `CKPT_EVERY_N_STEP` (500) into
  `OUTPUT_DIR/ckpt/step_NNNNNN/{transformer/, train_state.pt}` (weights + optimizer + scheduler,
  ~19 GB each). The newest `CKPT_TOPK` (3) are kept, plus every multiple of
  `CKPT_KEEP_INTERVAL` (2000).
- **Resume.** Re-run the same command with the same `OUTPUT_DIR`; the newest `ckpt/step_*` is
  loaded automatically and `MODEL_PATH` is ignored. Do not use `--resume_from_checkpoint`
  with `GEO=1`.
- **Validation.** Every `VALIDATION_STEPS` (500): `val/rgb_loss`, and GT | prediction videos for
  the validation manifest in `OUTPUT_DIR/vis/Step_NNNNNN/`.

### Wan2.1-1.3B

```bash
DATASET_FILE=/data/my_train.json \
VALIDATION_DATASET_FILE=/data/my_val.json \
MODEL_PATH=weights/Wan2.1-Fun-V1.1-1.3B-InP \
DA3_PRETRAINED=weights/da3 \
GEO=1 OUT=runs/my_wan_geo \
bash wan/train.sh
```

- **Geometry branch.** A hook on block `GEO_HOOK_LAYER` (15 of 30) feeds the Geo-DiT
  (`GEO_DIM` 1536, `GEO_HEADS` 12, `GEO_DEPTH` 4). DA3 depth is computed per clip, encoded by the
  Wan VAE into a 16-ch latent and noised with the RGB noise; `MASK_RATIO` (0.7) of the tokens are
  masked and `loss += LAMBDA_GEO (0.5) · ramp · mse`.
- **Starting point.** Optional `INIT_TRANSFORMER=<OUT>/checkpoint-N/transformer/diffusion_pytorch_model.safetensors`
  loads saved weights with a fresh optimizer.
- **Resume.** Same `OUT`; the newest `OUT/checkpoint-N/` (accelerate state + `geo_dit.pt`) is
  picked up automatically.
- `LEARNING_RATE` 1e-5, `NUM_FRAMES` 49, `SAMPLE_SIZE` 512.

## 4. Inference

### CogVideoX

```bash
export PYTHONPATH=$PWD/cogvideox:$PWD/depth/src
python cogvideox/inference.py \
  --manifest examples/val_manifest.json --data_root /data \
  --weights_path weights/checkpoint-2 \
  --base_model weights/cogvideox \
  --output_dir outputs/ckpt2 --side_by_side
```

- `--weights_path` also accepts `runs/x/ckpt/step_010000` or `runs/x/transformer`.
- The first frame of each manifest `video` (or the image itself) is the condition; the prompt is
  `prompt + robot`.
- Output: `outputs/ckpt2/<idx>_<clip>.mp4` (49 frames, 480 × 640, 8 fps); with
  `--side_by_side` also `<idx>_<clip>_gt_pred.mp4` (left GT, right prediction).
- Options: `--num_inference_steps 50`, `--guidance_scale 6.0`, `--num_videos_per_sample N`,
  `--memory_efficient` (VAE tiling + CPU offload for 24–40 GB GPUs).

### Wan2.1-1.3B

```bash
cd wan
MODEL_NAME=../weights/Wan2.1-Fun-V1.1-1.3B-InP \
TRANSFORMER_PATH=../runs/my_wan_geo/checkpoint-5000/transformer/diffusion_pytorch_model.safetensors \
IMAGE=first_frame.png PROMPT="pick up the red cube Franka robot arm" \
HEIGHT=512 WIDTH=512 VIDEO_LENGTH=49 SAVE_PATH=../outputs/wan \
python examples/wan2.1_fun/predict_i2v.py
```

Leave `TRANSFORMER_PATH` unset to run the base model. `geo_dit.pt` is not needed for inference.

## Wan status

The Wan line has **not produced a usable model yet**. With the old default learning rate (1e-4),
every checkpoint generated noise, already visible at step ~360, while the losses looked normal;
the base model through the same pipeline was fine, so fine-tuning itself corrupted the weights.
The suspected cause is the learning rate (the working CogVideoX runs use 1e-5). This repository
defaults to 1e-5, but the fix has **not been verified** — check the first validation videos
before committing GPU time.

## Known caveats

- Clips shorter than `FRAME_LENGTH` are padded with a frozen last frame, which teaches the
  model to stop moving; filter manifests by frame count where possible.
- The CogVideoX geometry target comes from frozen, randomly initialised 1×1 projections of DA3
  features, and `LAMBDA_GEO` is small (0.03). `GEO=0` (RGB-only) is a strong baseline.
- Validation and inference center-crop to the target aspect ratio; keep train- and test-time
  aspect handling the same.
- On nodes with a low process limit, OpenCV may fail to open h264 videos; the loader then falls
  back to decord.

## Citation

```bibtex
@inproceedings{zhou2026gem,
  title={Gem-4d: Geometry-enhanced video world models for robot manipulation},
  author={Zhou, Kaichen and Chen, Yuzhen and Zhan, Fangneng and Hua, Hang and Chen, Grace and Chang, Xinhai and Qu, Ao and Du, Yilun and Liu, Zhuang and Liang, Paul Pu and others},
  booktitle={European Conference on Computer Vision},
  pages={130--147},
  year={2026},
  organization={Springer}
}
```

## Acknowledgements

Built on [CogVideoX](https://github.com/THUDM/CogVideo), [TesserAct](https://github.com/UMass-Embodied-AGI/TesserAct),
[VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun) and
[Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3).
