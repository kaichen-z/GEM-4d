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
    <br/><sup>*</sup>Equal contribution &nbsp;&nbsp; <sup>†</sup>Joint supervision
  </p>

  <p>
    <a href="https://arxiv.org/abs/2605.22882"><img src="https://img.shields.io/badge/arXiv-2605.22882-F4B8B0?labelColor=FDE6E2" alt="arXiv" /></a>
    &nbsp;
    <a href="https://gem-4d.github.io/"><img src="https://img.shields.io/badge/Project-Page-F6D9A0?labelColor=FFF3DC" alt="Project Page" /></a>
    &nbsp;
    <a href="https://huggingface.co/depth-anything/DA3-LARGE-1.1"><img src="https://img.shields.io/badge/DA3%20Weights-HuggingFace-C9E4CA?labelColor=EAF5EB" alt="DA3 Weights" /></a>
  </p>

</div>

![GEM-4D vs. TesserAct: from one observation and an instruction, GEM-4D generates a rollout whose geometry stays consistent enough to act on.](assets/teaser.jpg)

**GEM-4D** distils dense 4D geometry from a frozen geometry foundation model (Depth Anything 3) into a
video world model during training. A small Geometry DiT reads one middle block of the video DiT and
has to predict the noised geometry latent; its loss can only fall if that block encodes depth, camera
motion and scene flow. After training the Geometry DiT is dropped: the deployed model is a plain
image-to-video generator with the backbone's size and speed. GEM-4D improves video prediction and
geometric consistency on DROID and RLBench, and real-robot success from **61 % to 81 %**.

## Contents

| | What | Where |
|---|---|---|
| 1 | Train GEM-4D on **Wan2.2-TI2V-5B** | [Training GEM-4D](#1-2-training-gem-4d) · `training/config/wan5b_gem4d.yaml` |
| 2 | Train GEM-4D on **Wan2.1-I2V-14B** | [Training GEM-4D](#1-2-training-gem-4d) · `training/config/wan14b_gem4d.yaml` |
| 3 | Inference with **GEM-4D-Wan14B** | [Inference](#3-inference-with-gem-4d-wan14b) · `inference/infer.py` |
| 4 | **GEM-4D-Wan14B** weights | [Checkpoint](#4-gem-4d-wan14b-checkpoint) · `tools/download_weights.py` |
| 5 | **Depth & scene flow** of generated videos | [Depth & scene flow](#5-depth--scene-flow-4d-process) · `--process4d`, `inference/process4d.py` |
| 6 | **LoRA fine-tuning** of GEM-4D-Wan14B on your data (no geometry branch) | [LoRA fine-tuning](#6-lora-fine-tuning-of-gem-4d-wan14b) · `training/config/wan14b_gem4d_lora.yaml` |

Also: [method](#method) · [results](#results) · [code layout](#code-layout) · [citation](#citation)

## Installation

```bash
conda create -n gem4d python=3.10 -y && conda activate gem4d
pip install -r requirements.txt
```

### Weights

```bash
python tools/download_weights.py wan2.1 gem-4d-wan14b da3   # 3, 5, 6: GEM-4D-Wan14B inference, depth & flow, LoRA
python tools/download_weights.py wan2.1 da3                # 2: train GEM-4D Wan-14B
python tools/download_weights.py wan2.2 da3                # 1: train GEM-4D Wan-5B
```

| Folder in `weights/` | Source | Used as |
|---|---|---|
| `Wan2.1-I2V-14B-480P/` | [`Wan-AI/Wan2.1-I2V-14B-480P`](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P) | Wan-14B base: DiT, VAE, CLIP, umT5, tokenizer |
| `Wan2.2-TI2V-5B-Diffusers/` | [`Wan-AI/Wan2.2-TI2V-5B-Diffusers`](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers) | Wan-5B base |
| `gem-4d-wan14b/` | [`kaichen-z/Gem-4D`](https://huggingface.co/datasets/kaichen-z/Gem-4D) | GEM-4D-Wan14B DiT ([4](#4-gem-4d-wan14b-checkpoint)) |
| `da3/` | [Official DA3-LARGE-1.1](https://huggingface.co/depth-anything/DA3-LARGE-1.1) | geometry teacher (training), depth (4D-process) |

CLIPSeg (`CIDAS/clipseg-rd64-refined`, foreground gate of the 4D-process) and RAFT-small (torchvision)
download on first use.

### Data

Training and manifest inference read a **manifest** (JSON / JSONL list); no dataset is predefined:

```json
[
  {"video": "rlbench/close_drawer/ep_0001/front.mp4", "prompt": "close the middle drawer", "robot": "Franka robot arm"},
  {"video": "droid/ep_0042.mp4", "prompt": "put the marker in the cup", "start": 35, "end": 212},
  {"video": "my_lab/frames_0042", "prompt": "push the red block to the left"}
]
```

| Field | Meaning |
|---|---|
| `video` | `.mp4` / `.webm`, or a folder of frames `0.png, 1.png, …` (an image also works for inference). Relative paths resolve against `data.data_root`, else the manifest's folder |
| `prompt` | the instruction |
| `robot` | optional embodiment text appended to the prompt; keep it identical at train and test time |
| `start`, `end` | optional frame range `[start, end)` of one event inside a longer video |

`[prompt, video]` / `[prompt, video, robot]` lists work too. Clips are center-cropped and resized to
`data.height × data.width` and brought to 49 frames: `frame_sampling: uniform` (Wan-5B) subsamples long
clips and pads short ones with the last frame; `frame_sampling: event` (Wan-14B) resamples each event to
exactly 49 frames. Frame 0 is the condition. No depth files are needed: geometry comes from DA3 on the fly.

## 1, 2. Training GEM-4D

```bash
# 1. GEM-4D Wan-5B (8 GPUs)
bash training_bash/train.sh training/config/wan5b_gem4d.yaml \
    data.train_manifest=/data/train.json data.val_manifest=/data/val.json

# 2. GEM-4D Wan-14B (8 GPUs with >= 141 GB; on 8 x 80 GB keep the optimizer in CPU memory)
bash training_bash/train.sh training/config/wan14b_gem4d.yaml \
    data.train_manifest=/data/events.json data.val_manifest=/data/val.json \
    train.accelerate_config=training/config/accelerate_zero2_offload.yaml
```

| Path | Backbone | Latent | Clips | Objective |
|---|---|---|---|---|
| Wan-5B | [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers), 30 blocks (diffusers) | 48 × F/4 × H/16 × W/16 | 49 × 480 × 640, uniform | TI2V first-frame conditioning, shift-5 rectified flow |
| Wan-14B | [Wan2.1-I2V-14B-480P](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P), 40 blocks (original Wan2.1 layout) | 16 × F/4 × H/8 × W/8 | 49 × 240 × 416, event | CLIP + latent first-frame conditioning, shift-3 flow matching |

Main settings (any key of `training/config/default.yaml` can be overridden as `key.sub=value`):

| Key | Wan-5B | Wan-14B | Meaning |
|---|---|---|---|
| `geometry.alpha` | 0.03 | 0.03 | weight α of L_geo (linear ramp over `geometry.warmup` = 100 steps) |
| `geometry.mask_ratio` | 0.5 | 0.5 | fraction of geometry tokens masked |
| `geometry.hook_layer` | 15 | 20 | block whose output `m_t` feeds the Geometry DiT |
| `geometry.depth` | 4 | 4 | Geometry DiT blocks |
| `optim.lr` | 1e-5 | 5e-6 | constant after 100 warm-up steps, AdamW (0.9, 0.95), wd 1e-3 |
| `train.max_steps` | 10000 | 10000 | batch 1 per GPU, bf16, gradient checkpointing |

`*_rgb.yaml` are the RGB-only full fine-tuning baselines (`geometry.enabled: false`).
`backbone.init_weights=<step_N>` continues from an export (plus `geometry.init_geometry=<step_N>/geometry.safetensors`).

**Outputs** (`experiment.output_dir`): `checkpoints/step_N/transformer/` (the video DiT: diffusers format
for Wan-5B, original Wan2.1 layout for Wan-14B), `checkpoints/step_N/geometry.safetensors` (Geometry DiT,
training only), `checkpoints/state/` (resumable state), `vis/step_N/*.mp4` (validation, GT | prediction),
`logs/` (TensorBoard). **Resume:** re-run the same command.

## 3. Inference with GEM-4D-Wan14B

```bash
# one photo + instruction (--compose: photo -> the 416 x 240 robot composite the model was trained on)
bash training_bash/infer.sh --backbone wan2.1-i2v-14b --base_model weights/Wan2.1-I2V-14B-480P \
    --weights weights/gem-4d-wan14b --image photo.jpg --compose --output_dir outputs/wan14b \
    --prompt "The white Franka robot arm performs the task: close the drawer. This is a composed robot video with the main third-person view on the left."

# every entry of a manifest, with [GT | prediction] videos
bash training_bash/infer.sh --backbone wan2.1-i2v-14b --base_model weights/Wan2.1-I2V-14B-480P \
    --weights weights/gem-4d-wan14b --manifest examples/val_manifest.json --output_dir outputs/wan14b --side_by_side

# with your own LoRA adapter (6) on top, merged at load time
bash training_bash/infer.sh ... --weights weights/gem-4d-wan14b --lora runs/wan14b_gem4d_lora/checkpoints/step_010000
```

`--weights` also takes any training export (`runs/<exp>/checkpoints/step_N`; for Wan-5B use
`--backbone wan2.2-ti2v-5b --base_model weights/Wan2.2-TI2V-5B-Diffusers`). Output: `<idx>_<clip>.mp4`
(49 frames), `<idx>_<clip>_gt_pred.mp4` with `--side_by_side`, `_<k>` suffixes with `--num_samples k`.

| Backbone | Steps | CFG | Size | fps | Seed | GPU memory |
|---|---|---|---|---|---|---|
| `wan2.1-i2v-14b` | 30 | 4.0 | 240 × 416 | 8 | 3402 | ~48 GB (shift 3, empty negative prompt) |
| `wan2.2-ti2v-5b` | 50 | 5.0 | 480 × 640 | 16 | 42 | `--cpu_offload` below 40 GB |

Override with `--steps`, `--guidance_scale`, `--shift`, `--height`, `--width`, `--fps`, `--seed`.

**Prompts.** Use the format GEM-4D-Wan14B was trained on:
`The <robot> performs the task: <instruction>. This is a composed robot video with the main third-person view on the left.`
with `<robot>` = white Franka robot arm (RLBench), Franka robot arm (DROID), black WidowX robot arm (Bridge),
Google mobile manipulator robot arm (RT-1). Training videos are 416 × 240 composites with the 4:3 main
view on the left (`x < 302`) and black on the right; `--compose` puts a plain photo into that layout.

## 4. GEM-4D-Wan14B checkpoint

**GEM-4D-Wan14B** is the released Wan-14B video model of GEM-4D: a Wan2.1-I2V-14B DiT for image +
instruction → 49-frame robot video (240 × 416), trained on RLBench / Bridge / RT-1 / DROID
(0.40 / 0.25 / 0.25 / 0.10). It has the layout of a [Wan-14B training export](#1-2-training-gem-4d)
(`runs/wan14b_gem4d/checkpoints/step_N/`) and loads the same way:

```
gem-4d-wan14b/
  transformer/config.json
  transformer/diffusion_pytorch_model-0000k-of-0000n.safetensors   bf16, original Wan2.1 parameter names
  transformer/diffusion_pytorch_model.safetensors.index.json
```

| | |
|---|---|
| Base | [Wan2.1-I2V-14B-480P](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P) (Apache-2.0) |
| Download | `python tools/download_weights.py wan2.1 gem-4d-wan14b` ([`kaichen-z/Gem-4D`](https://huggingface.co/datasets/kaichen-z/Gem-4D)) |
| Inference | `--base_model weights/Wan2.1-I2V-14B-480P --weights weights/gem-4d-wan14b` (VAE, CLIP, umT5 and tokenizer come from the base folder) |
| Further training | `backbone.init_weights=weights/gem-4d-wan14b` with `wan14b_gem4d.yaml` / `wan14b_rgb.yaml`, or [6](#6-lora-fine-tuning-of-gem-4d-wan14b) |

## 5. Depth & scene flow (4D-process)

`--process4d` reads 4D structure out of every generated video (inference only, nothing is trained):

```bash
# during generation (composite robot videos: process the main view, x < 302 of 416)
bash training_bash/infer.sh --backbone wan2.1-i2v-14b --base_model weights/Wan2.1-I2V-14B-480P \
    --weights weights/gem-4d-wan14b --image first_frame.png --prompt "..." --output_dir outputs/wan14b \
    --process4d --process4d_da3 weights/da3 --process4d_crop 0,0,302,240

# on existing videos, plus one page with all of them as tabs
python inference/process4d.py --videos outputs/wan14b/*.mp4 --da3 weights/da3 --crop 0,0,302,240 \
    --output_dir outputs/wan14b_4d --viewer_all outputs/wan14b_4d/all_4d.html
```

Per video:

| File | Content |
|---|---|
| `<clip>_4d.mp4` | frame \| DA3 depth with white scene-flow arrows on the moving robot / object only |
| `<clip>_4d.html` | interactive point-cloud viewer ([DynPoint](https://dynpoint.github.io/) style): RGB / depth colours, scene-flow segments of the dynamic points, camera frustum, frame slider |
| `<clip>_4d.npz` | depth, confidence, cameras, optical flow, scene flow, dynamic / foreground masks, camera shake |
| `<clip>_4d.json` | valid / dynamic / foreground fractions, flow statistics |

How it stays clean:

1. **Depth and cameras.** Depth Anything 3 predicts depth, intrinsics and poses for the whole clip;
   pixels are lifted to world points `X_t`.
2. **Scene flow.** [RAFT-small](https://pytorch.org/vision/stable/models/raft.html) optical flow `t → t+3`
   (and back) matches pixels; `S_t(p) = A_t(X_{t+3}(p + flow(p))) − X_t(p)`, where `A_t` is a similarity
   (Umeyama) fitted on the background, so DA3's frame-to-frame scale / pose drift cancels.
3. **Foreground only.** CLIPSeg masks ("a robot arm", "a white robot arm", "a robot gripper", "a human
   hand", "an object held by a robot gripper"); background scene flow is set to zero.
4. **No camera shake.** The background's median optical flow is subtracted; a foreground pixel moves only
   if the rest exceeds `--min_flow_px` (1.5 px). Matches failing the forward-backward check are dropped.
5. **Arrows, not colours.** `<clip>_4d.mp4` draws one anti-aliased arrow per 30 px cell of moving
   foreground (flow averaged over 3 frames, length relative to the clip's motion) over plain depth
   (`gem4d.process4d.flow_vis.FlowPanel`).

Options: `--process4d_gap` (3), `--process4d_process_res` (DA3 resolution, 504), `--process4d_seg none`
(no foreground gate). Scene flow is in DA3's non-metric units.

### Figures: teaser panel and demo video

```bash
# generated frames, 3D point clouds and an end-effector box for one generated video
python inference/teaser.py --video outputs/teaser/0000_photo.mp4 --initial photo.png \
    --crop 0,0,302,240 --da3 weights/da3 --output_dir outputs/teaser/figure
# demo video: one row per example, [input image | generated video | depth & scene flow]
python inference/demo_video.py --crop 0,0,302,240 --output outputs/demo.mp4 \
    --row "DROID|droid.png|outputs/0000_droid.mp4|outputs/0000_droid_4d.npz" \
    --row "Bridge|bridge.png|outputs/0001_bridge.mp4|outputs/0001_bridge_4d.npz"
```

## 6. LoRA fine-tuning of GEM-4D-Wan14B

Adapt GEM-4D-Wan14B to a new robot, camera or task set cheaply: the checkpoint stays frozen and a
rank-128 LoRA adapter is trained with the image-to-video loss only (no Geometry DiT, no DA3 teacher).

```bash
bash training_bash/train.sh training/config/wan14b_gem4d_lora.yaml \
    data.train_manifest=/data/my_robot.json data.val_manifest=/data/my_robot_val.json
```

| Key | Value | |
|---|---|---|
| `backbone.init_weights` | `weights/gem-4d-wan14b` | the frozen starting point |
| `lora.rank`, `lora.alpha` | 128, 128 | `lora.targets: null` = q, k, v, o, ffn.0, ffn.2 (400 layers, 0.6 B parameters) |
| `optim` | lr 1e-4 cosine, AdamW (0.9, 0.999), no weight decay | |
| batch | 2 per GPU × 8 GPUs × 4 accumulation = 64 | global batch of the original fine-tune |
| `lora.init` | an adapter to continue from | |

Each export holds only the adapter, `checkpoints/step_N/lora/{lora.safetensors, lora_config.json}`.
Use it directly (`infer.sh ... --lora <step_N>`, optional `--lora_scale`) or merge it into a stand-alone
checkpoint:

```bash
python tools/merge_lora.py --backbone wan2.1-i2v-14b --base_model weights/Wan2.1-I2V-14B-480P \
    --weights weights/gem-4d-wan14b --lora runs/wan14b_gem4d_lora/checkpoints/step_010000 \
    --output weights/gem-4d-wan14b-myrobot
```

`lora.enabled: true` works with every config (e.g. on the Wan-5B path, targets `to_q, to_k, to_v,
to_out.0, ffn.net.0.proj, ffn.net.2`) and can be combined with the geometry loss.

## Method

```
first frame + instruction ──► video DiT ──────────────────────────► video velocity ──► L_vid
                                │ m_t  (block `hook_layer`, not detached)
                                ▼
 masked, noised geometry latent ──► Geometry DiT ──► geometry velocity ──► L_geo
 g0 = projector(DA3 features)   (same noise as the video latent)

                        L = L_vid + α · L_geo
```

The Geometry DiT sees the scene only through `m_t`, so `α · ∂L_geo/∂m_t` pushes depth, camera motion
and scene flow into the video backbone; the geometry teacher stays frozen. The branch only reads from
`m_t`, so it is removed after training.

| Paper (Sec. 3.2) | Code |
|---|---|
| Video DiT, mid-level feature `m_t` | `model/gem4d/models/{wan5b,wan14b}.py`; forward hook on block `geometry.hook_layer` (`models/gem4d.py`) |
| Frozen geometry model `G`, representation `g_0` | `geometry/teacher.py`: DA3 ViT-L layer 12 → frozen projector to the latent grid |
| Mask mechanism `M` | `heads/geo_dit.py`: `geometry.mask_ratio` of tokens replaced by a learned mask token |
| `v_ψ^geo(M(g_t), t, m_t)` | `GeoDiT`: cross-attention to `m_t` only |
| `L = L_vid + α L_geo` | `models/gem4d.py`, `training/trainer.py` |
| Geometry branch dropped at inference | `inference/infer.py` loads only the video DiT |

Not in this repository: the inverse-dynamics module (video → robot actions) and the VGGT / depth-target ablations.

## Results

4D scene generation (paper Tab. 1; GEM-4D on a CogVideoX-5B backbone, not part of this release).
R = real (DROID), S = simulated (RLBench).

| D | Method | FVD ↓ | SSIM ↑ | PSNR ↑ | AbsRel ↓ | δ1 ↑ | Chamfer ↓ | Tracking ↑ |
|---|---|---|---|---|---|---|---|---|
| R | CogVideoX | 35.56 | 75.91 | 20.18 | 22.33 | 68.32 | 0.2670 | 66.22 |
| R | Wan 2.2-14B | 33.43 | 76.24 | 20.70 | 21.39 | 71.18 | 0.2349 | 68.18 |
| R | TesserAct | 33.28 | 75.66 | 20.08 | 22.07 | 66.80 | 0.2630 | 67.14 |
| R | **GEM-4D** | **31.82** | **82.05** | **21.11** | **20.13** | **78.19** | **0.2001** | **71.23** |
| S | CogVideoX | 40.21 | 75.51 | 20.03 | 15.41 | 70.99 | 0.2913 | 58.32 |
| S | Wan 2.2-14B | 49.20 | 73.01 | 19.87 | 17.81 | 67.07 | 0.1762 | 61.99 |
| S | TesserAct | 41.97 | 76.72 | 19.71 | 16.02 | 69.26 | 0.1813 | 61.15 |
| S | **GEM-4D** | **27.94** | **80.27** | **23.36** | **14.11** | **74.13** | **0.0702** | **68.18** |

More videos and real-robot rollouts: [gem-4d.github.io](https://gem-4d.github.io/).

## Code layout

```
model/gem4d/
  models/wan5b.py, wan14b.py   backbones (Wan2.2-TI2V-5B diffusers; Wan2.1-I2V-14B + 49-frame sampler)
  models/wan21/                Wan2.1 DiT, VAE, CLIP, umT5, loader, flow schedule (Apache-2.0, NOTICE.md)
  models/gem4d.py              video DiT + Geometry DiT + frozen projector
  models/lora.py               LoRA layers: inject, save / load (incl. peft format), merge
  heads/geo_dit.py             Geometry DiT
  geometry/teacher.py          frozen Depth Anything 3 teacher + projector
  process4d/                   depth & scene flow: scene_flow.py (DA3 + RAFT + CLIPSeg), flow_vis.py (arrows),
                               viewer.py (3D viewer), render.py (point clouds), end_effector.py (gripper box)
model/dependency/depth_anything_3/   vendored Depth Anything 3
training/                      trainer.py, launch.py, config/*.yaml, data/ (manifest, dataset), train_utils/
training_bash/train.sh         multi-GPU launcher (accelerate + DeepSpeed ZeRO-2)
training_bash/infer.sh         inference launcher (= python inference/infer.py)
inference/                     infer.py, process4d.py, teaser.py, demo_video.py
tools/                         download_weights.py, merge_lora.py
tests/test_smoke.py            CPU tests with tiny random models: PYTHONPATH=model:. python tests/test_smoke.py
```

## Implementation notes

- **Wan-14B objective.** `x_t = (1 − s) x0 + s ε`, `s = 3u / (1 + 2u)`, target `ε − x0`,
  loss over all latent frames weighted by a Gaussian of the timestep; the DiT input is 16 noisy + 4 mask +
  16 condition channels, plus 257 CLIP tokens. Sampling: 30 Euler steps of the discrete shift-3 schedule
  in fp32, CFG against an empty prompt.
- **Wan-5B objective.** Shift-5 rectified flow, clean image latent at frame 0 with timestep 0 (excluded
  from the loss). Exports are plain diffusers models (`WanImageToVideoPipeline(expand_timesteps=True)`).
- **Geometry latent.** DA3 layer-12 features → frozen 1×1 conv to 256 → temporal average (stride 4) →
  resize to the latent grid → frozen 1×1 conv to the latent channels. Geometry and video share one noise sample.
- **Memory (Wan-14B).** Full fine-tuning plus the Geometry DiT (18.1 B trainable parameters): plain ZeRO-2
  needs ≥ 141 GB GPUs; `accelerate_zero2_offload.yaml` peaks at 61 GB per 80 GB GPU. Offload JIT-builds
  DeepSpeed's `cpu_adam` (torch ≥ 2.10 needs C++20; set `DS_SKIP_CUDA_CHECK=1` on a CUDA minor mismatch).
  `train.sh` raises `ulimit -u` (8 ranks open many threads).

## License and attribution

- `model/gem4d/models/wan21/`: Wan2.1 and DiffSynth-Studio components (Apache-2.0, MIT packaging);
  see `NOTICE.md` and the license files there.
- `model/dependency/depth_anything_3/`: Depth Anything 3 (`LICENSE_depth_anything_3`).
- Weights keep their own terms: Wan2.1 / Wan2.2 (Apache-2.0), DA3 (model card).

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

Built on [diffusers](https://github.com/huggingface/diffusers), [Wan2.1](https://github.com/Wan-Video/Wan2.1),
[Wan2.2](https://github.com/Wan-Video/Wan2.2), [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio),
[Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) and
[TesserAct](https://github.com/UMass-Embodied-AGI/TesserAct).
