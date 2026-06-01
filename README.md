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
    <a href="https://huggingface.co/datasets/zhouk777/gem4d-weight"><img src="https://img.shields.io/badge/Weights-HuggingFace-C9E4CA?labelColor=EAF5EB" alt="Weights" /></a>
  </p>

</div>

![GEM-4D vs. TesserAct: from one observation and an instruction, GEM-4D generates a rollout whose geometry stays consistent enough to act on.](assets/teaser.jpg)

**GEM-4D** distils dense 4D geometry from a frozen geometry foundation model (Depth Anything 3) into a
video world model during training. A small Geometry DiT reads one middle block of the video DiT and
has to predict the noised geometry latent; its loss can only fall if that block encodes depth, camera
motion and scene flow. After training the Geometry DiT is dropped: the deployed model is a plain
image-to-video generator with the backbone's size and speed. GEM-4D improves video prediction and
geometric consistency on DROID and RLBench, and real-robot success from **61 % to 81 %**.

## Installation

```bash
conda create -n gem4d python=3.10 -y && conda activate gem4d
pip install -r requirements.txt
```

### Weights

```bash
python tools/download_weights.py wan2.2 da3
```

### Data

See [the example manifest](examples/train_manifest.json) for the video and instruction format.

## 1. Training GEM-4D Wan-5B

```bash
# 1. GEM-4D Wan-5B (8 GPUs)
bash training_bash/train.sh training/config/wan5b_gem4d.yaml \
    data.train_manifest=/data/train.json data.val_manifest=/data/val.json
```

## License and attribution

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
