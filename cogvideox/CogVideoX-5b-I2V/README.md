---
license: other
license_link: https://huggingface.co/THUDM/CogVideoX-5b-I2V/blob/main/LICENSE
language:
  - en
tags:
  - cogvideox
  - video-generation
  - thudm
  - image-to-video
inference: false
---

# CogVideoX-5B-I2V

<p style="text-align: center;">
  <div align="center">
  <img src=https://github.com/THUDM/CogVideo/raw/main/resources/logo.svg width="50%"/>
  </div>
  <p align="center">
  <a href="https://huggingface.co/THUDM//CogVideoX-5b-I2V/blob/main/README.md">📄 Read in English</a> | 
  <a href="https://huggingface.co/spaces/THUDM/CogVideoX-5B-Space">🤗 Huggingface Space</a> |
  <a href="https://github.com/THUDM/CogVideo">🌐 Github </a> | 
  <a href="https://arxiv.org/pdf/2408.06072">📜 arxiv </a>
</p>
<p align="center">
📍 Visit <a href="https://chatglm.cn/video?fr=osm_cogvideox">Qingying</a> and <a href="https://open.bigmodel.cn/?utm_campaign=open&_channel_track_key=OWTVNma9">API Platform</a> for the commercial version of the video generation model
</p>

## Model Introduction

CogVideoX is an open-source video generation model originating
from [Qingying](https://chatglm.cn/video?fr=osm_cogvideo). The table below presents information related to the video
generation models we offer in this version.

<table  style="border-collapse: collapse; width: 100%;">
  <tr>
    <th style="text-align: center;">Model Name</th>
    <th style="text-align: center;">CogVideoX-2B</th>
    <th style="text-align: center;">CogVideoX-5B</th>
    <th style="text-align: center;">CogVideoX-5B-I2V (This Repository)</th>
  </tr>
  <tr>
    <td style="text-align: center;">Model Description</td>
    <td style="text-align: center;">Entry-level model, balancing compatibility. Low cost for running and secondary development.</td>
    <td style="text-align: center;">Larger model with higher video generation quality and better visual effects.</td>
    <td style="text-align: center;">CogVideoX-5B image-to-video version.</td>
  </tr>
  <tr>
    <td style="text-align: center;">Inference Precision</td>
    <td style="text-align: center;"><b>FP16*(recommended)</b>, BF16, FP32, FP8*, INT8, not supported: INT4</td>
    <td colspan="2" style="text-align: center;"><b>BF16 (recommended)</b>, FP16, FP32, FP8*, INT8, not supported: INT4</td>
  </tr>
  <tr>
    <td style="text-align: center;">Single GPU Memory Usage<br></td>
    <td style="text-align: center;"><a href="https://github.com/THUDM/SwissArmyTransformer">SAT</a> FP16: 18GB <br><b>diffusers FP16: from 4GB* </b><br><b>diffusers INT8 (torchao): from 3.6GB*</b></td>
    <td colspan="2" style="text-align: center;"><a href="https://github.com/THUDM/SwissArmyTransformer">SAT</a> BF16: 26GB <br><b>diffusers BF16: from 5GB* </b><br><b>diffusers INT8 (torchao): from 4.4GB*</b></td>
  </tr>
  <tr>
    <td style="text-align: center;">Multi-GPU Inference Memory Usage</td>
    <td style="text-align: center;"><b>FP16: 10GB* using diffusers</b><br></td>
    <td colspan="2" style="text-align: center;"><b>BF16: 15GB* using diffusers</b><br></td>
  </tr>
  <tr>
    <td style="text-align: center;">Inference Speed<br>(Step = 50, FP/BF16)</td>
    <td style="text-align: center;">Single A100: ~90 seconds<br>Single H100: ~45 seconds</td>
    <td colspan="2" style="text-align: center;">Single A100: ~180 seconds<br>Single H100: ~90 seconds</td>
  </tr>
  <tr>
    <td style="text-align: center;">Fine-tuning Precision</td>
    <td style="text-align: center;"><b>FP16</b></td>
    <td colspan="2" style="text-align: center;"><b>BF16</b></td>
  </tr>
  <tr>
    <td style="text-align: center;">Fine-tuning Memory Usage</td>
    <td style="text-align: center;">47 GB (bs=1, LORA)<br> 61 GB (bs=2, LORA)<br> 62GB (bs=1, SFT)</td>
    <td style="text-align: center;">63 GB (bs=1, LORA)<br> 80 GB (bs=2, LORA)<br> 75GB (bs=1, SFT)<br></td>
    <td style="text-align: center;">78 GB (bs=1, LORA)<br> 75GB (bs=1, SFT, 16GPU)<br></td>
  </tr>
  <tr>
    <td style="text-align: center;">Prompt Language</td>
    <td colspan="3" style="text-align: center;">English*</td>
  </tr>
  <tr>
    <td style="text-align: center;">Maximum Prompt Length</td>
    <td colspan="3" style="text-align: center;">226 Tokens</td>
  </tr>
  <tr>
    <td style="text-align: center;">Video Length</td>
    <td colspan="3" style="text-align: center;">6 Seconds</td>
  </tr>
  <tr>
    <td style="text-align: center;">Frame Rate</td>
    <td colspan="3" style="text-align: center;">8 Frames / Second</td>
  </tr>
  <tr>
    <td style="text-align: center;">Video Resolution</td>
    <td colspan="3" style="text-align: center;">720 x 480, no support for other resolutions (including fine-tuning)</td>
  </tr>
    <tr>
    <td style="text-align: center;">Position Embedding</td>
    <td style="text-align: center;">3d_sincos_pos_embed</td>
    <td style="text-align: center;">3d_rope_pos_embed</td>
    <td style="text-align: center;">3d_rope_pos_embed + learnable_pos_embed</td>
  </tr>
</table>

**Data Explanation**

+ While testing using the diffusers library, all optimizations included in the diffusers library were enabled. This
  scheme has not been tested for actual memory usage on devices outside of **NVIDIA A100 / H100** architectures.
  Generally, this scheme can be adapted to all **NVIDIA Ampere architecture** and above devices. If optimizations are
  disabled, memory consumption will multiply, with peak memory usage being about 3 times the value in the table.
  However, speed will increase by about 3-4 times. You can selectively disable some optimizations, including:

```
pipe.enable_sequential_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()
```

+ For multi-GPU inference, the `enable_sequential_cpu_offload()` optimization needs to be disabled.
+ Using INT8 models will slow down inference, which is done to accommodate lower-memory GPUs while maintaining minimal
  video quality loss, though inference speed will significantly decrease.
+ The CogVideoX-2B model was trained in `FP16` precision, and all CogVideoX-5B models were trained in `BF16` precision.
  We recommend using the precision in which the model was trained for inference.
+ [PytorchAO](https://github.com/pytorch/ao) and [Optimum-quanto](https://github.com/huggingface/optimum-quanto/) can be
  used to quantize the text encoder, transformer, and VAE modules to reduce the memory requirements of CogVideoX. This
  allows the model to run on free T4 Colabs or GPUs with smaller memory! Also, note that TorchAO quantization is fully
  compatible with `torch.compile`, which can significantly improve inference speed. FP8 precision must be used on
  devices with NVIDIA H100 and above, requiring source installation of `torch`, `torchao`, `diffusers`, and `accelerate`
  Python packages. CUDA 12.4 is recommended.
+ The inference speed tests also used the above memory optimization scheme. Without memory optimization, inference speed
  increases by about 10%. Only the `diffusers` version of the model supports quantization.
+ The model only supports English input; other languages can be translated into English for use via large model
  refinement.
+ The memory usage of model fine-tuning is tested in an `8 * H100` environment, and the program automatically
  uses `Zero 2` optimization. If a specific number of GPUs is marked in the table, that number or more GPUs must be used
  for fine-tuning.

**Reminders**

+ Use [SAT](https://github.com/THUDM/SwissArmyTransformer) for inference and fine-tuning SAT version models. Feel free
  to visit our GitHub for more details.

## Getting Started Quickly 🤗

This model supports deployment using the Hugging Face diffusers library. You can follow the steps below to get started.

**We recommend that you visit our [GitHub](https://github.com/THUDM/CogVideo) to check out prompt optimization and
conversion to get a better experience.**

1. Install the required dependencies

```shell
# diffusers>=0.30.3
# transformers>=0.44.2
# accelerate>=0.34.0
# imageio-ffmpeg>=0.5.1
pip install --upgrade transformers accelerate diffusers imageio-ffmpeg 
```

2. Run the code

```python
import torch
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image

prompt = "A little girl is riding a bicycle at high speed. Focused, detailed, realistic."
image = load_image(image="input.jpg")
pipe = CogVideoXImageToVideoPipeline.from_pretrained(
    "THUDM/CogVideoX-5b-I2V",
    torch_dtype=torch.bfloat16
)

pipe.enable_sequential_cpu_offload()
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

video = pipe(
    prompt=prompt,
    image=image,
    num_videos_per_prompt=1,
    num_inference_steps=50,
    num_frames=49,
    guidance_scale=6,
    generator=torch.Generator(device="cuda").manual_seed(42),
).frames[0]

export_to_video(video, "output.mp4", fps=8)
```

## Quantized Inference

[PytorchAO](https://github.com/pytorch/ao) and [Optimum-quanto](https://github.com/huggingface/optimum-quanto/) can be
used to quantize the text encoder, transformer, and VAE modules to reduce CogVideoX's memory requirements. This allows
the model to run on free T4 Colab or GPUs with lower VRAM! Also, note that TorchAO quantization is fully compatible
with `torch.compile`, which can significantly accelerate inference.

```
# To get started, PytorchAO needs to be installed from the GitHub source and PyTorch Nightly.
# Source and nightly installation is only required until the next release.

import torch
from diffusers import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel, CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from transformers import T5EncoderModel
from torchao.quantization import quantize_, int8_weight_only

quantization = int8_weight_only

text_encoder = T5EncoderModel.from_pretrained("THUDM/CogVideoX-5b-I2V", subfolder="text_encoder", torch_dtype=torch.bfloat16)
quantize_(text_encoder, quantization())

transformer = CogVideoXTransformer3DModel.from_pretrained("THUDM/CogVideoX-5b-I2V",subfolder="transformer", torch_dtype=torch.bfloat16)
quantize_(transformer, quantization())

vae = AutoencoderKLCogVideoX.from_pretrained("THUDM/CogVideoX-5b-I2V", subfolder="vae", torch_dtype=torch.bfloat16)
quantize_(vae, quantization())

# Create pipeline and run inference
pipe = CogVideoXImageToVideoPipeline.from_pretrained(
    "THUDM/CogVideoX-5b-I2V",
    text_encoder=text_encoder,
    transformer=transformer,
    vae=vae,
    torch_dtype=torch.bfloat16,
)

pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

prompt = "A little girl is riding a bicycle at high speed. Focused, detailed, realistic."
image = load_image(image="input.jpg")
video = pipe(
    prompt=prompt,
    image=image,
    num_videos_per_prompt=1,
    num_inference_steps=50,
    num_frames=49,
    guidance_scale=6,
    generator=torch.Generator(device="cuda").manual_seed(42),
).frames[0]

export_to_video(video, "output.mp4", fps=8)
```

Additionally, these models can be serialized and stored using PytorchAO in quantized data types to save disk space. You
can find examples and benchmarks at the following links:

- [torchao](https://gist.github.com/a-r-r-o-w/4d9732d17412888c885480c6521a9897)
- [quanto](https://gist.github.com/a-r-r-o-w/31be62828b00a9292821b85c1017effa)

## Further Exploration

Feel free to enter our [GitHub](https://github.com/THUDM/CogVideo), where you'll find:

1. More detailed technical explanations and code.
2. Optimized prompt examples and conversions.
3. Detailed code for model inference and fine-tuning.
4. Project update logs and more interactive opportunities.
5. CogVideoX toolchain to help you better use the model.
6. INT8 model inference code.

## Model License

This model is released under the [CogVideoX LICENSE](LICENSE).

## Citation

```
@article{yang2024cogvideox,
  title={CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer},
  author={Yang, Zhuoyi and Teng, Jiayan and Zheng, Wendi and Ding, Ming and Huang, Shiyu and Xu, Jiazheng and Yang, Yuanming and Hong, Wenyi and Zhang, Xiaohan and Feng, Guanyu and others},
  journal={arXiv preprint arXiv:2408.06072},
  year={2024}
}
```