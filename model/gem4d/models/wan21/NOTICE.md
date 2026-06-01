# Third-party notice: Wan2.1 components

The files in this folder implement the Wan2.1-I2V-14B model (DiT, VAE, CLIP image encoder,
umT5 text encoder) in the parameter layout of the original Wan2.1 checkpoints.

- **Wan2.1** — Copyright Alibaba Wan Team, Apache License 2.0.
  <https://github.com/Wan-Video/Wan2.1> ([LICENSE](https://github.com/Wan-Video/Wan2.1/blob/main/LICENSE.txt)).
- **DiffSynth-Studio** — Copyright ModelScope, Apache License 2.0.
  <https://github.com/modelscope/DiffSynth-Studio>.
- Packaging of these components: MIT License, Copyright 2026 The FastWAM Authors. `vae.py`, `clip.py` and `text_encoder.py` are unchanged apart
  from their header; `dit.py` keeps the same parameters and arithmetic, restricted to bidirectional
  video attention, with gradient checkpointing and checkpoint export added; `flow.py` re-implements the
  same flow-matching schedule; `loader.py` is new.

Full license texts: [LICENSE-APACHE-2.0.txt](LICENSE-APACHE-2.0.txt), [LICENSE-MIT.txt](LICENSE-MIT.txt).
Model weights are not covered by these licenses; they keep the terms of their own releases.
