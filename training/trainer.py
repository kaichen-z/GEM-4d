"""GEM-4D trainer (one code path for every backbone).

    L = L_vid + alpha * min(1, step / geo_warmup) * L_geo
"""
import logging
import math
import os
import random
from datetime import timedelta
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader

from gem4d import GEM4D, build_backbone
from gem4d.geometry.teacher import DA3Teacher
from gem4d.utils.video_io import first_frame, save_mp4, side_by_side, to_uint8

from .data.manifest import build_prompt, load_manifest
from .data.video_dataset import ManifestVideoDataset, center_crop_resize, collate, read_video
from .train_utils import checkpoint as ckpt
from .train_utils.config import dump_config

logger = logging.getLogger("gem4d")


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        out = cfg.experiment.output_dir
        self.acc = Accelerator(
            gradient_accumulation_steps=cfg.optim.grad_accum,
            mixed_precision=cfg.train.mixed_precision,
            log_with=cfg.logging.report_to,
            project_config=ProjectConfiguration(project_dir=out, logging_dir=os.path.join(out, "logs")),
            kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=1800))])
        logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
        set_seed(cfg.train.seed + self.acc.process_index)
        self.dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}[cfg.train.mixed_precision]
        if self.acc.is_main_process:
            os.makedirs(out, exist_ok=True)
            dump_config(cfg, os.path.join(out, "config.yaml"))

        # ---- model ----
        b, g = cfg.backbone, cfg.geometry
        self.bb = build_backbone(b.name, base_model=b.base_model, init_weights=b.init_weights)
        self.model = GEM4D(self.bb, geometry=g.enabled, hook_layer=g.hook_layer, geo_depth=g.depth,
                           mask_ratio=g.mask_ratio, geo_scale=g.scale, use_eps_hat=g.use_eps_hat,
                           x0_target=g.x0_target, coupled_noise=g.coupled_noise)
        if g.enabled and g.init_geometry:
            ckpt.load_geometry(self.model, g.init_geometry)
        if cfg.train.gradient_checkpointing:
            self.bb.transformer.enable_gradient_checkpointing()
        self.model.to(self.acc.device, dtype=self.dtype)
        self.bb.to(self.acc.device, self.dtype)
        self.teacher = (DA3Teacher(g.teacher, feat_layer=g.feat_layer).to(self.acc.device)
                        if g.enabled else None)

        # ---- data ----
        d = cfg.data
        ds_kw = dict(data_root=d.data_root, num_frames=d.num_frames, height=d.height, width=d.width,
                     min_frames=d.min_frames)
        self.train_ds = ManifestVideoDataset(d.train_manifest, caption_dropout=d.caption_dropout, **ds_kw)
        self.val_ds = ManifestVideoDataset(d.val_manifest, caption_dropout=0.0, **ds_kw) if d.val_manifest else None
        dl = DataLoader(self.train_ds, batch_size=cfg.train.batch_size, shuffle=True, collate_fn=collate,
                        num_workers=d.num_workers, drop_last=True, persistent_workers=d.num_workers > 0)

        # ---- optimisation ----
        o = cfg.optim
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=o.lr, betas=tuple(o.betas), weight_decay=o.weight_decay, eps=o.eps)
        n = self.acc.num_processes
        sched = get_scheduler(o.scheduler, optimizer=opt, num_warmup_steps=o.warmup_steps * n,
                              num_training_steps=cfg.train.max_steps * n)
        self.model, self.opt, self.dl, self.sched = self.acc.prepare(self.model, opt, dl, sched)
        self.step = ckpt.resume(self.acc, out)
        if self.acc.is_main_process:
            self.acc.init_trackers(cfg.experiment.name, config={"lr": o.lr, "backbone": b.name,
                                                                "geometry": g.enabled})
            n_train = sum(p.numel() for p in params)
            logger.info(f"backbone={b.name} geometry={g.enabled} trainable={n_train / 1e9:.2f}B "
                        f"clips={len(self.train_ds)} start_step={self.step}")

    # ------------------------------------------------------------------
    def encode(self, batch, image_dropout: float):
        bb, video = self.bb, batch["video"].to(self.acc.device, non_blocking=True)
        ctx = bb.encode_text(batch["prompt"])
        x0 = bb.encode_video(video)
        cond = bb.encode_condition(video, x0, drop=random.random() < image_dropout)
        g0 = None
        if self.teacher is not None:
            feats = self.teacher((video.float() + 1) / 2)                   # DA3 wants [0, 1]
            g0 = self.acc.unwrap_model(self.model).geometry_latent(feats, x0.shape[1:])
        return x0, cond, ctx, g0

    def geo_weight(self):
        g = self.cfg.geometry
        return g.alpha * min(1.0, self.step / max(1, g.warmup)) if g.enabled else 0.0

    def train(self):
        cfg, acc = self.cfg, self.acc
        max_steps = cfg.train.max_steps
        while self.step < max_steps:
            for batch in self.dl:
                with acc.accumulate(self.model):
                    x0, cond, ctx, g0 = self.encode(batch, cfg.data.image_dropout)
                    t = self.bb.sample_timesteps(x0.shape[0], acc.device)
                    l_vid, l_geo = self.model(x0, cond, ctx, t, g0)
                    w = self.geo_weight()
                    loss = l_vid + w * l_geo
                    acc.backward(loss)
                    gnorm = acc.clip_grad_norm_(self.model.parameters(), cfg.optim.max_grad_norm) \
                        if acc.sync_gradients else None
                    self.opt.step()
                    self.sched.step()
                    self.opt.zero_grad(set_to_none=True)
                if not acc.sync_gradients:
                    continue
                self.step += 1
                logs = {"loss": loss.item(), "video_loss": l_vid.item(), "geo_loss": l_geo.item(),
                        "geo_weight": w, "lr": self.sched.get_last_lr()[0]}
                if gnorm is not None:
                    logs["grad_norm"] = float(gnorm)
                acc.log(logs, step=self.step)
                if acc.is_main_process and self.step % cfg.logging.log_every == 0:
                    logger.info(f"step {self.step} " + " ".join(f"{k}={v:.4g}" for k, v in logs.items()))
                if self.step % cfg.checkpoint.every == 0 or self.step == max_steps:
                    ckpt.save(acc, self.model, cfg.experiment.output_dir, self.step,
                              keep_last=cfg.checkpoint.keep_last, keep_every=cfg.checkpoint.keep_every,
                              save_state=cfg.checkpoint.save_state)
                v = cfg.validation
                if self.val_ds is not None and (self.step % v.every == 0 or self.step == 1):
                    self.validate()
                if self.step >= max_steps:
                    break
        acc.end_training()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def validate(self):
        cfg, acc, v = self.cfg, self.acc, self.cfg.validation
        model = acc.unwrap_model(self.model)
        model.eval()
        # (1) loss at fixed timesteps, every rank on the same clips -> averaged
        n = min(v.loss_batches, len(self.val_ds))
        tot_v = tot_g = 0.0
        for i in range(n):
            batch = collate([self.val_ds[i]])
            x0, cond, ctx, g0 = self.encode(batch, image_dropout=0.0)
            t = self.bb.fixed_timesteps((i + 0.5) / n, 1, acc.device)
            l_vid, l_geo = self.model(x0, cond, ctx, t, g0)
            tot_v += l_vid.item(); tot_g += l_geo.item()
        logs = {"val/video_loss": tot_v / max(1, n), "val/geo_loss": tot_g / max(1, n)}
        acc.log(logs, step=self.step)
        # (2) sample videos on the main process: [GT | prediction]
        if acc.is_main_process and v.num_videos > 0:
            logger.info(f"step {self.step} " + " ".join(f"{k}={x:.4g}" for k, x in logs.items()))
            if hasattr(model, "capture"):
                model.capture = False
            pipe = self.bb.pipeline(model.transformer)
            d = cfg.data
            for i, entry in enumerate(load_manifest(d.val_manifest, d.data_root)[: v.num_videos]):
                try:
                    frames = pipe(image=first_frame(entry["video"], (d.height, d.width)),
                                  prompt=build_prompt(entry), height=d.height, width=d.width,
                                  num_frames=d.num_frames, num_inference_steps=v.num_inference_steps,
                                  guidance_scale=v.guidance_scale,
                                  generator=torch.Generator(device=acc.device).manual_seed(cfg.train.seed),
                                  output_type="np").frames[0]
                    gt = [center_crop_resize([f], (d.height, d.width))[0] for f in read_video_safe(entry)]
                    out = os.path.join(cfg.experiment.output_dir, "vis", f"step_{self.step:06d}", f"{i:02d}.mp4")
                    save_mp4(side_by_side(gt, to_uint8(frames)) if gt else frames, out, fps=v.fps)
                except Exception as e:  # sampling must never kill a long run
                    logger.warning(f"validation video {i} failed: {e}")
            if hasattr(model, "capture"):
                model.capture = True
        model.train()
        acc.wait_for_everyone()


def read_video_safe(entry):
    try:
        return read_video(Path(entry["video"]))
    except Exception:
        return []
