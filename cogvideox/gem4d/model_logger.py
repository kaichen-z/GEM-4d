import os, torch
from accelerate import Accelerator
import glob
import re

class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x, save_latest_steps=150):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0
        os.makedirs(self.output_path, exist_ok=True)
        self.log_file = os.path.join(self.output_path, "training_log.txt")
        # 功能一：检测上次最后一步的 step number
        self.num_steps = self._detect_last_step()
        self.previous_model = None
        # 功能二：将之前最大step model作为下一次训练的开始
        if self.num_steps > 0:
            self.previous_model = self._detect_model_epoch(self.num_steps)
            print(f"Resume from step {self.num_steps} ================ In ModelLogger ================")
        else: print(f"Start training from step 0 ================ In ModelLogger ================")
        self.save_latest_steps = save_latest_steps
        self.backbone_ckpt = self._detect_origin()

    def _detect_origin(self):
        ckpt_path = os.path.join(self.output_path, "origin_backbone.safetensors")
        return ckpt_path if os.path.isfile(ckpt_path) else None

    def _detect_model_epoch(self, num_steps):
        pattern_1 = os.path.join(self.output_path, f"main_latest_{num_steps}.safetensors")
        pattern_2 = os.path.join(self.output_path, f"projection_latest_{num_steps}.safetensors")
        old_epoch_files = []
        if os.path.exists(pattern_1):
            old_epoch_files = [pattern_1]
        if os.path.exists(pattern_1) and os.path.exists(pattern_2):
            old_epoch_files = [pattern_1, pattern_2]
        else: 
            print("No checkpoint found")
            return None
        return old_epoch_files

    def _detect_last_step(self):
        """检测 output_path 中已保存的最大 step 数"""
        checkpoint_files = glob.glob(os.path.join(self.output_path, "*_latest_*.safetensors"))
        if not checkpoint_files: return 0
        step_numbers = []
        for ckpt_file in checkpoint_files:
            match = re.search(r'latest_(\d+)\.safetensors', os.path.basename(ckpt_file))
            if match: step_numbers.append(int(match.group(1)))
        return max(step_numbers) if step_numbers else 0

    def step_count(self):
        self.num_steps += 1

    def on_step_end(self, accelerator, model, save_steps=None, name="main"):
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"{name}_step_{self.num_steps}.safetensors")
        if self.num_steps % self.save_latest_steps == 0:  # 可以调整频率
            self._remove_old_latest(name)
            self.save_model(accelerator, model, f"{name}_latest_{self.num_steps}.safetensors")

    def _remove_old_latest(self, name):
        pattern = os.path.join(self.output_path, f"{name}_latest_*.safetensors")
        old_latest_files = glob.glob(pattern)
        for old_file in old_latest_files:
            try:
                os.remove(old_file)
                print(f"[ModelLogger] Removed old latest checkpoint: {os.path.basename(old_file)}")
            except Exception as e:
                print(f"[ModelLogger] Warning: Failed to remove {old_file}: {e}")

    def log_losses(self, epoch_id, losses: dict = None):
        """记录任意数量的 loss 字典，例如 {"train_loss": 0.1, "val_loss": 0.2, "proj_loss": 0.05}"""
        with open(self.log_file, 'a') as f:
            log_line = f"Epoch: {epoch_id}, Step: {self.num_steps}"
            if losses is not None:
                for key, value in losses.items():
                    if isinstance(value, (float, int)):
                        log_line += f", {key}: {value:.4f}"
                    else: log_line += f", {key}: {value}"
            log_line += "\n"
            f.write(log_line)

    def on_training_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None, name="main"):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"{name}_step-{self.num_steps}.safetensors")

    def on_epoch_end(self, accelerator: Accelerator, model: torch.nn.Module, epoch_id, name="main"):
        self.save_model(accelerator, model, f"{name}_epoch-{epoch_id}.safetensors")

    def save_model(self, accelerator: Accelerator, model: torch.nn.Module, file_name):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)