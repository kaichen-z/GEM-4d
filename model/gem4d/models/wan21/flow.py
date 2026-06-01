"""Flow matching for the Wan2.1-I2V-14B backbone (rectified flow, timesteps t = 1000 * sigma).

Training (shifted-uniform sigma, Gaussian timestep weight):
    sigma = shift * u / (1 + (shift - 1) * u),  u ~ U(0, 1)
    x_t   = (1 - sigma) x0 + sigma eps,  target v* = eps - x0
    w(t)  = (exp(-2 ((t - 500) / 1000)^2) - y_min) / mean over the shifted 1000-step grid

Sampling (discrete schedule 999 -> 0, Euler step in fp32):
    t_i   = round(1000 * shift * s_i / (1 + (shift - 1) s_i) - 1),  s_i = 1, 1 - 1/N, ..., 1/N
"""
import numpy as np
import torch


def shift_sigma(u, shift: float):
    return shift * u / (1.0 + (shift - 1.0) * u)


class FlowMatching:
    def __init__(self, shift: float = 3.0, num_train_timesteps: int = 1000, eps: float = 1e-10):
        if shift <= 0:
            raise ValueError(f"`shift` must be positive, got {shift}")
        self.shift = float(shift)
        self.num_train_timesteps = int(num_train_timesteps)
        self.eps = float(eps)
        steps = self.num_train_timesteps
        u = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)[:-1]
        y = torch.exp(-2.0 * ((shift_sigma(u, self.shift) * steps - steps / 2.0) / steps) ** 2)
        self._y_min = float(y.min())
        self._norm = float((y - self._y_min).mean())

    def sample_t(self, batch_size: int, device, dtype=torch.float32) -> torch.Tensor:
        u = torch.rand((batch_size,), device=device, dtype=torch.float32)
        return (shift_sigma(u, self.shift) * float(self.num_train_timesteps)).to(dtype)

    def weight(self, t: torch.Tensor) -> torch.Tensor:
        steps = float(self.num_train_timesteps)
        y = torch.exp(-2.0 * ((t.float() - steps / 2.0) / steps) ** 2)
        return (y - self._y_min) / (self._norm + self.eps)


def sampling_timesteps(steps: int, shift: float, device) -> torch.Tensor:
    sigma = np.arange(1.0, 0, -1.0 / steps)
    sigma = shift * sigma / (1 + (shift - 1) * sigma)
    return torch.from_numpy(np.round(sigma * 1000 - 1).astype(np.int64)).to(device)


def flow_step(prediction, sample, timestep, next_timestep):
    """Euler step from t to t_next, written through the clean / noise estimates (fp32 latents)."""
    a = 1 - timestep / 1000
    b = timestep / 1000
    clean = (sample - b * prediction) / (a + b)
    noise = (sample + a * prediction) / (a + b)
    return (1 - next_timestep / 1000) * clean + (next_timestep / 1000) * noise
