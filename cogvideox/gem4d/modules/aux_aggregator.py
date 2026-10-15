"""Task-19: per-mode aggregation of RGB-DiT middle features -> one token feature
fed to the existing depth head (forward_depth). Trainable; dropped at inference
(forward() / RGB sampling never calls it).

Modes (all operate in token space [B, N, dim]):
  A  : Y-branch          -> use a single middle tap h_{L*}.
  B  : per-layer 0-init  -> final + sum_l Z_l(h_l), Z_l zero-init Linear (starts == final).
  C  : cross-attention   -> position-wise attention of `final` over the stacked taps.
  D  : shared / LoRA-ish -> final + mean_l Adapter(h_l), one shared low-rank adapter.
"""
import torch
import torch.nn as nn


class AuxAggregator(nn.Module):
    def __init__(self, mode: str, dim: int, n_taps: int, a_index: int = -1, rank: int = 64):
        super().__init__()
        self.mode = mode.upper()
        self.dim = dim
        self.n_taps = n_taps
        self.a_index = a_index  # which tap is L* for mode A
        if self.mode == 'B':
            self.zs = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_taps)])
            for z in self.zs:  # zero-init -> no-op at start
                nn.init.zeros_(z.weight); nn.init.zeros_(z.bias)
        elif self.mode == 'C':
            self.q = nn.Linear(dim, dim)
            self.k = nn.Linear(dim, dim)
            self.scale = dim ** -0.5
            self.out = nn.Linear(dim, dim); nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        elif self.mode == 'D':
            self.down = nn.Linear(dim, rank); self.up = nn.Linear(rank, dim)
            nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)  # 0-init -> no-op at start

    def forward(self, taps, final):
        """taps: list of [B, N, dim] middle features; final: [B, N, dim]. -> [B, N, dim]."""
        # Harmonize dtypes: cast inputs to this module's param dtype (or final's if no params).
        try:
            wd = next(self.parameters()).dtype
        except StopIteration:
            wd = final.dtype
        final = final.to(wd)
        taps = [t.to(wd) for t in taps]
        if self.mode == 'A':
            return taps[self.a_index]
        if self.mode == 'B':
            out = final
            for z, h in zip(self.zs, taps):
                out = out + z(h)
            return out
        if self.mode == 'C':
            # position-wise attention of `final` (query) over taps (keys/values), per token.
            S = torch.stack(taps, dim=2)                       # [B,N,T,dim]
            q = self.q(final).unsqueeze(2)                     # [B,N,1,dim]
            k = self.k(S)                                      # [B,N,T,dim]
            attn = (q * k).sum(-1) * self.scale                # [B,N,T]
            w = attn.softmax(dim=-1).unsqueeze(-1)             # [B,N,T,1]
            ctx = (w * S).sum(dim=2)                           # [B,N,dim]
            return final + self.out(ctx)
        if self.mode == 'D':
            agg = sum(self.up(torch.relu(self.down(h))) for h in taps) / max(1, len(taps))
            return final + agg
        raise ValueError(self.mode)
