from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from prism.config import Dims, ModelConfig
from prism.fields import Batch, Forecast, Point
from prism.models.base import Model, models


def pow2_ste(w, terms, e_min=-10, e_max=6):
    """Snap each weight to a sum of `terms` signed powers of two.

    Multiplications by such a weight become shifts and additions, so the
    datapath needs no multiplier at all. The gradient passes through unchanged.
    """
    if terms <= 0:
        return w
    out = torch.zeros_like(w)
    r = w
    for _ in range(terms):
        a = r.abs().clamp_min(1e-12)
        e = torch.round(torch.log2(a)).clamp(e_min, e_max)
        p = torch.pow(2.0, e) * torch.sign(r)
        out = out + p
        r = r - p
    return w + (out - w).detach()


def q_ste(x, bits, dim=None):
    """Symmetric fixed-point quantization with a straight-through gradient."""
    if not bits:
        return x
    qmax = 2 ** (bits - 1) - 1
    a = x.detach().abs()
    s = (a.amax(dim=dim, keepdim=True) if dim is not None else a.amax()).clamp_min(1e-8) / qmax
    q = torch.clamp(torch.round(x / s), -qmax - 1, qmax) * s
    return x + (q - x).detach()


class Quant:
    """Fixed-point emulation of the datapath, shared by every block."""

    def __init__(self, pow2=0, weight_bits=0, act_bits=0):
        self.pow2 = pow2
        self.weight_bits = weight_bits
        self.act_bits = act_bits

    def weight(self, w):
        if self.weight_bits:
            w = q_ste(w, self.weight_bits, dim=0)
        if self.pow2:
            w = pow2_ste(w, self.pow2)
        return w

    def act(self, x):
        return q_ste(x, self.act_bits) if self.act_bits else x


class LocalNorm(nn.Module):
    """Normalization driven by the statistics of the most recent `window` steps.

    `scope='global'` falls back to statistics over the whole input window, which
    is what a plain instance-normalization layer does.
    """

    def __init__(self, window, scope='local', floor=0.1, min_scale=0.05, eps=1e-5):
        super().__init__()
        self.window = window
        self.scope = scope
        self.floor = floor
        self.min_scale = min_scale
        self.eps = eps

    def forward(self, x):
        # x: [B, L, C]
        w = x.shape[1] if self.scope == 'global' else min(self.window, x.shape[1])
        seg = x[:, -w:, :]
        mean = seg.mean(dim=1, keepdim=True)
        std_local = seg.var(dim=1, unbiased=False, keepdim=True).add(self.eps).sqrt()
        std_all = x.var(dim=1, unbiased=False, keepdim=True).add(self.eps).sqrt()
        std = torch.maximum(std_local, torch.maximum(self.floor * std_all,
                                                    torch.full_like(std_local, self.min_scale)))
        return (x - mean) / std, mean, std


class PhaseFold(nn.Module):
    """Fold [B, L, C] into a (phase x period) lattice [B, C, Lp, P].

    Each row of the lattice collects the values sharing one phase offset across
    consecutive periods. Padding is circular so the fold stays aligned with the
    dominant period of the series.
    """

    def __init__(self, phase_len):
        super().__init__()
        self.phase_len = phase_len

    def forward(self, x):
        b, l, c = x.shape
        lp = self.phase_len
        p = (l + lp - 1) // lp
        pad = p * lp - l
        if pad:
            x = torch.cat([x[:, -pad:, :], x], dim=1)
        return x.reshape(b, p, lp, c).permute(0, 3, 2, 1).contiguous()


class LatticeMix(nn.Module):
    """One mixing block over the (variate x phase) lattice.

    Three cheap operators, no attention:
      local   depthwise k_v x k_l convolution, circular along the phase axis
      global  rank-r_z operator along the phase axis, shared across channels
      point   1x1 mixing across the feature axis

    Either of the three can be switched off, which is how the ablations
    isolate what each axis contributes.
    """

    def __init__(self, d, kv, kl, rz, phase_len, q, act='gelu', local=True, glob=True, point=True):
        super().__init__()
        assert kv % 2 == 1 and kl % 2 == 1, 'kernel sizes must be odd'
        self.kv = kv
        self.kl = kl
        self.local = local
        self.glob = glob
        self.point = point
        self.q = q
        if local:
            self.dw = nn.Conv2d(d, d, (kv, kl), groups=d, bias=False)
        if glob:
            self.down = nn.Linear(phase_len, rz, bias=False)
            self.up = nn.Linear(rz, phase_len, bias=False)
        self.mix = nn.Linear(d, d) if point else None
        self.act = {'gelu': nn.GELU(), 'relu': nn.ReLU(), 'none': nn.Identity()}[act]

    def forward(self, z):
        # z: [B, C, Lp, d]
        h = z
        if self.local:
            av, al = self.kv // 2, self.kl // 2
            h = h.permute(0, 3, 1, 2)                                  # [B, d, C, Lp]
            if al:
                h = torch.cat([h[..., -al:], h, h[..., :al]], dim=-1)  # circular, phase axis
            if av:
                h = F.pad(h, (0, 0, av, av))                           # zeros, variate axis
            h = F.conv2d(h, self.q.weight(self.dw.weight), None,
                         groups=h.shape[1]).permute(0, 2, 3, 1)        # [B, C, Lp, d]
        if self.glob:
            hp = h.permute(0, 1, 3, 2)                                 # [B, C, d, Lp]
            g = F.linear(F.linear(hp, self.q.weight(self.down.weight)),
                         self.q.weight(self.up.weight))
            h = h + g.permute(0, 1, 3, 2)
        h = self.q.act(self.act(h))
        if self.mix is None:
            return z + h
        return z + F.linear(h, self.q.weight(self.mix.weight), self.mix.bias)


@dataclass(frozen=True, slots=True)
class PriSmConfig(ModelConfig):
    d_model: int = 512
    e_layers: int = 2
    k_l: int = 3
    k_v: int = 3
    norm_window: int = 32
    phase_len: int = 24
    prism_act: str = "gelu"
    prism_global: int = 1
    prism_local: int = 1
    prism_norm: str = "local"
    prism_norm_floor: float = 0.1
    prism_norm_min: float = 0.05
    prism_skip: int = 1
    prism_point: int = 1
    prism_pow2: int = 0
    prism_qat: int = 0
    prism_act_bits: int = 0
    r_z: int = 2


@models.register("PriSm")
class PriSm(Model):
    """PriSm: phase-lattice forecasting: fold the series, then mix locally on every axis."""

    def __init__(self, config: PriSmConfig, dims: Dims) -> None:
        super().__init__()
        self.config = config
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.phase_len = config.phase_len
        self.d = config.d_model
        self.q = Quant(pow2=getattr(config, 'prism_pow2', 0),
                       weight_bits=getattr(config, 'prism_qat', 0),
                       act_bits=getattr(config, 'prism_act_bits', 0))
        self.norm = LocalNorm(config.norm_window, config.prism_norm,
                               getattr(config, "prism_norm_floor", 0.1),
                               getattr(config, "prism_norm_min", 0.05))
        self.fold = PhaseFold(self.phase_len)
        p_in = (self.seq_len + self.phase_len - 1) // self.phase_len
        self.p_out = (self.pred_len + self.phase_len - 1) // self.phase_len
        self.in_proj = nn.Linear(p_in, self.d)
        self.blocks = nn.ModuleList(
            LatticeMix(self.d, config.k_v, config.k_l, config.r_z, self.phase_len, self.q,
                       config.prism_act, bool(config.prism_local), bool(config.prism_global),
                       bool(config.prism_point))
            for _ in range(config.e_layers))
        self.out_proj = nn.Linear(self.d, self.p_out)
        # linear path from the folded input straight to the folded output, so the
        # long horizon keeps an exact linear extrapolator alongside the local mixing
        self.skip = nn.Linear(p_in, self.p_out, bias=False) if getattr(config, 'prism_skip', 1) else None

    @classmethod
    def config_type(cls) -> type[ModelConfig]:
        return PriSmConfig

    @classmethod
    def build(cls, config: ModelConfig, dims: Dims) -> Model:
        return cls(config, dims)

    def forward(self, batch: Batch) -> Forecast:
        x_enc = batch.x
        x, mean, std = self.norm(x_enc)
        zk = self.fold(x)
        z = F.linear(zk, self.q.weight(self.in_proj.weight), self.in_proj.bias)
        for blk in self.blocks:
            z = blk(z)
        y = F.linear(z, self.q.weight(self.out_proj.weight), self.out_proj.bias)
        if self.skip is not None:
            y = y + F.linear(zk, self.q.weight(self.skip.weight))
        y = y.permute(0, 3, 2, 1).reshape(x_enc.shape[0], -1, x_enc.shape[2])[:, :self.pred_len, :]
        return Point(y * std + mean)

