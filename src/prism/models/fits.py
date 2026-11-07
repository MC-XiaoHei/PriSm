from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from prism.config import Dims, ModelConfig
from prism.fields import Batch, Forecast, Point
from prism.models.base import Model, models


@dataclass(frozen=True, slots=True)
class FITSConfig(ModelConfig):
    cut_freq: int = 4
    individual: bool = False


@models.register("FITS")
class FITS(Model):

    def __init__(self, config: FITSConfig, dims: Dims) -> None:
        super().__init__()
        self.config = config
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.individual = config.individual
        self.channels = dims.enc_in

        self.dominance_freq = config.cut_freq
        self.length_ratio = (self.seq_len + self.pred_len) / self.seq_len

        if self.individual:
            self.freq_upsampler = nn.ModuleList()
            for i in range(self.channels):
                self.freq_upsampler.append(
                    nn.Linear(self.dominance_freq, int(self.dominance_freq * self.length_ratio)).to(torch.cfloat))

        else:
            self.freq_upsampler = nn.Linear(self.dominance_freq, int(self.dominance_freq * self.length_ratio)).to(
                torch.cfloat)  # complex layer for frequency upcampling]

    @classmethod
    def config_type(cls) -> type[ModelConfig]:
        return FITSConfig

    @classmethod
    def build(cls, config: ModelConfig, dims: Dims) -> Model:
        return cls(config, dims)

    def forward(self, batch: Batch) -> Forecast:
        x = batch.x
        # RIN
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x = x - x_mean
        x_var = torch.var(x, dim=1, keepdim=True) + 1e-5
        x = x / torch.sqrt(x_var)

        low_specx = torch.fft.rfft(x, dim=1)
        low_specx[:, self.dominance_freq:] = 0  # LPF
        low_specx = low_specx[:, 0:self.dominance_freq, :]  # LPF
        if self.individual:
            low_specxy_ = torch.zeros(
                [low_specx.size(0), int(self.dominance_freq * self.length_ratio), low_specx.size(2)],
                dtype=low_specx.dtype).to(low_specx.device)
            for i in range(self.channels):
                low_specxy_[:, :, i] = self.freq_upsampler[i](low_specx[:, :, i].permute(0, 1)).permute(0, 1)
        else:
            low_specxy_ = self.freq_upsampler(low_specx.permute(0, 2, 1)).permute(0, 2, 1)
        low_specxy = torch.zeros(
            [low_specxy_.size(0), int((self.seq_len + self.pred_len) / 2 + 1), low_specxy_.size(2)],
            dtype=low_specxy_.dtype).to(low_specxy_.device)
        low_specxy[:, 0:low_specxy_.size(1), :] = low_specxy_  # zero padding
        low_xy = torch.fft.irfft(low_specxy, dim=1)
        low_xy = low_xy * self.length_ratio  # energy compemsation for the length change

        xy = (low_xy) * torch.sqrt(x_var) + x_mean
        return Point(xy[:, -self.pred_len:, :])
