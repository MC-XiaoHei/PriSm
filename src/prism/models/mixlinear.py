from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from prism.config import Dims, ModelConfig
from prism.fields import Batch, Forecast, Point
from prism.models.base import Model, models


@dataclass(frozen=True, slots=True)
class MixLinearConfig(ModelConfig):
    alpha: float = 0.1
    lpf: int = 1
    period_len: int = 24


@models.register("MixLinear")
class MixLinear(Model):

    def __init__(self, config: MixLinearConfig, dims: Dims) -> None:
        super().__init__()
        self.config = config

        # get parameters
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.enc_in = dims.enc_in
        self.period_len = config.period_len

        self.kernel = self.period_len
        self.lpf = config.lpf
        self.alpha = config.alpha

        self.seg_num_x = math.ceil(self.seq_len / self.period_len)
        self.seg_num_y = math.ceil(self.pred_len / self.period_len)

        self.sqrt_seg_num_x = math.ceil(math.sqrt(self.seq_len / self.period_len))
        self.sqrt_seg_num_y = math.ceil(math.sqrt(self.pred_len / self.period_len))

        # TLinear
        self.TLinear1 = nn.Linear(self.sqrt_seg_num_x, self.sqrt_seg_num_y, bias=False)
        self.TLinear2 = nn.Linear(self.sqrt_seg_num_x, self.sqrt_seg_num_y, bias=False)

        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=self.kernel + 1,
                                stride=1, padding=int(self.kernel / 2), padding_mode="zeros", bias=False)

        # FLinear
        self.FLinear1 = nn.Linear(self.lpf, 2, bias=False).to(torch.cfloat)
        self.FLinear2 = nn.Linear(2, self.seg_num_y, bias=False).to(torch.cfloat)

    @classmethod
    def config_type(cls) -> type[ModelConfig]:
        return MixLinearConfig

    @classmethod
    def build(cls, config: ModelConfig, dims: Dims) -> Model:
        return cls(config, dims)

    def forward(self, batch: Batch) -> Forecast:
        x = batch.x
        batch_size = x.shape[0]
        # normalization and permute     b,s,c -> b,c,s
        seq_mean = torch.mean(x, dim=1).unsqueeze(1)
        x = (x - seq_mean).permute(0, 2, 1)
        x = self.conv1d(x.reshape(-1, 1, self.seq_len)).reshape(-1, self.enc_in, self.seq_len) + x

        # ->b,e,w,n
        x = x.reshape(batch_size, self.enc_in, -1, self.period_len).permute(0, 1, 3, 2)

        # Time Domain
        x_o = F.pad(x, (0, self.sqrt_seg_num_x ** 2 - x.shape[-1], 0, 0, 0, 0))

        x_o = x_o.reshape(batch_size, self.enc_in, self.period_len, self.sqrt_seg_num_x, self.sqrt_seg_num_x)

        x_o = self.TLinear1(x_o).permute(0, 1, 2, 4, 3)
        x_t = self.TLinear2(x_o).permute(0, 1, 2, 4, 3)

        x_t = x_t.reshape(batch_size, self.enc_in, self.period_len, -1).permute(0, 1, 3, 2).reshape(batch_size,
                                                                                                    self.enc_in,
                                                                                                    -1).permute(0, 2, 1)

        # Frequency Domain

        x_fft = torch.fft.fft(x, dim=3)[:, :, :, :self.lpf]
        x_fft = self.FLinear1(x_fft)
        x_fft = self.FLinear2(x_fft).reshape(batch_size, self.enc_in, self.period_len, -1)

        x_rfft = torch.fft.ifft(x_fft, dim=3).float()
        x_f = x_rfft.permute(0, 1, 3, 2).reshape(batch_size, self.enc_in, -1).permute(0, 2, 1)


        # Mix

        x = x_t[:, :self.pred_len, :] * self.alpha + seq_mean + x_f[:, :self.pred_len, :] * (1 - self.alpha)

        return Point(x[:, :self.pred_len, :])
