from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from prism.config import Dims, ModelConfig
from prism.fields import Batch, Forecast, Point
from prism.layers.encdec import series_decomp
from prism.models.base import Model, models


@dataclass(frozen=True, slots=True)
class DLinearConfig(ModelConfig):
    moving_avg: int = 25
    individual: bool = False


@models.register("DLinear")
class DLinear(Model):

    def __init__(self, config: DLinearConfig, dims: Dims) -> None:
        super().__init__()
        self.config = config
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.channels = dims.enc_in
        self.individual = config.individual
        self.decomposition = series_decomp(config.moving_avg)

        count = self.channels if self.individual else 1
        self.seasonal = nn.ModuleList([nn.Linear(self.seq_len, self.pred_len) for _ in range(count)])
        self.trend = nn.ModuleList([nn.Linear(self.seq_len, self.pred_len) for _ in range(count)])
        for layer in (*self.seasonal, *self.trend):
            layer.weight = nn.Parameter((1 / self.seq_len) * torch.ones(self.pred_len, self.seq_len))

    @classmethod
    def config_type(cls) -> type[ModelConfig]:
        return DLinearConfig

    @classmethod
    def build(cls, config: ModelConfig, dims: Dims) -> Model:
        return cls(config, dims)

    def _project(self, layer: nn.ModuleList, x: Tensor) -> Tensor:
        if not self.individual:
            return layer[0](x)
        out = torch.zeros(x.size(0), x.size(1), self.pred_len, device=x.device, dtype=x.dtype)
        for i in range(self.channels):
            out[:, i, :] = layer[i](x[:, i, :])
        return out

    def forward(self, batch: Batch) -> Forecast:
        seasonal, trend = self.decomposition(batch.x)
        seasonal = self._project(self.seasonal, seasonal.permute(0, 2, 1))
        trend = self._project(self.trend, trend.permute(0, 2, 1))
        return Point((seasonal + trend).permute(0, 2, 1))
