from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from prism.config import Dims, ModelConfig
from prism.fields import Batch, Forecast, Point
from prism.models.base import Model, models


class RecurrentCycle(nn.Module):
    def __init__(self, cycle_len: int, channel_size: int) -> None:
        super().__init__()
        self.cycle_len = cycle_len
        self.channel_size = channel_size
        self.data = nn.Parameter(torch.zeros(cycle_len, channel_size), requires_grad=True)

    def forward(self, index: Tensor, length: int) -> Tensor:
        steps = torch.arange(length, device=index.device).view(1, -1)
        gather = (index.view(-1, 1) + steps) % self.cycle_len
        return self.data[gather]


@dataclass(frozen=True, slots=True)
class CycleNetConfig(ModelConfig):
    cycle: int = 24
    d_model: int = 512
    model_type: str = "linear"
    use_revin: int = 1


@models.register("CycleNet")
class CycleNet(Model):

    def __init__(self, config: CycleNetConfig, dims: Dims) -> None:
        super().__init__()
        self.config = config
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.enc_in = dims.enc_in
        self.cycle_len = config.cycle
        self.use_revin = bool(config.use_revin)
        self.cycle_queue = RecurrentCycle(config.cycle, dims.enc_in)

        if config.model_type == "linear":
            self.head = nn.Linear(self.seq_len, self.pred_len)
        elif config.model_type == "mlp":
            self.head = nn.Sequential(
                nn.Linear(self.seq_len, config.d_model),
                nn.ReLU(),
                nn.Linear(config.d_model, self.pred_len),
            )
        else:
            raise ValueError(f"unknown model_type {config.model_type}")

    @classmethod
    def config_type(cls) -> type[ModelConfig]:
        return CycleNetConfig

    @classmethod
    def build(cls, config: ModelConfig, dims: Dims) -> Model:
        return cls(config, dims)

    def _cycle_index(self, marks: Tensor) -> Tensor:
        hour = torch.round(marks[:, 0, 3] * 23).long()
        slots = max(self.cycle_len // 24, 1)
        if marks.shape[-1] > 4 and slots > 1:
            minute = torch.round(marks[:, 0, 4] * 59).long()
            hour = hour * slots + minute // (60 // slots)
        return hour % self.cycle_len

    def forward(self, batch: Batch) -> Forecast:
        index = self._cycle_index(batch.x_mark)
        x = batch.x

        if self.use_revin:
            mean = x.mean(dim=1, keepdim=True)
            var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - mean) / torch.sqrt(var)

        x = x - self.cycle_queue(index, self.seq_len)
        y = self.head(x.permute(0, 2, 1)).permute(0, 2, 1)
        y = y + self.cycle_queue((index + self.seq_len) % self.cycle_len, self.pred_len)

        if self.use_revin:
            y = y * torch.sqrt(var) + mean
        return Point(y)
