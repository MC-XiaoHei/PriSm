from __future__ import annotations

from abc import ABC, abstractmethod

from torch import nn

from prism.config import Dims, ModelConfig
from prism.fields import Batch, Forecast
from prism.registry import Registry


class Model(nn.Module, ABC):

    @classmethod
    @abstractmethod
    def config_type(cls) -> type[ModelConfig]: ...

    @classmethod
    @abstractmethod
    def build(cls, config: ModelConfig, dims: Dims) -> Model: ...

    @abstractmethod
    def forward(self, batch: Batch) -> Forecast: ...


models: Registry[type[Model]] = Registry("model")
