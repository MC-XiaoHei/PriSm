from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch import nn

from prism.config import BackboneConfig
from prism.registry import Registry


class Backbone(nn.Module, ABC):
    def __init__(self, config: BackboneConfig) -> None:
        super().__init__()
        self.config = config

    @classmethod
    @abstractmethod
    def config_type(cls) -> type[BackboneConfig]: ...

    @classmethod
    @abstractmethod
    def build(cls, config: BackboneConfig) -> Backbone: ...

    def freeze(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)


backbones: Registry[type[Backbone]] = Registry("backbone")


def locate(root: nn.Module) -> list[Backbone]:
    return [module for module in root.modules() if isinstance(module, Backbone)]


def load_weights(backbone: Backbone) -> bool:
    path = Path(backbone.config.weights_dir) / f"{backbone.config.id}.pth"
    if not path.exists():
        return False
    backbone.load_state_dict(torch.load(path, map_location="cpu"))
    return True
