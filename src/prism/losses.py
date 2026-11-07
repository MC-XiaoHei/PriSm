from __future__ import annotations

from collections.abc import Callable

from torch import nn

from prism.registry import Registry

LossFactory = Callable[[], nn.Module]

losses: Registry[LossFactory] = Registry("loss")


@losses.register("MSE")
def _mse() -> nn.Module:
    return nn.MSELoss()


@losses.register("MAE")
def _mae() -> nn.Module:
    return nn.L1Loss()
