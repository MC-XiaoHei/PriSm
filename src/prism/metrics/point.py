from __future__ import annotations

import numpy as np
import numpy.typing as npt
from torch import Tensor

from prism.metrics.base import MetricInput, point_metric

Array = npt.NDArray[np.float32]


def _np(tensor: Tensor) -> Array:
    return tensor.detach().cpu().numpy()


@point_metric("mae")
def mae(data: MetricInput) -> float:
    return float(np.mean(np.abs(_np(data.true) - _np(data.pred))))


@point_metric("mse")
def mse(data: MetricInput) -> float:
    return float(np.mean((_np(data.true) - _np(data.pred)) ** 2))


@point_metric("rmse")
def rmse(data: MetricInput) -> float:
    return float(np.sqrt(mse(data)))


@point_metric("mape")
def mape(data: MetricInput) -> float:
    true = _np(data.true)
    pred = _np(data.pred)
    return float(np.mean(np.abs((true - pred) / true)))


@point_metric("mspe")
def mspe(data: MetricInput) -> float:
    true = _np(data.true)
    pred = _np(data.pred)
    return float(np.mean(np.square((true - pred) / true)))
