from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from torch import Tensor
import torch


class Field(StrEnum):
    X = "x"
    Y = "y"
    X_MARK = "x_mark"
    Y_MARK = "y_mark"
    EMBEDDING = "embedding"
    COV = "cov"
    POINT = "point"
    QUANTILES = "quantiles"
    SAMPLES = "samples"
    PARAMS = "params"
    TRUE = "true"
    TRAIN = "train_series"
    INDEX = "index"


@dataclass(frozen=True, slots=True)
class Point:
    value: Tensor


@dataclass(frozen=True, slots=True)
class Quantile:
    levels: tuple[float, ...]
    value: Tensor


@dataclass(frozen=True, slots=True)
class Sample:
    value: Tensor


@dataclass(frozen=True, slots=True)
class Parametric:
    family: str
    params: Mapping[str, Tensor]


Forecast = Point | Quantile | Sample | Parametric


def point_of(forecast: Forecast) -> Tensor:
    match forecast:
        case Point(value=value):
            return value
        case Quantile(levels=levels, value=value):
            mid = min(range(len(levels)), key=lambda i: abs(levels[i] - 0.5))
            return value[:, mid]
        case Sample(value=value):
            return value.mean(dim=1)
        case Parametric(family=family):
            raise ValueError(f"parametric family {family} has no point estimate")


def _empty_aux() -> dict[Field, Tensor]:
    return {}


@dataclass(frozen=True, slots=True)
class Batch:
    x: Tensor
    y: Tensor
    x_mark: Tensor
    y_mark: Tensor
    aux: Mapping[Field, Tensor] = field(default_factory=_empty_aux)


@dataclass(frozen=True, slots=True)
class Unavailable:
    reason: str


Measured = float | Unavailable
