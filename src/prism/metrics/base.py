from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from torch import Tensor

from prism.fields import Field
from prism.registry import Registry


@dataclass(frozen=True, slots=True)
class MetricInput:
    pred: Tensor
    true: Tensor


F = TypeVar("F", bound=Callable[[MetricInput], float])


@dataclass(frozen=True, slots=True)
class MetricSpec:
    id: str
    fn: Callable[[MetricInput], float]
    requires: frozenset[Field]


metrics: Registry[MetricSpec] = Registry("metric")


def point_metric(ident: str) -> Callable[[F], F]:
    requires = frozenset({Field.POINT, Field.TRUE})

    def deco(fn: F) -> F:
        metrics.register(ident)(MetricSpec(id=ident, fn=fn, requires=requires))
        return fn

    return deco
