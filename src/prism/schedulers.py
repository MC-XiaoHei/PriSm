from __future__ import annotations

import math
from typing import Protocol

from prism.config import RunConfig
from prism.registry import Registry


class Scheduler(Protocol):
    def __call__(self, epoch: int, run: RunConfig) -> float | None: ...


schedulers: Registry[Scheduler] = Registry("lradj")

_TYPE2: dict[int, float] = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}


@schedulers.register("type1")
def _type1(epoch: int, run: RunConfig) -> float:
    return run.learning_rate * (0.5 ** ((epoch - 1) // 1))


@schedulers.register("type2")
def _type2(epoch: int, run: RunConfig) -> float | None:
    return _TYPE2.get(epoch)


@schedulers.register("type3")
def _type3(epoch: int, run: RunConfig) -> float:
    if epoch < 3:
        return run.learning_rate
    return run.learning_rate * (0.9 ** ((epoch - 3) // 1))


@schedulers.register("constant")
def _constant(epoch: int, run: RunConfig) -> float:
    return run.learning_rate


@schedulers.register("sigmoid")
def _sigmoid(epoch: int, run: RunConfig) -> float:
    rate, smooth, warm = 0.5, 10, 10
    return run.learning_rate / (1 + math.exp(-rate * (epoch - warm))) - run.learning_rate / (
        1 + math.exp(-rate / smooth * (epoch - warm * smooth))
    )


@schedulers.register("cosine")
def _cosine(epoch: int, run: RunConfig) -> float:
    return run.learning_rate / 2 * (1 + math.cos(epoch / run.train_epochs * math.pi))
