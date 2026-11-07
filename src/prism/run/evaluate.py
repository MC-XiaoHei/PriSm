from __future__ import annotations

from collections.abc import Sequence

import torch

from prism.fields import Field
from prism.metrics.base import MetricInput, metrics
from prism.run.artifacts import RunDir

_SAVED: frozenset[Field] = frozenset({Field.POINT, Field.TRUE})


def evaluate(pred: object, true: object, names: Sequence[str] | None = None) -> dict[str, float]:
    pred_tensor = torch.as_tensor(pred)
    true_tensor = torch.as_tensor(true)

    chosen = tuple(names) if names else metrics.ids()
    results: dict[str, float] = {}
    skipped: dict[str, str] = {}
    for name in chosen:
        spec = metrics.get(name)
        missing = spec.requires - _SAVED
        if missing:
            skipped[name] = "missing " + ",".join(sorted(field.value for field in missing))
            continue
        data = MetricInput(pred=pred_tensor, true=true_tensor)
        results[name] = spec.fn(data)

    return results


def evaluate_run(rundir: RunDir, names: Sequence[str] | None = None) -> dict[str, float]:
    pred, true = rundir.load_predictions()
    results = evaluate(pred, true, names)
    rundir.write_metrics(results, {})
    return results
