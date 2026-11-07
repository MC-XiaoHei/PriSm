from __future__ import annotations

import hashlib
import json

import torch
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt

from prism.config import DataConfig, Dims, ModelConfig, Resolved
from prism.run.telemetry import ResourceStats

SCHEMA_VERSION = 1

Array = npt.NDArray[np.float32]


def run_id(resolved: Resolved[ModelConfig, DataConfig], fingerprint: str) -> str:
    payload = {
        "run": asdict(resolved.run),
        "model": asdict(resolved.model),
        "data": asdict(resolved.data),
        "dims": asdict(resolved.dims),
        "fingerprint": fingerprint,
    }
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=_encode)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    model = resolved.model
    return (
        f"{resolved.run.model}_{resolved.data.data}"
        f"_sl{model.seq_len}_pl{model.pred_len}_{digest}"
    )


@dataclass(frozen=True, slots=True)
class RunMeta:
    run_id: str
    schema_version: int
    model: str
    dataset: str
    dims: Dims
    fingerprint: str
    epochs: int
    best_val: float
    created: str
    train_resources: ResourceStats
    test_resources: ResourceStats


def _encode(value: object) -> object:
    if isinstance(value, torch.device):
        return str(value)
    raise TypeError(f"cannot serialise {type(value).__name__}")


def _dump(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=_encode)


class RunDir:
    def __init__(self, root: str | Path, ident: str) -> None:
        self.path = Path(root) / ident
        self.config_path = self.path / "config.json"
        self.meta_path = self.path / "meta.json"
        self.checkpoint = self.path / "checkpoint" / "best.pth"
        self.predictions = self.path / "predictions"
        self.metrics_path = self.path / "metrics" / "metrics.json"

    def ensure(self) -> None:
        for folder in (self.checkpoint.parent, self.predictions, self.metrics_path.parent):
            folder.mkdir(parents=True, exist_ok=True)

    def write_config(self, resolved: Resolved[ModelConfig, DataConfig]) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "run": asdict(resolved.run),
            "model": asdict(resolved.model),
            "data": asdict(resolved.data),
            "dims": asdict(resolved.dims),
        }
        self.config_path.write_text(_dump(payload), encoding="utf-8")

    def write_meta(self, meta: RunMeta) -> None:
        self.meta_path.write_text(_dump(asdict(meta)), encoding="utf-8")

    def save_predictions(self, pred: Array, true: Array) -> None:
        np.savez_compressed(self.predictions / "pred.npz", array=pred.astype(np.float32))
        np.savez_compressed(self.predictions / "true.npz", array=true.astype(np.float32))

    def load_predictions(self) -> tuple[Array, Array]:
        with np.load(self.predictions / "pred.npz") as handle:
            pred: Array = handle["array"]
        with np.load(self.predictions / "true.npz") as handle:
            true: Array = handle["array"]
        return pred, true

    def write_metrics(
        self, results: Mapping[str, float], skipped: Mapping[str, str]
    ) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "metrics": dict(results),
            "skipped": dict(skipped),
        }
        self.metrics_path.write_text(_dump(payload), encoding="utf-8")


def new_meta(
    ident: str,
    resolved: Resolved[ModelConfig, DataConfig],
    fingerprint: str,
    epochs: int,
    best_val: float,
    train_resources: ResourceStats,
    test_resources: ResourceStats,
) -> RunMeta:
    return RunMeta(
        run_id=ident,
        schema_version=SCHEMA_VERSION,
        model=resolved.run.model,
        dataset=resolved.data.data,
        dims=resolved.dims,
        fingerprint=fingerprint,
        epochs=epochs,
        best_val=best_val,
        created=datetime.now().isoformat(timespec="seconds"),
        train_resources=train_resources,
        test_resources=test_resources,
    )
