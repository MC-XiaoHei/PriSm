from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torch.utils.data import Dataset, default_collate

from prism.config import DataConfig, Dims, ModelConfig
from prism.data.scaler import Scaler
from prism.data.timefeatures import time_features
from prism.fields import Batch
from prism.registry import Registry

Flag = Literal["train", "val", "test"]

_FLAGS: dict[str, int] = {"train": 0, "val": 1, "test": 2}
Flags: tuple[Flag, ...] = ("train", "val", "test")

Array = npt.NDArray[np.float32]


def collate(items: list[Batch]) -> Batch:
    return Batch(
        x=default_collate([item.x for item in items]),
        y=default_collate([item.y for item in items]),
        x_mark=default_collate([item.x_mark for item in items]),
        y_mark=default_collate([item.y_mark for item in items]),
        aux={
            key: default_collate([item.aux[key] for item in items])
            for key in items[0].aux
        },
    )


class BaseDataset(Dataset[Batch], ABC):
    def __init__(self, data: DataConfig, model: ModelConfig, flag: Flag) -> None:
        self.cfg = data
        self.seq_len = model.seq_len
        self.label_len = model.label_len
        self.pred_len = model.pred_len
        self.index = _FLAGS[flag]

        frame = self.read_raw()
        columns = self.select_columns(frame)
        self.channels = len(columns)
        values = frame[columns].to_numpy(dtype=np.float32)

        self._starts, self._ends = zip(*self.windows(len(frame)))

        self.scaler: Scaler | None = Scaler() if data.scale else None
        if self.scaler is not None:
            self.scaler.fit(values[self._starts[0]:self._ends[0]])
            values = self.scaler.transform(values)

        marks = self.time_marks(frame)
        start, end = self._starts[self.index], self._ends[self.index]
        self.values: Array = np.ascontiguousarray(values[start:end], dtype=np.float32)
        self.marks: Array = np.ascontiguousarray(marks[start:end], dtype=np.float32)

    @abstractmethod
    def read_raw(self) -> pd.DataFrame: ...

    @abstractmethod
    def breaks(self, n: int) -> tuple[int, int, int]: ...

    def windows(self, n: int) -> tuple[tuple[int, int], ...]:
        train_end, val_end, test_end = self.breaks(n)
        return (
            (0, train_end),
            (train_end - self.seq_len, val_end),
            (val_end - self.seq_len, test_end),
        )

    def select_columns(self, frame: pd.DataFrame) -> list[str]:
        if self.cfg.features in ("M", "MS"):
            return [name for name in frame.columns if name != "date"]
        return [self.cfg.target]

    def time_marks(self, frame: pd.DataFrame) -> Array:
        dates = pd.DatetimeIndex(frame["date"])
        if self.cfg.embed == "timeF":
            return time_features(dates, self.cfg.freq).T.astype(np.float32)
        return self.fixed_marks(dates)

    def fixed_marks(self, dates: pd.DatetimeIndex) -> Array:
        stacked = np.stack([dates.month, dates.day, dates.weekday, dates.hour], axis=1)
        return stacked.astype(np.float32)

    @property
    def enc_in(self) -> int:
        return self.channels

    @property
    def c_out(self) -> int:
        return self.channels

    def inverse_transform(self, data: Array) -> Array:
        if self.scaler is None:
            return data
        return self.scaler.inverse_transform(data)

    def __getitem__(self, index: int) -> Batch:
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        return Batch(
            x=torch.from_numpy(self.values[s_begin:s_end]),
            y=torch.from_numpy(self.values[r_begin:r_end]),
            x_mark=torch.from_numpy(self.marks[s_begin:s_end]),
            y_mark=torch.from_numpy(self.marks[r_begin:r_end]),
        )

    def __len__(self) -> int:
        return len(self.values) - self.seq_len - self.pred_len + 1


datasets: Registry[type[BaseDataset]] = Registry("dataset")


def build_datasets(data: DataConfig, model: ModelConfig) -> dict[Flag, BaseDataset]:
    cls = datasets.get(data.data)
    return {flag: cls(data, model, flag) for flag in Flags}


def dims_of(dataset: BaseDataset) -> Dims:
    return Dims(enc_in=dataset.enc_in, c_out=dataset.c_out)


def prepare(data: DataConfig, model: ModelConfig) -> tuple[Dims, dict[Flag, BaseDataset]]:
    built = build_datasets(data, model)
    return dims_of(built["train"]), built


def fingerprint(data: DataConfig) -> str:
    path = Path(data.root_path) / data.data_path
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
