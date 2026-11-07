from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from prism.data.base import Array, BaseDataset, datasets


class _ETT(BaseDataset):
    hours_per_day = 24

    def read_raw(self) -> pd.DataFrame:
        return pd.read_csv(Path(self.cfg.root_path) / self.cfg.data_path)

    def breaks(self, n: int) -> tuple[int, int, int]:
        day = self.hours_per_day
        train_end = 12 * 30 * day
        return (train_end, train_end + 4 * 30 * day, train_end + 8 * 30 * day)


@datasets.register("ETTh1")
@datasets.register("ETTh2")
class ETTHour(_ETT):
    hours_per_day = 24


@datasets.register("ETTm1")
@datasets.register("ETTm2")
class ETTMinute(_ETT):
    hours_per_day = 96

    def fixed_marks(self, dates: pd.DatetimeIndex) -> Array:
        stacked = np.stack(
            [dates.month, dates.day, dates.weekday, dates.hour, dates.minute // 15], axis=1
        )
        return stacked.astype(np.float32)


@datasets.register("custom")
@datasets.register("ECL")
@datasets.register("Weather")
@datasets.register("Traffic")
@datasets.register("Exchange")
class Custom(BaseDataset):
    def read_raw(self) -> pd.DataFrame:
        frame = pd.read_csv(Path(self.cfg.root_path) / self.cfg.data_path)
        others = [name for name in frame.columns if name not in ("date", self.cfg.target)]
        return frame[["date", *others, self.cfg.target]]

    def breaks(self, n: int) -> tuple[int, int, int]:
        train_end = int(n * self.cfg.train_ratio)
        test_count = int(n * self.cfg.test_ratio)
        return (train_end, n - test_count, n)


@datasets.register("Solar")
class Solar(BaseDataset):
    """TSLib Dataset_Solar: headerless numeric file, 0.7/0.1/0.2 split, zero time features."""

    def read_raw(self) -> pd.DataFrame:
        frame = pd.read_csv(Path(self.cfg.root_path) / self.cfg.data_path, header=None)
        frame.columns = frame.columns.astype(str)
        return frame

    def breaks(self, n: int) -> tuple[int, int, int]:
        train = int(n * 0.7)
        return (train, train + int(n * 0.1), n)

    def windows(self, n: int) -> tuple[tuple[int, int], ...]:
        train = int(n * 0.7)
        valid = int(n * 0.1)
        test = int(n * 0.2)
        return ((0, train), (train - self.seq_len, train + valid), (n - test - self.seq_len, n))

    def time_marks(self, frame: pd.DataFrame) -> Array:
        return np.zeros((len(frame), 4), dtype=np.float32)


@datasets.register("PEMS03")
@datasets.register("PEMS04")
@datasets.register("PEMS07")
@datasets.register("PEMS08")
class PEMS(BaseDataset):
    """TSLib Dataset_PEMS: channel 0 of the npz, 0.6/0.2/0.2 non-overlapping split."""

    def read_raw(self) -> pd.DataFrame:
        raw = np.load(Path(self.cfg.root_path) / self.cfg.data_path, allow_pickle=True)["data"]
        frame = pd.DataFrame(raw[:, :, 0].astype(np.float32))
        frame.columns = frame.columns.astype(str)
        return frame.ffill().bfill()

    def breaks(self, n: int) -> tuple[int, int, int]:
        return (int(n * 0.6), int(n * 0.8), n)

    def windows(self, n: int) -> tuple[tuple[int, int], ...]:
        train = int(n * 0.6)
        valid = int(n * 0.2)
        return ((0, train), (train, train + valid), (train + valid, n))

    def time_marks(self, frame: pd.DataFrame) -> Array:
        return np.zeros((len(frame), 4), dtype=np.float32)
