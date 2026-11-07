from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset

_ALIASES: dict[str, str] = {
    "t": "min",
    "T": "min",
    "H": "h",
    "S": "s",
    "L": "ms",
    "U": "us",
    "N": "ns",
    "d": "D",
    "w": "W",
    "m": "ME",
    "b": "B",
}


class TimeFeature:
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class SecondOfMinute(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return np.asarray(index.second / 59.0 - 0.5, dtype=np.float64)


class MinuteOfHour(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return np.asarray(index.minute / 59.0 - 0.5, dtype=np.float64)


class HourOfDay(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return np.asarray(index.hour / 23.0 - 0.5, dtype=np.float64)


class DayOfWeek(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return np.asarray(index.dayofweek / 6.0 - 0.5, dtype=np.float64)


class DayOfMonth(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return np.asarray((index.day - 1) / 30.0 - 0.5, dtype=np.float64)


class DayOfYear(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return np.asarray((index.dayofyear - 1) / 365.0 - 0.5, dtype=np.float64)


class MonthOfYear(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return np.asarray((index.month - 1) / 11.0 - 0.5, dtype=np.float64)


class WeekOfYear(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        week = np.asarray(index.isocalendar().week, dtype=np.float64)
        return (week - 1) / 52.0 - 0.5


def time_features_from_frequency_str(freq_str: str) -> list[TimeFeature]:
    features_by_offsets: dict[type[offsets.BaseOffset], list[type[TimeFeature]]] = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [MinuteOfHour, HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Second: [
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }

    offset = to_offset(_ALIASES.get(freq_str, freq_str))
    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]
    raise RuntimeError(f"unsupported frequency {freq_str}")


def time_features(dates: pd.DatetimeIndex, freq: str = "h") -> np.ndarray:
    return np.vstack([feature(dates) for feature in time_features_from_frequency_str(freq)])
