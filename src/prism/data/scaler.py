from __future__ import annotations

import numpy as np
import numpy.typing as npt
from sklearn.preprocessing import StandardScaler as _StandardScaler

Array = npt.NDArray[np.float32]


class Scaler:
    def __init__(self) -> None:
        self._inner = _StandardScaler()

    def fit(self, data: Array) -> None:
        self._inner.fit(data)

    def transform(self, data: Array) -> Array:
        return np.asarray(self._inner.transform(data), dtype=np.float32)

    def inverse_transform(self, data: Array) -> Array:
        return np.asarray(self._inner.inverse_transform(data), dtype=np.float32)
