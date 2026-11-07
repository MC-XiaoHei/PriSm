from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from types import TracebackType

import psutil
import torch

from prism.fields import Measured, Unavailable


def _rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


class _PeakRam:
    def __init__(self, interval: float = 0.05) -> None:
        self._interval = interval
        self._peak = 0.0
        self._done = threading.Event()
        self._worker: threading.Thread | None = None

    def _loop(self) -> None:
        while not self._done.is_set():
            self._peak = max(self._peak, _rss_mb())
            self._done.wait(self._interval)

    def start(self) -> None:
        self._peak = _rss_mb()
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    def stop(self) -> float:
        self._done.set()
        if self._worker is not None:
            self._worker.join(timeout=2.0)
        return self._peak


@dataclass(frozen=True, slots=True)
class ResourceStats:
    wall_s: float
    peak_ram_mb: Measured
    peak_vram_mb: Measured
    device: str
    amp: bool
    batch_size: int
    includes_dataloading: bool


class ResourceMonitor:
    def __init__(
        self,
        *,
        device: torch.device,
        amp: bool,
        batch_size: int,
        includes_dataloading: bool,
    ) -> None:
        self._device = device
        self._cuda = device.type == "cuda"
        self._amp = amp
        self._batch_size = batch_size
        self._includes = includes_dataloading
        self._peak_ram = _PeakRam()
        self._ram = 0.0
        self._vram = 0.0
        self._start = 0.0
        self._wall = 0.0

    def __enter__(self) -> ResourceMonitor:
        if self._cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self._start = time.perf_counter()
        self._peak_ram.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._cuda:
            torch.cuda.synchronize()
        self._wall = time.perf_counter() - self._start
        self._ram = self._peak_ram.stop()
        if self._cuda:
            self._vram = torch.cuda.max_memory_allocated() / (1024 * 1024)

    def stats(self) -> ResourceStats:
        vram: Measured = self._vram if self._cuda else Unavailable(f"{self._device.type} has no vram metric")
        return ResourceStats(
            wall_s=self._wall,
            peak_ram_mb=self._ram,
            peak_vram_mb=vram,
            device=self._device.type,
            amp=self._amp,
            batch_size=self._batch_size,
            includes_dataloading=self._includes,
        )
