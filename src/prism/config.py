from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from typing import Any, Generic, TypeVar


def param(default: Any, *, help: str = "", required: bool = False) -> Any:
    return field(default=default, metadata={"help": help, "required": required})


@dataclass(frozen=True, slots=True)
class ModelConfig:
    seq_len: int = param(96, help="lookback length")
    label_len: int = param(48, help="decoder start length")
    pred_len: int = param(96, help="prediction length")
    freq: str = param("h", help="time feature frequency, aligned with the dataset at run time")


@dataclass(frozen=True, slots=True)
class DataConfig:
    data: str = param("ETTh1", help="dataset name")
    root_path: str = param("./dataset", help="data root")
    data_path: str = param("ETTh1.csv", help="data file")
    features: str = param("M", help="M multivariate to multivariate, S univariate, MS multivariate to univariate")
    target: str = param("OT", help="target column under S and MS")
    freq: str = param("h", help="time frequency")
    embed: str = param("timeF", help="time encoding, timeF or fixed")
    scale: bool = param(True, help="normalise the data")
    inverse: bool = param(False, help="denormalise at evaluation")
    train_ratio: float = param(0.7, help="training share under a ratio split")
    test_ratio: float = param(0.2, help="test share under a ratio split")


@dataclass(frozen=True, slots=True)
class BackboneConfig:
    id: str = param("", help="backbone id")
    weights_dir: str = param("./weights", help="backbone weights directory")
    freeze: bool = param(True, help="freeze backbone parameters")


@dataclass(frozen=True, slots=True)
class RunConfig:
    task: str = param("long_term_forecast", help="task name")
    model: str = param("DLinear", help="model name")
    train_epochs: int = param(30, help="training epochs")
    patience: int = param(3, help="early stopping patience")
    batch_size: int = param(32, help="batch size")
    learning_rate: float = param(1e-4, help="learning rate")
    lradj: str = param("type1", help="learning rate schedule")
    loss: str = param("MSE", help="loss")
    seed: int = param(2021, help="random seed")
    num_workers: int = param(0, help="dataloader workers")
    device: str = param("auto", help="device, auto/cpu/cuda/mps")
    use_amp: bool = param(False, help="use mixed precision")
    save_checkpoint: bool = param(True, help="write the best weights to disk; keep them in memory only when off")
    out_dir: str = param("./runs", help="run output root")


@dataclass(frozen=True, slots=True)
class Dims:
    enc_in: int
    c_out: int
    classes: int | None = None


MC = TypeVar("MC", bound=ModelConfig)
DC = TypeVar("DC", bound=DataConfig)


@dataclass(frozen=True, slots=True)
class Resolved(Generic[MC, DC]):
    run: RunConfig
    model: MC
    data: DC
    dims: Dims


def align_freq(model: ModelConfig, data: DataConfig) -> ModelConfig:
    """Time-encoding frequency follows the dataset so minute-level data keeps its embedding width."""
    if any(item.name == "freq" for item in fields(type(model))):
        return replace(model, freq=data.freq)
    return model
