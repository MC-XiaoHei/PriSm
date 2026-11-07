from __future__ import annotations

import os
import random

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader

from prism.backbones import load_weights, locate
from prism.config import DataConfig, ModelConfig, Resolved
from prism.data import BaseDataset, Flag, Flags, collate, fingerprint
from prism.fields import Batch, point_of
from prism.losses import losses
from prism.models.base import models
from prism.run.artifacts import RunDir, new_meta, run_id
from prism.run.evaluate import evaluate
from prism.run.telemetry import ResourceMonitor, ResourceStats
from prism.schedulers import schedulers


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _move(batch: Batch, device: torch.device) -> Batch:
    return Batch(
        x=batch.x.to(device),
        y=batch.y.to(device),
        x_mark=batch.x_mark.to(device),
        y_mark=batch.y_mark.to(device),
        aux={key: value.to(device) for key, value in batch.aux.items()},
    )


def _target(batch: Batch, pred_len: int) -> Tensor:
    return batch.y[:, -pred_len:, :]


def score(pred: Tensor, true: Tensor, features: str) -> tuple[Tensor, Tensor]:
    if features == "MS":
        return pred[:, :, -1:], true[:, :, -1:]
    return pred, true


class Trainer:
    def __init__(
        self,
        resolved: Resolved[ModelConfig, DataConfig],
        datasets: dict[Flag, BaseDataset],
        out_root: str,
        reuse: bool = False,
    ) -> None:
        self.resolved = resolved
        self.run = resolved.run
        self.features = resolved.data.features
        self.device = pick_device(self.run.device)
        self.pred_len = resolved.model.pred_len
        set_seed(self.run.seed)

        model_cls = models.get(self.run.model)
        self.model = model_cls.build(resolved.model, resolved.dims).to(self.device)
        self.loaders: dict[Flag, DataLoader[Batch]] = {
            flag: DataLoader(
                datasets[flag],
                batch_size=self.run.batch_size,
                shuffle=flag == "train",
                num_workers=self.run.num_workers,
                drop_last=False,
                collate_fn=collate,
            )
            for flag in Flags
        }
        self.fingerprint = fingerprint(resolved.data)
        self.ident = run_id(resolved, self.fingerprint)
        self.rundir = RunDir(out_root, self.ident)
        self.reuse = reuse

    def _prepare_backbones(self) -> None:
        for backbone in locate(self.model):
            if backbone.config.freeze:
                backbone.freeze()
            if load_weights(backbone):
                print(f"loading backbone weights {backbone.config.id}")

    def _terms(self, batch: Batch) -> tuple[Tensor, Tensor]:
        pred = point_of(self.model(batch))
        return score(pred, _target(batch, self.pred_len), self.features)

    def _epoch(self, optimizer: Optimizer, criterion: nn.Module) -> float:
        self.model.train()
        total = 0.0
        count = 0
        for batch in self.loaders["train"]:
            moved = _move(batch, self.device)
            optimizer.zero_grad()
            if self.run.use_amp:
                with torch.autocast(device_type=self.device.type):
                    pred, true = self._terms(moved)
                    loss = criterion(pred, true)
                loss.backward()
            else:
                pred, true = self._terms(moved)
                loss = criterion(pred, true)
                loss.backward()
            optimizer.step()
            total += float(loss.item())
            count += 1
        return total / max(count, 1)

    def _validate(self, criterion: nn.Module) -> float:
        self.model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for batch in self.loaders["val"]:
                pred, true = self._terms(_move(batch, self.device))
                total += float(criterion(pred, true).item())
                count += 1
        return total / max(count, 1)

    def predict(self) -> tuple[np.ndarray, np.ndarray, ResourceStats]:
        self.model.eval()
        preds: list[np.ndarray] = []
        trues: list[np.ndarray] = []
        monitor = ResourceMonitor(
            device=self.device,
            amp=self.run.use_amp,
            batch_size=self.run.batch_size,
            includes_dataloading=True,
        )
        with monitor:
            with torch.no_grad():
                for batch in self.loaders["test"]:
                    pred, true = self._terms(_move(batch, self.device))
                    preds.append(pred.detach().cpu().numpy())
                    trues.append(true.detach().cpu().numpy())
        pred = np.concatenate(preds, axis=0)
        true = np.concatenate(trues, axis=0)
        return pred, true, monitor.stats()

    def fit(self) -> RunDir:
        if self.reuse and self.rundir.metrics_path.exists():
            print(f"reuse {self.rundir.path}")
            return self.rundir
        self.rundir.ensure()
        self.rundir.write_config(self.resolved)
        self._prepare_backbones()

        criterion = losses.get(self.run.loss)()
        optimizer = (
            Adam(
                [parameter for parameter in self.model.parameters() if parameter.requires_grad],
                lr=self.run.learning_rate,
            )
            if self.run.train_epochs > 0
            else None
        )
        step_lr = schedulers.get(self.run.lradj)

        best_val = float("inf")
        best_epoch = 0
        patience = 0
        best_state: dict[str, Tensor] | None = None
        monitor = ResourceMonitor(
            device=self.device,
            amp=self.run.use_amp,
            batch_size=self.run.batch_size,
            includes_dataloading=True,
        )
        with monitor:
            for epoch in range(1, self.run.train_epochs + 1):
                assert optimizer is not None
                train_loss = self._epoch(optimizer, criterion)
                val_loss = self._validate(criterion)
                print(f"epoch {epoch} train {train_loss:.6f} val {val_loss:.6f}")
                if val_loss < best_val:
                    best_val = val_loss
                    best_epoch = epoch
                    patience = 0
                    if self.run.save_checkpoint:
                        torch.save(self.model.state_dict(), self.rundir.checkpoint)
                    else:
                        best_state = {
                            key: value.detach().clone() for key, value in self.model.state_dict().items()
                        }
                else:
                    patience += 1
                    if patience >= self.run.patience:
                        print(f"early stop at epoch {epoch}")
                        break
                lr = step_lr(epoch, self.run)
                if lr is not None:
                    for group in optimizer.param_groups:
                        group["lr"] = lr

        train_resources = monitor.stats()
        if self.run.train_epochs > 0 and best_state is not None:
            self.model.load_state_dict(best_state)
        elif self.run.train_epochs > 0:
            state = torch.load(self.rundir.checkpoint, map_location=self.device)
            self.model.load_state_dict(state)
        pred, true, test_resources = self.predict()
        # test-set dumps dominate disk use on wide datasets; metrics stay in metrics/
        if os.environ.get("PRISM_SAVE_PREDICTIONS", "1") != "0":
            self.rundir.save_predictions(pred, true)
        self.rundir.write_meta(
            new_meta(
                self.ident,
                self.resolved,
                self.fingerprint,
                best_epoch,
                best_val,
                train_resources,
                test_resources,
            )
        )
        self.rundir.write_metrics(evaluate(pred, true), {})
        return self.rundir


def train(
    resolved: Resolved[ModelConfig, DataConfig],
    datasets: dict[Flag, BaseDataset],
    out_root: str,
    reuse: bool = False,
) -> RunDir:
    return Trainer(resolved, datasets, out_root, reuse).fit()
