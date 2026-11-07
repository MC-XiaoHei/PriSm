"""Accuracy of a trained checkpoint after export to the fixed-point engine.

For one run directory: export the weights to the hex tables the RTL reads, score the
float network and the integer reference over the test split, and drop one test window
into the tables so the same directory can drive the RTL simulation.

    python experiments/deploy.py <run-dir> [windows]

`windows` samples that many evenly spaced windows; without it the whole split is used.
Channel counts differ per dataset, and the integer reference is pure Python, so a wide
dataset wants a sample.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prism.config import DataConfig, ModelConfig  # noqa: E402
from prism.data import prepare  # noqa: E402
from prism.fields import Batch  # noqa: E402
from prism.hw import export  # noqa: E402
from prism.models import prism_fixed as fx  # noqa: E402
from prism.models.prism import PriSm, PriSmConfig  # noqa: E402


def load(run_dir: Path):
    resolved = json.loads((run_dir / "config.json").read_text())
    model_cfg = PriSmConfig(**resolved["model"])
    short = {k: resolved["model"][k] for k in ("seq_len", "label_len", "pred_len", "freq")}
    dims, sets = prepare(DataConfig(**resolved["data"]), ModelConfig(**short))
    model = PriSm.build(model_cfg, dims)
    model.load_state_dict(torch.load(run_dir / "checkpoint/best.pth", map_location="cpu"))
    model.eval()
    return resolved, model, sets["test"]


def predict(model, item, pred_len, w, channels):
    with torch.no_grad():
        y = model(Batch(x=item.x[None], y=item.y[None],
                        x_mark=item.x_mark[None], y_mark=item.y_mark[None])).value[0]
    truth = item.y[-pred_len:][:, channels]
    fixed = []
    for c in channels:
        xq = [int(round(v * 4096)) for v in item.x[:, c].tolist()]
        yq, mean, std, _ = fx.run(xq, w)
        fixed.append(torch.tensor([v / 4096.0 * (std / 4096.0) + mean / 4096.0 for v in yq],
                                  dtype=torch.float64))
    return y[:, channels], truth, torch.stack(fixed, 1)


def main() -> None:
    run_dir = ROOT / sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    resolved, model, test = load(run_dir)
    pred_len = resolved["model"]["pred_len"]

    out = ROOT / "build" / "deploy" / run_dir.name
    export.write_hex(export.tables(model.state_dict()), out)
    fx.OUT = out
    w = fx.make_weights()

    picks = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    width = test[0].x.shape[1]
    channels = range(width) if not picks or picks >= width else range(0, width, max(1, width // picks))
    step = max(1, len(test) // limit) if limit else 1
    idx = range(0, len(test), step)

    f_sum = i_sum = 0.0
    first = None
    n = 0
    for k in idx:
        y, truth, fixed = predict(model, test[k], pred_len, w, channels)
        f_sum += float(((y - truth) ** 2).mean())
        i_sum += float(((fixed - truth) ** 2).mean())
        if first is None:
            first = (test[k], fixed)
        n += 1
    print(f"{run_dir.name}: windows {n}  float {f_sum / n:.4f}  integer {i_sum / n:.4f}")

    item, _ = first
    xq = [int(round(v * 4096)) for v in item.x[:, 0].tolist()]
    fx.emit("x_in.hex", xq)
    yq, *_ = fx.run(xq, w)
    fx.emit("y_ref.hex", yq)


if __name__ == "__main__":
    main()
