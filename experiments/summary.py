from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(runs: Path) -> list[dict]:
    rows = []
    for c in sorted(runs.rglob("config.json")):
        m = c.parent / "metrics" / "metrics.json"
        if not (m.exists() and c.exists()):
            continue
        cfg = json.loads(c.read_text(encoding="utf-8"))
        rows.append({"model": cfg["run"]["model"], "data": cfg["data"]["data"],
                     "pred": cfg["model"]["pred_len"],
                     "mse": json.loads(m.read_text(encoding="utf-8"))["metrics"]["mse"]})
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="experiments/summary.py")
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--metric", default="mse")
    ns = ap.parse_args(argv)

    best: dict[tuple[str, str, int], float] = defaultdict(lambda: float("inf"))
    for r in load(Path(ns.runs)):
        key = (r["model"], r["data"], r["pred"])
        best[key] = min(best[key], r[ns.metric])

    datasets = sorted({k[1] for k in best})
    models = sorted({k[0] for k in best})
    print("| model | " + " | ".join(datasets) + " | mean |")
    print("|---" * (len(datasets) + 2) + "|")
    for model in models:
        cells, means = [], []
        for ds in datasets:
            vals = [best[(model, ds, pl)] for pl in (96, 192, 336, 720) if (model, ds, pl) in best]
            cells.append(f"{sum(vals) / len(vals):.4f}" if len(vals) == 4 else "-")
            means.extend(vals)
        total = f"{sum(means) / len(means):.4f}" if means else "-"
        print(f"| {model} | " + " | ".join(cells) + f" | {total} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
