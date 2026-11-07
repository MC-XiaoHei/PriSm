from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="experiments/select.py")
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--model", default="PriSm")
    ap.add_argument("--field", default="k_v")
    ns = ap.parse_args(argv)

    per: dict[tuple[int, str, int], float] = {}
    for c in sorted(Path(ns.runs).rglob("config.json")):
        m = c.parent / "metrics" / "metrics.json"
        if not (m.exists() and c.exists()):
            continue
        cfg = json.loads(c.read_text(encoding="utf-8"))
        if cfg["run"]["model"] != ns.model:
            continue
        value = str(cfg["model"].get(ns.field))
        key = (cfg["model"]["pred_len"], cfg["data"]["data"], 0)
        mse = json.loads(m.read_text(encoding="utf-8"))["metrics"]["mse"]
        k2 = (cfg["model"]["pred_len"], value, 0)
        per[k2] = min(per.get(k2, float("inf")), mse)

    grouped: dict[str, dict[int, float]] = defaultdict(dict)
    for (pred, value, _), mse in per.items():
        grouped[value][pred] = mse
    for value in sorted(grouped, key=lambda v: sum(grouped[v].values()) / len(grouped[v])):
        vals = grouped[value]
        print(f"{ns.field}={value:>4s} horizons={len(vals)} mean={sum(vals.values()) / len(vals):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
