from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_rows(runs: Path) -> dict[tuple[str, str, int], float]:
    best: dict[tuple[str, str, int], float] = {}
    for config in sorted(runs.glob("*/config.json")):
        metrics = config.parent / "metrics" / "metrics.json"
        if not metrics.exists():
            continue
        payload = json.loads(config.read_text(encoding="utf-8"))
        key = (payload["run"]["model"], payload["data"]["data"],
               int(payload["model"]["pred_len"]))
        value = json.loads(metrics.read_text(encoding="utf-8"))["metrics"]["mse"]
        if key not in best or value < best[key]:
            best[key] = value
    return best


def ledger_rows(path: Path) -> dict[tuple[str, str, int], float]:
    best: dict[tuple[str, str, int], float] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if not row["status"].startswith("ok"):
                continue
            key = (row["model"], row["dataset"], int(row["pred_len"]))
            value = float(row["mse"])
            if key not in best or value < best[key]:
                best[key] = value
    return best


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="experiments/compare.py")
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--ledger", default=str(ROOT / "experiments" / "ledger.csv"))
    ap.add_argument("--tol", type=float, default=1e-3)
    ns = ap.parse_args(argv)

    runs = run_rows(Path(ns.runs))
    ledger = ledger_rows(Path(ns.ledger))
    shared = sorted(set(runs) & set(ledger))
    worst = 0.0
    over = 0
    for key in shared:
        delta = abs(runs[key] - ledger[key])
        worst = max(worst, delta)
        over += delta > ns.tol
        if delta > ns.tol:
            print("{:14s} {:9s} {:4d} runs={:.6f} ledger={:.6f} d={:+.6f}".format(
                *key, runs[key], ledger[key], runs[key] - ledger[key]))
    print("shared={} over_tol={} worst={:.6f} runs_only={} ledger_only={}".format(
        len(shared), over, worst, len(set(runs) - set(ledger)),
        len(set(ledger) - set(runs))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
