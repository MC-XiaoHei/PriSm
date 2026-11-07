from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = Path("/home/amax/plnet")
DEFAULT_DST = ROOT / "runs"


def ledger_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def legacy_runs(src: Path) -> list[tuple[Path, dict]]:
    out = []
    for config in sorted(src.glob("*/*/*/config.json")):
        out.append((config, json.loads(config.read_text(encoding="utf-8"))))
    return out


def migrate(src: Path, dst: Path) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    rows = ledger_rows(src / "results.csv")
    by_cfg: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_cfg.setdefault(row["cfg"], []).append(row)

    copied = 0
    for config, payload in legacy_runs(src / "prism" / "runs"):
        metrics = config.parent / "metrics" / "metrics.json"
        if not metrics.exists():
            continue
        cfg = payload.get("ledger", {}).get("cfg", "")
        matched = [r for r in by_cfg.get(cfg, [])
                   if r["model"] == payload["run"]["model"]
                   and r["dataset"] == payload["data"]["data"]
                   and int(r["pred_len"]) == payload["model"]["pred_len"]]
        run = payload["run"]
        ident = "{}_sl96_pl{}_{}".format(
            payload["run"]["model"], payload["model"]["pred_len"],
            cfg or "legacy")
        target = dst / ident
        if target.exists():
            continue
        target.mkdir(parents=True)
        shutil.copy2(config, target / "config.json")
        (target / "metrics").mkdir()
        shutil.copy2(metrics, target / "metrics" / "metrics.json")
        (target / "ledger.json").write_text(
            json.dumps(matched[0] if matched else {}, indent=2), encoding="utf-8")
        copied += 1
    return copied


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="experiments/migrate.py")
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    ap.add_argument("--dst", default=str(DEFAULT_DST))
    ns = ap.parse_args(argv)
    print(migrate(Path(ns.src), Path(ns.dst)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
