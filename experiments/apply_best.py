from __future__ import annotations

import argparse
import json
import pathlib
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[1]
KEYS = ("d_model", "e_layers", "k_l", "k_v", "norm_window", "r_z",
        "prism_act", "phase_len")


def best_runs(roots: list[pathlib.Path]) -> dict[tuple[str, str, int], tuple[float, dict]]:
    best: dict[tuple[str, str, int], tuple[float, dict]] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for config in root.rglob("config.json"):
            metrics = config.parent / "metrics" / "metrics.json"
            if not metrics.exists():
                continue
            try:
                payload = json.loads(config.read_text(encoding="utf-8"))
                value = json.loads(metrics.read_text(encoding="utf-8"))["metrics"]["mse"]
            except Exception:
                continue
            if payload["run"]["model"] != "PriSm":
                continue
            key = (payload["data"]["data"], int(payload["model"]["pred_len"]))
            if key not in best or value < best[key][0]:
                best[key] = (value, payload["model"])
    return best


def best_export(path: pathlib.Path) -> dict[tuple[str, str, int], tuple[float, dict]]:
    flat = json.loads(path.read_text(encoding="utf-8"))
    best: dict[tuple[str, str, int], tuple[float, dict]] = {}
    for key, entry in flat.items():
        dataset, pred = key.split("|")
        best[(dataset, int(pred))] = (entry["mse"], entry["model"])
    return best


def rewrite(preset: pathlib.Path, model: dict) -> bool:
    text = preset.read_text(encoding="utf-8")
    head, sep, body = text.partition("[model]\n")
    if not sep:
        return False
    lines = [line for line in body.splitlines() if line.strip()]
    values: dict[str, str] = {}
    order: list[str] = []
    for line in lines:
        key, _, value = line.partition("=")
        key = key.strip()
        values[key] = value.strip()
        order.append(key)
    for key in KEYS:
        if key not in model or model[key] is None:
            continue
        text_value = json.dumps(model[key]) if isinstance(model[key], str) else str(model[key])
        if key in values:
            values[key] = text_value
        else:
            values[key] = text_value
            order.append(key)
    rebuilt = "\n".join(f"{key} = {values[key]}" for key in order) + "\n"
    updated = head + sep + rebuilt
    if updated == text:
        return False
    preset.write_text(updated, encoding="utf-8")
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="experiments/apply_best.py")
    ap.add_argument("--presets", default=str(ROOT / "presets" / "PriSm"))
    ap.add_argument("--roots", default=str(ROOT / "runs"))
    ap.add_argument("--extra", default="", help="comma separated extra run roots")
    ap.add_argument("--best", default="", help="json export from the tuning grid")
    ap.add_argument("--dry-run", action="store_true")
    ns = ap.parse_args(argv)

    if ns.best:
        best = best_export(pathlib.Path(ns.best))
    else:
        roots = [pathlib.Path(ns.roots)] + [pathlib.Path(p) for p in ns.extra.split(",") if p]
        best = best_runs(roots)
    for preset in sorted(pathlib.Path(ns.presets).glob("*.toml")):
        spec = tomllib.loads(preset.read_text(encoding="utf-8"))
        key = (spec["data"]["data"], int(spec["model"]["pred_len"]))
        if key not in best:
            print(f"{preset.name}: no run")
            continue
        value, model = best[key]
        if ns.dry_run:
            print(f"{preset.name}: mse {value:.4f} {[ (k, model.get(k)) for k in KEYS ]}")
            continue
        changed = rewrite(preset, model)
        print(f"{preset.name}: mse {value:.4f} {'updated' if changed else 'unchanged'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
