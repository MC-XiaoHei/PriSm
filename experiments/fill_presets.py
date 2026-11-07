from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
PRESETS = ROOT / "presets"


def field_names(model: str) -> list[str]:
    from prism.models.base import models

    return [item.name for item in dataclasses.fields(models.get(model).config_type())]


def defaults(model: str) -> dict[str, object]:
    from prism.models.base import models

    return {item.name: item.default
            for item in dataclasses.fields(models.get(model).config_type())}


def load_runs() -> list[tuple[float, pathlib.Path, dict]]:
    out = []
    for config in RUNS.glob("*/config.json"):
        metrics = config.parent / "metrics" / "metrics.json"
        if not metrics.exists():
            continue
        payload = json.loads(config.read_text(encoding="utf-8"))
        value = json.loads(metrics.read_text(encoding="utf-8"))["metrics"]["mse"]
        out.append((value, config.parent, payload))
    return out


def best_run(runs: list[tuple[float, pathlib.Path, dict]], model: str,
             dataset: str, pred: int) -> tuple[float, pathlib.Path, dict] | None:
    if model == "PriSm":
        pool = [item for item in runs
                if item[2]["run"]["model"].startswith("PriSm")
                and item[2]["data"]["data"] == dataset
                and int(item[2]["model"]["pred_len"]) == pred]
    else:
        pool = [item for item in runs
                if item[2]["run"]["model"] == model
                and item[2]["data"]["data"] == dataset
                and int(item[2]["model"]["pred_len"]) == pred]
    return min(pool, key=lambda item: item[0]) if pool else None


def resolve(model: str, payload: dict | None, existing: dict) -> tuple[dict, str]:
    names = field_names(model)
    base = defaults(model)
    recorded = dict(payload["model"]) if payload else {}
    out: dict[str, object] = {}
    origin = "run" if recorded else "default"
    for name in names:
        if name in recorded:
            out[name] = recorded[name]
        elif name in existing:
            out[name] = existing[name]
            if origin == "run":
                origin = "mixed"
        else:
            out[name] = base[name]
    return out, origin


def format_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f'"{value}"'
    return str(value)


def render(spec: dict, section: dict) -> str:
    lines: list[str] = []
    for name in ("run", "data"):
        if name not in spec:
            continue
        lines.append(f"[{name}]")
        for key, value in spec[name].items():
            lines.append(f"{key} = {format_value(value)}")
        lines.append("")
    lines.append("[model]")
    for key, value in section.items():
        lines.append(f"{key} = {format_value(value)}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="experiments/fill_presets.py")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--show", type=int, default=0)
    ns = ap.parse_args(argv)

    runs = load_runs()
    tally: dict[str, int] = {}
    shown = 0
    for preset in sorted(PRESETS.glob("*/*.toml")):
        spec = tomllib.loads(preset.read_text(encoding="utf-8"))
        model = preset.parent.name
        dataset = spec["data"]["data"]
        pred = int(spec["model"]["pred_len"])
        hit = best_run(runs, model, dataset, pred)
        payload = hit[2] if hit else None
        section, origin = resolve(model, payload, spec["model"])
        tally[origin] = tally.get(origin, 0) + 1
        if ns.show and shown < ns.show:
            print(f"== {preset.relative_to(ROOT)}  {origin}")
            print(f"   old {spec['model']}")
            print(f"   new {section}")
            shown += 1
        if ns.apply:
            preset.write_text(render(spec, section), encoding="utf-8")
    print(tally, "apply", ns.apply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
