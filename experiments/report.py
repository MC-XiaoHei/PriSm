from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "ECL",
            "Traffic", "Solar", "Exchange"]
MODELS = ["PriSm", "CycleNet", "DLinear", "FITS", "MixLinear", "PatchTST",
          "PhaseFormer", "SparseTSF", "TimeKAN", "TimeMixer", "iTransformer"]
PRED = (96, 192, 336, 720)
PEMS_DATASETS = ["PEMS03", "PEMS04", "PEMS07", "PEMS08"]
PEMS_PRED = (12, 24, 48, 96)
HORIZONS = {dataset: PRED for dataset in DATASETS}
HORIZONS.update({dataset: PEMS_PRED for dataset in PEMS_DATASETS})
# Period-folded models have no period that fits a 96-step PEMS window.
PEMS_MODELS = ["PriSm", "CycleNet", "DLinear", "FITS", "PatchTST",
               "PhaseFormer", "TimeKAN", "TimeMixer", "iTransformer"]
# iTransformer paper Table 1, mean MSE over the four horizons, lookback 96.
PEMS_REFERENCE = {
    "iTransformer": 0.119,
    "SCINet": 0.121,
    "TimesNet": 0.148,
    "Stationary": 0.151,
    "PatchTST": 0.217,
    "Crossformer": 0.220,
    "FEDformer": 0.224,
    "DLinear": 0.320,
    "TiDE": 0.375,
    "RLinear": 0.514,
    "Autoformer": 0.614,
}
STYLE = """<style>
body { font-family: system-ui, sans-serif; margin: 2rem; }
table { border-collapse: collapse; font-size: 13px; table-layout: fixed; }
th, td { border: 1px solid #ccc; padding: 4px 6px; text-align: center;
         width: 92px; box-sizing: border-box; }
th { background: #f4f4f4; }
th.dataset, td.dataset { width: 78px; background: #fafafa; vertical-align: middle; }
th.horizon, td.horizon { width: 62px; background: #fafafa; }
.best { color: #d00; font-weight: 700; }
.second { color: #06c; text-decoration: underline; }
.third { color: #080; font-weight: 700; font-style: italic; }
</style>
"""


def cells(roots: list[Path]) -> dict[tuple[str, str, int], float]:
    best: dict[tuple[str, str, int], float] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for config in root.rglob("config.json"):
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


def pick(best: dict[tuple[str, str, int], float], model: str,
         dataset: str, pred: int) -> float | None:
    if model != "PriSm":
        return best.get((model, dataset, pred))
    values = [value for (name, ds, pl), value in best.items()
              if name.startswith("PriSm") and ds == dataset and pl == pred]
    return min(values) if values else None


def rows_of(best: dict[tuple[str, str, int], float]
            ) -> tuple[list[tuple[str, list[tuple[str, dict[str, float | None]]]]],
                       dict[str, float | None]]:
    groups: list[tuple[str, list[tuple[str, dict[str, float | None]]]]] = []
    flat: list[dict[str, float | None]] = []
    for dataset in DATASETS + PEMS_DATASETS:
        models = MODELS if dataset not in PEMS_DATASETS else PEMS_MODELS
        body: list[tuple[str, dict[str, float | None]]] = []
        for pred in HORIZONS[dataset]:
            row = {model: pick(best, model, dataset, pred) for model in models}
            body.append((str(pred), row))
            flat.append(row)
        groups.append((dataset, body))
    overall: dict[str, float | None] = {}
    scored = [row for dataset, body in groups if dataset not in PEMS_DATASETS
              for _, row in body]
    for model in MODELS:
        values = [row[model] for row in scored if row.get(model) is not None]
        overall[model] = sum(values) / len(values) if values else None
    return groups, overall


def render(groups: list[tuple[str, list[tuple[str, dict[str, float | None]]]]]) -> str:
    out = ["<!doctype html>", '<html lang="en"><head><meta charset="utf-8">',
           "<title>PriSm main table</title>", STYLE, "</head><body>",
           "<h1>Main table</h1>",
           "<p>MSE per horizon; each cell is the best over lr and seed. "
           "The PriSm column is also the best over the tuned structural grid. "
           "Red bold is first, blue underlined is second, "
           "green bold italic is third. "
           "Dashes are cells the protocol leaves empty: Weather and Solar have no "
           "period-folded baseline, and the PEMS block leaves out the period-folded "
           "models because no period fits a 96-step window.</p>",
           "<table>",
           '<tr><th class="dataset">dataset</th><th class="horizon">horizon</th>' +
           "".join(f"<th>{model}</th>" for model in MODELS) + "</tr>"]
    for dataset, body in groups:
        models = MODELS if dataset not in PEMS_DATASETS else PEMS_MODELS
        for index, (horizon, row) in enumerate(body):
            order = [name for name in
                     sorted(models, key=lambda name: (row.get(name) is None,
                                                      row.get(name) or 0.0))]
            head = ""
            if index == 0:
                head = (f'<td class="dataset" rowspan="{len(body)}">'
                        f"{dataset}</td>")
            html = []
            for model in MODELS:
                value = row.get(model)
                if value is None:
                    html.append("<td>-</td>")
                    continue
                mark = ""
                if order and order[0] == model:
                    mark = ' class="best"'
                elif len(order) > 1 and order[1] == model:
                    mark = ' class="second"'
                elif len(order) > 2 and order[2] == model:
                    mark = ' class="third"'
                html.append(f"<td{mark}>{value:.4f}</td>")
            out.append(f'<tr>{head}<td class="horizon">{horizon}</td>' +
                       "".join(html) + "</tr>")
    out += ["</table>"]
    out += ["</body></html>"]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="experiments/report.py")
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--extra", default="", help="comma separated extra run roots")
    ap.add_argument("--best", default="", help="json dump of model|dataset|pred -> mse")
    ap.add_argument("--out", default=str(ROOT / "build" / "main_table.html"))
    ap.add_argument("--dump-best", default="", help="write the flat best map here")
    ns = ap.parse_args(argv)
    if ns.best:
        flat = json.loads(Path(ns.best).read_text(encoding="utf-8"))
        best = {}
        for key, value in flat.items():
            model, dataset, pred = key.split("|")
            best[(model, dataset, int(pred))] = value
    else:
        best = cells([Path(ns.runs)] + [Path(p) for p in ns.extra.split(",") if p])
    if ns.dump_best:
        flat = {f"{model}|{dataset}|{pred}": value
                for (model, dataset, pred), value in best.items()}
        Path(ns.dump_best).write_text(json.dumps(flat, indent=2, sort_keys=True),
                                      encoding="utf-8")
    groups, overall = rows_of(best)
    Path(ns.out).write_text(render(groups), encoding="utf-8")
    scored = sorted((overall[m], m) for m in MODELS if overall[m] is not None)
    place = [m for _, m in scored].index("PriSm") + 1
    print(f"{'all':9s} mean  PriSm {place}/{len(scored)}  {overall['PriSm']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
