from __future__ import annotations

import re
from pathlib import Path

from prism.hw import reports

ROOT = Path(__file__).resolve().parents[3]
BUILD = ROOT / "build"

CYCLES = {"prism": 4760, "fits": 8091, "pf": 200180, "tk": 924025, "tm": 109976}
NAMES = {"prism": "PriSm", "fits": "FITS", "pf": "PhaseFormer",
         "tk": "TimeKAN", "tm": "TimeMixer"}
GOWIN = {"prism": "gw5a25b", "fits": "gw5a25b", "pf": "gw5a25b",
         "tk": "gw5a138c", "tm": "gw5a138c"}
XILINX = {"prism": "xc7s25", "fits": "xc7s25", "pf": "xc7s25",
          "tk": "xc7s100", "tm": "xc7s100"}
BLOCKED: dict[tuple[str, str], str] = {}


def gowin_row(model: str) -> dict[str, object]:
    device = GOWIN[model]
    d = BUILD / "gowin" / device / model
    row: dict[str, object] = {"model": NAMES[model], "device": device}
    if not (d / "rpt.txt").exists():
        return {**row, "state": BLOCKED.get((model, device), "missing")}
    row.update(reports.gowin_area(d / "rpt.txt"))
    fmax = reports.gowin_fmax(d / "tr_content.html")
    power = reports.gowin_power(d / "power.html")
    row.update(fmax=fmax, power=power or None)
    if fmax:
        row["latency_us"] = CYCLES[model] / fmax
        row["energy_uj"] = (power or 0) * row["latency_us"] / 1000.0
    return row


def xilinx_row(model: str) -> dict[str, object]:
    device = XILINX[model]
    d = BUILD / "xilinx" / device / model
    row: dict[str, object] = {"model": NAMES[model], "device": device}
    if not (d / "util.txt").exists():
        return {**row, "state": "missing"}
    row.update(reports.vivado_util(d / "util.txt"))
    wns = reports.vivado_wns(d / "timing.txt")
    found = re.search(r"-period ([0-9.]+)",
                      (ROOT / "rtl" / "syn" / "xilinx" / f"{model}.xdc").read_text(encoding="utf-8"))
    if found is None:
        raise RuntimeError(f"{model}: no period in xdc")
    period = float(found.group(1))
    fmax = 1000.0 / (period - wns) if wns is not None else None
    power = reports.vivado_power(d / "power.txt") if (d / "power.txt").exists() else None
    row.update(fmax=fmax, power=power)
    if fmax:
        row["latency_us"] = CYCLES[model] / fmax
        row["energy_uj"] = (power or 0) * row["latency_us"] / 1000.0
    return row


def rows() -> list[dict[str, object]]:
    order = ("prism", "fits", "pf", "tk", "tm")
    return [f(model) for model in order for f in (gowin_row, xilinx_row)]


def fmt(value: object, unit: str = "", nd: int = 0) -> str:
    if value is None:
        return "pending"
    if isinstance(value, float):
        return f"{value:.{nd}f}{unit}"
    return f"{value}{unit}"


def render(items: list[dict[str, object]]) -> str:
    out = ["| model | device | logic | ff | bram | dsp | fmax | power | latency | energy | activity |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in items:
        if r.get("state"):
            out.append("| {} | {} | {} |".format(r["model"], r["device"], r["state"]))
            continue
        key = "Logic" if "Logic" in r else "LUT"
        out.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            r["model"], r["device"],
            fmt(r.get(key)), fmt(r.get("Register") or r.get("FF")),
            fmt(r.get("BSRAM") or r.get("BRAM")), fmt(r.get("DSP"), "", 1),
            fmt(r.get("fmax"), " MHz", 1), fmt(r.get("power"), " mW", 1),
            fmt(r.get("latency_us"), " us", 2), fmt(r.get("energy_uj"), " uJ", 2),
            "vectorless"))
    return "\n".join(out)
