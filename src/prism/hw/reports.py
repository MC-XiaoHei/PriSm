from __future__ import annotations

import re
from pathlib import Path


def gowin_area(rpt: Path) -> dict[str, float]:
    text = rpt.read_text(encoding="utf-8", errors="ignore")
    out: dict[str, float] = {}
    for key in ("Logic", "Register", "BSRAM", "DSP"):
        m = re.search(rf"^\s*{key}\s*\|\s*([0-9.]+)/([0-9]+)", text, re.M)
        if m:
            out[key] = float(m.group(1))
            out[f"{key}_max"] = float(m.group(2))
    return out


def gowin_fmax(html: Path) -> float | None:
    text = html.read_text(encoding="utf-8", errors="ignore")
    i = text.find("Max Frequency Summary")
    if i < 0:
        return None
    vals = re.findall(r"([0-9]+\.[0-9]+)\(MHz\)", text[i:i + 4000])
    return float(vals[1]) if len(vals) > 1 else None


def gowin_power(html: Path) -> float:
    text = html.read_text(encoding="utf-8", errors="ignore")
    i = text.find("Total Power (mW)")
    mw = re.search(r"([0-9]+\.[0-9]+)", text[i:i + 400]) if i >= 0 else None
    return float(mw.group(1)) if mw else 0.0


def vivado_util(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    out: dict[str, float] = {}
    for label, key in (("Slice LUTs", "LUT"), ("Slice Registers", "FF"),
                       ("Block RAM Tile", "BRAM"), ("DSPs", "DSP")):
        m = re.search(rf"\|\s*{label}\*?\s*\|\s*([0-9.]+)\s*\|", text)
        if m:
            out[key] = float(m.group(1))
    return out


def vivado_wns(path: Path) -> float | None:
    text = path.read_text(encoding="utf-8", errors="ignore")
    i = text.find("Design Timing Summary")
    if i < 0:
        return None
    m = re.search(r"\n\s*(-?[0-9]+\.[0-9]+)\s+(-?[0-9]+\.[0-9]+)", text[i:i + 2000])
    return float(m.group(1)) if m else None


def vivado_power(path: Path) -> float:
    text = path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"Total On-Chip Power \(W\)\s*\|\s*([0-9.]+)", text)
    return float(m.group(1)) * 1000 if m else 0.0
