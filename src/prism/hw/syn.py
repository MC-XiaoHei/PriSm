from __future__ import annotations

import shutil
import subprocess
import tomllib
from pathlib import Path
from typing import cast

from prism.hw import reports
from prism.hw.sim import SPECS, vectors

ROOT = Path(__file__).resolve().parents[3]
RTL = ROOT / "rtl"
BUILD = ROOT / "build"
GW_SH = Path(r"C:\Gowin\Gowin_V1.9.12.03_x64\IDE\bin\gw_sh.exe")

MODELS = ("prism", "fits", "pf", "tk", "tm")


def devices() -> dict[str, dict[str, object]]:
    with (RTL / "syn" / "devices.toml").open("rb") as fh:
        return tomllib.load(fh)


def workdir(device: str, model: str) -> Path:
    return BUILD / "gowin" / device / model


def settings(model: str, device: str) -> dict[str, object]:
    """Device settings with any per-model override applied."""
    dev = dict(devices()[device])
    overrides = cast(dict[str, dict[str, object]], dev.pop("override", {}))
    dev.update(overrides.get(model, {}))
    return dev


def render(model: str, device: str) -> str:
    dev = settings(model, device)
    tpl = (RTL / "syn" / "gowin" / f"{model}.tcl").read_text(encoding="utf-8")
    text = (tpl
            .replace("@family@", str(dev["family"]))
            .replace("@part@", str(dev["part"]))
            .replace("@place@", str(dev.get("place", 0)))
            .replace("@route@", str(dev.get("route", 2)))
            .replace("@rtl@", str((RTL / "src")).replace("\\", "/"))
            .replace("@syn@", str((RTL / "syn" / "gowin")).replace("\\", "/")))
    return text


def run(model: str, device: str) -> dict[str, float | str | None]:
    d = workdir(device, model)
    d.mkdir(parents=True, exist_ok=True)
    (d / "syn.tcl").write_text(render(model, device), encoding="utf-8")
    vectors(SPECS[model], d)
    with (d / "run.log").open("w") as fh:
        subprocess.run([str(GW_SH), "syn.tcl"], cwd=d, stdout=fh,
                       stderr=subprocess.STDOUT, check=False)
    if "Placement and routing completed" not in (d / "run.log").read_text(errors="ignore"):
        return {"model": model, "device": device, "error": "routing did not finish"}

    base = d / "impl" / "pnr" / f"{model}"
    rpt = base.with_suffix(".rpt.txt")
    tr = Path(f"{base}_tr_content.html")
    pw = base.with_suffix(".power.html")
    if rpt.exists():
        shutil.copy2(rpt, d / "rpt.txt")
    if tr.exists():
        shutil.copy2(tr, d / "tr_content.html")
    if pw.exists():
        shutil.copy2(pw, d / "power.html")

    area = reports.gowin_area(d / "rpt.txt") if (d / "rpt.txt").exists() else {}
    fmax = reports.gowin_fmax(d / "tr_content.html") if (d / "tr_content.html").exists() else None
    power = reports.gowin_power(d / "power.html") if (d / "power.html").exists() else 0.0
    return {"model": model, "device": device, "fmax": fmax, "power": power, **area}
