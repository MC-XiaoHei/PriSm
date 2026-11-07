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
BUILD = ROOT / "build" / "xilinx"
VIVADO = Path(r"D:\AMDDesignTools\2026.1\Vivado\bin\vivado.bat")

MODELS = ("prism", "fits", "pf", "tk", "tm")
DEVICES = {"gw5a25b": "xc7s25", "gw5a138c": "xc7s100",
           "xc7s25": "xc7s25", "xc7s100": "xc7s100"}


def devices() -> dict[str, dict[str, object]]:
    with (RTL / "syn" / "devices.toml").open("rb") as fh:
        return tomllib.load(fh)


def render(model: str, device: str, out: Path) -> str:
    dev = devices()[device]
    tpl = (RTL / "syn" / "xilinx" / f"{model}.tcl").read_text(encoding="utf-8")
    return (tpl.replace("@part@", str(dev["part"]))
               .replace("@syn@", (RTL / "syn" / "xilinx").as_posix())
               .replace("@rtl@", (RTL / "src").as_posix())
               .replace("@out@", out.as_posix()))


def run(model: str, device: str) -> dict[str, object]:
    out = BUILD / device / model
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    spec = SPECS[model]
    vectors(spec, out)
    for extra in spec.extra:
        src = ROOT / "rtl" / "src" / extra
        if src.exists():
            shutil.copy2(src, out / extra)

    (out / "run.tcl").write_text(render(model, device, out), encoding="utf-8")
    with (out / "run.log").open("w") as fh:
        subprocess.run([str(VIVADO), "-mode", "batch", "-source", "run.tcl",
                        "-nojournal", "-nolog"], cwd=out, stdout=fh,
                       stderr=subprocess.STDOUT, check=False)

    util = reports.vivado_util(out / "util.txt") if (out / "util.txt").exists() else {}
    wns = reports.vivado_wns(out / "timing.txt") if (out / "timing.txt").exists() else None
    power = reports.vivado_power(out / "power.txt") if (out / "power.txt").exists() else None
    period = float(cast(float, devices()[device].get("period", 20.0)))
    return {"model": model, "device": device, **util, "wns": wns,
            "fmax": 1000.0 / (period - wns) if wns is not None else None,
            "power": power}
