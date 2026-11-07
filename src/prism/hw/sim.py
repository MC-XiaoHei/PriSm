from __future__ import annotations

import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RTL = ROOT / "rtl"
BUILD = ROOT / "build" / "sim"


@dataclass(frozen=True)
class Spec:
    model: str
    sources: tuple[str, ...]
    tb: str
    gen: str
    top: str
    extra: tuple[str, ...] = ()


SPECS = {
    s.model: s
    for s in (
        Spec("prism", ("prism_top.v", "lattice_w.v", "dpram.v"), "tb_full.v",
             "prism.models.prism_fixed", "tb_full"),
        Spec("fits", ("fits_top.v",), "tb_fits.v", "prism.models.fits_fixed", "tb_fits"),
        Spec("pf", ("phaseformer_top.v",), "tb_pf.v",
             "prism.models.phaseformer_fixed", "tb_pf"),
        Spec("tk", ("timekan_top.v",), "tb_tk.v",
             "prism.models.timekan_fixed", "tb_tk"),
        Spec("tm", ("timemixer_top.v", "dpram.v"), "tb_tm.v",
             "prism.models.timemixer_fixed", "tb_tm"),
    )
}


def wsl(path: Path) -> str:
    text = str(path)
    if len(text) > 1 and text[1] == ":":
        drive, rest = text[0].lower(), text[2:].replace("\\", "/")
        return f"/mnt/{drive}{rest}"
    return text.replace("\\", "/")


def prepare(spec: Spec) -> Path:
    d = BUILD / spec.model
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for name in spec.sources:
        shutil.copy2(RTL / "src" / name, d / name)
    shutil.copy2(RTL / "tb" / spec.tb, d / spec.tb)
    for name in spec.extra:
        shutil.copy2(RTL / "src" / name, d / name)
    return d


def vectors(spec: Spec, d: Path) -> None:
    subprocess.run([sys.executable, "-m", spec.gen], cwd=d, check=True,
                   stdout=subprocess.DEVNULL)


def compile_(spec: Spec, d: Path) -> Path:
    sources = " ".join(spec.sources) + f" {spec.tb}"
    cmd = (
        f"cd {wsl(d)} && verilator --binary --timing -Wno-fatal "
        f"--top-module {spec.top} --Mdir obj {sources} -o Vsim"
    )
    subprocess.run(["wsl", "-e", "bash", "-lc", cmd], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return d / "obj" / "Vsim"


def run(spec: Spec, d: Path, exe: Path) -> tuple[int, int]:
    out = subprocess.run(["wsl", "-e", "bash", "-lc", f"cd {wsl(d)} && ./obj/Vsim"],
                         capture_output=True, text=True, check=True).stdout
    cycles = re.search(r"cycles=(\d+)", out)
    mism = re.search(r"mismatches=(\d+)", out)
    if not cycles or not mism:
        print(out[-400:])
        raise RuntimeError(f"{spec.model}: no verdict")
    return int(cycles.group(1)), int(mism.group(1))


def check(names: list[str]) -> int:
    failed = 0
    for name in names:
        spec = SPECS[name]
        d = prepare(spec)
        vectors(spec, d)
        exe = compile_(spec, d)
        cycles, mism = run(spec, d, exe)
        print(f"{name:12s} cycles={cycles:<8d} mismatches={mism}")
        failed += mism != 0
    return failed
