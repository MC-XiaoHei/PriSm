from __future__ import annotations

import argparse
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRESETS = ROOT / "presets"

DRIVER = '''\
import os, subprocess, sys, pathlib
free = {gpus!r}
queue = sorted(pathlib.Path("presets").glob("*/*.toml"))
if {models!r}:
    queue = [p for p in queue if p.parent.name in {models!r}]
if {datasets!r}:
    queue = [p for p in queue if p.stem.rsplit("_", 1)[0].upper() in {datasets!r}]
seeds = {seeds!r}
queue = [(p, s) for p in queue for s in seeds] if seeds else [(p, None) for p in queue]
pathlib.Path("sweep_logs").mkdir(exist_ok=True)
procs = {{}}
while queue or procs:
    while queue and free:
        p, seed = queue.pop(0)
        g = free.pop(0)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=g, OMP_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
                   PRISM_SAVE_PREDICTIONS="0")
        tag = f"{{p.parent.name}}_{{p.stem}}" + (f"_s{{seed}}" if seed else "")
        log = open(f"sweep_logs/{{tag}}.log", "w")
        print("start", p, seed or "", "gpu", g, flush=True)
        argv = [sys.executable, "-m", "prism", "train", "--preset", str(p), "--skip-existing"]
        if seed:
            argv += ["--run.seed", str(seed)]
        procs[subprocess.Popen(argv, env=env, stdout=log,
                               stderr=subprocess.STDOUT)] = (p, g)
    for proc, (p, g) in list(procs.items()):
        if proc.poll() is not None:
            print(p, "gpu", g, "exit", proc.returncode, flush=True)
            free.append(g)
            del procs[proc]
            break
'''


@dataclass(frozen=True)
class Cell:
    model: str
    data: str
    pred: int
    preset: Path


def cells() -> list[Cell]:
    out: list[Cell] = []
    for path in sorted(PRESETS.glob("*/*.toml")):
        with path.open("rb") as fh:
            spec = tomllib.load(fh)
        out.append(Cell(spec["run"]["model"], spec["data"]["data"],
                        spec["model"]["pred_len"], path))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="experiments/sweep.py")
    ap.add_argument("--models", default="")
    ap.add_argument("--datasets", default="")
    ap.add_argument("--pred-len", type=int, default=0)
    ap.add_argument("--seeds", default="")
    ap.add_argument("--dry-run", action="store_true")
    ns = ap.parse_args(argv)

    models = [m for m in ns.models.split(",") if m]
    datasets = [d for d in ns.datasets.split(",") if d]
    picked = [c for c in cells()
              if (not models or c.preset.parent.name in models)
              and (not datasets or c.data in datasets)
              and (not ns.pred_len or c.pred == ns.pred_len)]
    if ns.dry_run:
        for c in picked:
            print(f"{c.model:14s} {c.data:9s} {c.pred:4d}  {c.preset.relative_to(ROOT)}")
        print(f"{len(picked)} cells")
        return 0

    host = tomllib.loads((ROOT / "experiments" / "host.toml").read_text())
    driver = DRIVER.format(
        gpus=[g.strip() for g in str(host["gpus"]).split(",")],
        models=models, datasets=datasets,
        seeds=[int(t) for t in ns.seeds.split(",") if t])
    script = (f"cd {host['root']} && cat > /tmp/prism_sweep.py << 'PYEOF'\n{driver}\nPYEOF\n"
              f"nohup {host['python']} /tmp/prism_sweep.py > sweep.log 2>&1 &")
    return subprocess.call(["ssh", "-p", str(host["port"]), str(host["host"]), script])


if __name__ == "__main__":
    raise SystemExit(main())
