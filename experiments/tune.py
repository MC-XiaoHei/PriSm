from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
HOST = ROOT / "experiments" / "host.toml"


def cards(allowed: list[str]) -> list[str]:
    used = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used",
                           "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout
    taken = {x.split(",")[0].strip(): int(x.split(",")[1])
             for x in used.strip().splitlines()}
    return [g for g in allowed if taken.get(g, 0) < 200]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="experiments/tune.py")
    ap.add_argument("--presets", required=True, help="comma separated preset paths")
    ap.add_argument("--out", required=True)
    ap.add_argument("--configs", required=True, help="json file holding a list of override dicts")
    ap.add_argument("--gpus", default="4,5,6,7")
    ns = ap.parse_args(argv)

    configs = json.loads(pathlib.Path(ns.configs).read_text(encoding="utf-8"))
    allowed = [g.strip() for g in ns.gpus.split(",") if g.strip()]
    jobs = []
    for preset in [p for p in ns.presets.split(",") if p]:
        stem = pathlib.Path(preset).stem
        parent = pathlib.Path(preset).parent.name
        for index, overrides in enumerate(configs):
            jobs.append((f"{parent}_{stem}_c{index:03d}", preset, overrides))

    runs = pathlib.Path(ns.out)
    runs.mkdir(parents=True, exist_ok=True)
    procs: dict[subprocess.Popen[bytes], tuple[str, str]] = {}
    busy: set[str] = set()
    queue = list(jobs)
    while queue or procs:
        for gpu in cards(allowed):
            if not queue or gpu in busy:
                continue
            tag, preset, overrides = queue.pop(0)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu,
                       OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
            cmd = [sys.executable, "-m", "prism", "train", "--preset", preset,
                   "--run.out_dir", f"{runs}/{tag}"]
            for key, value in overrides.items():
                cmd += [f"--{key}", str(value)]
            procs[subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.STDOUT)] = (tag, gpu)
            busy.add(gpu)
        done = [p for p in procs if p.poll() is not None]
        for proc in done:
            tag, gpu = procs.pop(proc)
            busy.discard(gpu)
            print(tag, "exit", proc.returncode, flush=True)
        if not done:
            time.sleep(15)
    print("drained", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
