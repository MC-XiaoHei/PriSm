from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from dataclasses import fields
from pathlib import Path
from typing import Any, TypeVar, cast

from prism.config import DataConfig, Resolved, RunConfig, align_freq
from prism.data import prepare
from prism.models.base import models
from prism.presets import EMPTY, EMPTY_SECTIONS, apply, load
from prism.run import RunDir, evaluate_run, train

T = TypeVar("T")


def _add_group(parser: argparse.ArgumentParser, prefix: str, cls: type[Any], instance: Any) -> None:
    for item in fields(cls):
        default = getattr(instance, item.name)
        help_text = str(item.metadata.get("help", ""))
        flag = f"--{prefix}.{item.name}"
        if isinstance(default, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=default, help=help_text)
        elif isinstance(default, int):
            parser.add_argument(flag, type=int, default=default, help=help_text)
        elif isinstance(default, float):
            parser.add_argument(flag, type=float, default=default, help=help_text)
        else:
            parser.add_argument(flag, type=str, default=default, help=help_text)


def _build(cls: type[T], ns: argparse.Namespace, prefix: str) -> T:
    kwargs: dict[str, Any] = {
        item.name: getattr(ns, f"{prefix}.{item.name}") for item in fields(cast(Any, cls))
    }
    return cls(**kwargs)


def _do_train(tokens: list[str]) -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--run.model", dest="model", default=RunConfig().model)
    pre.add_argument("--preset", dest="preset", default=None)
    known, _ = pre.parse_known_args(tokens)

    preset: Mapping[str, Mapping[str, Any]] = (
        load(known.preset) if known.preset else EMPTY_SECTIONS
    )
    explicit_model = any(
        token == "--run.model" or token.startswith("--run.model=") for token in tokens
    )
    run_section = preset.get("run", EMPTY)
    model_name = known.model if explicit_model else str(run_section.get("model", known.model))
    model_cls = models.get(model_name)
    config_cls = model_cls.config_type()

    run_defaults = apply(RunConfig, RunConfig(), run_section)
    data_defaults = apply(DataConfig, DataConfig(), preset.get("data", EMPTY))
    model_defaults = apply(config_cls, config_cls(), preset.get("model", EMPTY))

    parser = argparse.ArgumentParser(prog="prism train")
    parser.add_argument("--preset", default=None, help="preset TOML path")
    parser.add_argument("--skip-existing", action="store_true")
    _add_group(parser, "run", RunConfig, run_defaults)
    _add_group(parser, "data", DataConfig, data_defaults)
    _add_group(parser, "model", config_cls, model_defaults)
    ns = parser.parse_args(tokens)

    run = _build(RunConfig, ns, "run")
    data = _build(DataConfig, ns, "data")
    model_config = align_freq(_build(config_cls, ns, "model"), data)
    dims, datasets = prepare(data, model_config)
    resolved = Resolved(run=run, model=model_config, data=data, dims=dims)
    rundir = train(resolved, datasets, run.out_dir, reuse=ns.skip_existing)
    print(f"run dir {rundir.path}")
    return 0


def _do_eval(tokens: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="prism eval")
    parser.add_argument("--run", required=True)
    parser.add_argument("--metrics", default="")
    ns = parser.parse_args(tokens)

    target = Path(ns.run)
    rundir = RunDir(target.parent, target.name)
    names = tuple(name for name in str(ns.metrics).split(",") if name) or None
    results = evaluate_run(rundir, names)
    for name in sorted(results):
        print(f"{name} {results[name]:.6f}")
    return 0


def _do_hw(tokens: list[str]) -> int:
    from prism.hw import sim

    parser = argparse.ArgumentParser(prog="prism hw")
    sub = parser.add_subparsers(dest="action", required=True)
    p_sim = sub.add_parser("sim")
    p_sim.add_argument("models", nargs="*", default=[])
    p_sim.add_argument("--all", action="store_true")
    p_syn = sub.add_parser("syn")
    p_syn.add_argument("--device", required=True)
    p_syn.add_argument("--models", default="", help="comma separated, empty for all")
    p_x = sub.add_parser("xsyn")
    p_x.add_argument("--device", default="xc7s25")
    p_x.add_argument("--models", default="")
    sub.add_parser("table")
    ns = parser.parse_args(tokens)
    if ns.action == "sim":
        names = list(sim.SPECS) if ns.all or not ns.models else ns.models
        return 1 if sim.check(names) else 0
    if ns.action == "xsyn":
        from prism.hw import xsyn

        models = [m.strip() for m in ns.models.split(",")] if ns.models else list(xsyn.MODELS)
        for model in models:
            row = xsyn.run(model, ns.device)
            print(" ".join(f"{k}={v}" for k, v in row.items() if v is not None))
        return 0
    if ns.action == "table":
        from prism.hw import table

        print(table.render(table.rows()))
        return 0
    if ns.action == "syn":
        from prism.hw import syn

        models = [m.strip() for m in ns.models.split(",")] if ns.models else list(syn.MODELS)
        for model in models:
            row = syn.run(model, ns.device)
            print(" ".join(f"{k}={v}" for k, v in row.items() if v is not None))
        return 0
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    if not tokens or tokens[0] in ("-h", "--help"):
        print("usage: prism train [--preset FILE] [--run.* --data.* --model.*] | prism eval --run DIR | prism hw sim")
        return 0
    command, rest = tokens[0], tokens[1:]
    if command == "train":
        return _do_train(rest)
    if command == "eval":
        return _do_eval(rest)
    if command == "hw":
        return _do_hw(rest)
    print(f"unknown subcommand {command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
