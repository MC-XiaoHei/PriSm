from __future__ import annotations

import pathlib
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[1]
PRESETS = ROOT / "presets"
BASELINES = ["CycleNet", "DLinear", "FITS", "MixLinear", "PatchTST",
             "PhaseFormer", "SparseTSF", "TimeKAN", "TimeMixer", "iTransformer"]
DATASETS = {"pems03": "PEMS03", "pems04": "PEMS04",
            "pems07": "PEMS07", "pems08": "PEMS08"}
# Period-folded models need a period that divides both the window and the
# horizon; PEMS is five-minute traffic, so no period fits inside 96 steps.
SKIP = {"SparseTSF", "MixLinear"}
PRED = (12, 24, 48, 96)
# PEMS07 has 883 channels; the official large batches do not fit on a 24 GB card
# once the channel count is that high, so every baseline is capped here.
BATCH_CAP = 32


def dump(spec: dict[str, dict[str, object]], path: pathlib.Path) -> None:
    lines = []
    for section in ("run", "data", "model"):
        lines.append(f"[{section}]")
        for key, value in spec[section].items():
            if isinstance(value, str):
                lines.append(f'{key} = "{value}"')
            else:
                lines.append(f"{key} = {value}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    written = 0
    for model in BASELINES:
        if model in SKIP:
            continue
        with (PRESETS / model / "etth1_96.toml").open("rb") as fh:
            base = tomllib.load(fh)
        for stem, name in DATASETS.items():
            for pred in PRED:
                spec = {section: dict(base[section]) for section in ("run", "data", "model")}
                spec["data"].update(data=name, root_path="./dataset/PEMS",
                                    data_path=f"{name}.npz", freq="h")
                spec["model"]["pred_len"] = pred
                spec["run"]["batch_size"] = min(int(spec["run"]["batch_size"]), BATCH_CAP)
                dump(spec, PRESETS / model / f"{stem}_{pred}.toml")
                written += 1
    print("written", written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
