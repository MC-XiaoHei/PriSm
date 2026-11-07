"""Export a trained PriSm checkpoint to the hex tables the RTL reads.

The layout is the one `prism.models.prism_fixed` consumes, so the exported
network can be checked against the integer reference and the RTL:
weights and biases are signed Q14 words, one per line, four hex digits.

Bank 0 of the lattice tables comes from block 0 and bank 1 from block 1; the
core holds both in one array and selects on the pass counter.
"""
from __future__ import annotations

from pathlib import Path

import torch

from prism.models import prism_fixed as fx


def _q(t: torch.Tensor) -> list[int]:
    """Signed Q14 with saturation; the RTL word is 16 bits."""
    vals = torch.round(t.detach().double() * (1 << fx.WQ)).to(torch.int64).flatten()
    return [max(-32768, min(32767, int(v))) for v in vals]


def tables(sd: dict[str, torch.Tensor]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}

    w_in = sd["in_proj.weight"]            # [D, LP]
    out["w_in"] = _q(w_in.t().reshape(-1))
    out["b_in"] = _q(sd["in_proj.bias"])

    w_out = sd["out_proj.weight"]          # [HP, D]
    out["w_out"] = _q(w_out.t().reshape(-1))
    out["b_out"] = _q(sd["out_proj.bias"])

    w_skip = sd["skip.weight"]             # [HP, LP]
    out["w_skip"] = _q(w_skip.t().reshape(-1))

    for bank, suffix in ((0, ""), (1, "_1")):
        b = f"blocks.{bank}"
        dw = sd[f"{b}.dw.weight"].reshape(fx.D, fx.KL)
        buf = [0] * (fx.D * fx.PAD_KL)
        for d in range(fx.D):
            row = _q(dw[d])
            buf[d * fx.PAD_KL:d * fx.PAD_KL + fx.KL] = row
        out[f"w_dw_pad{suffix}"] = buf

        down = sd[f"{b}.down.weight"].reshape(fx.RZ, fx.P)
        buf = [0] * (fx.RZ * fx.PAD_LP)
        for r in range(fx.RZ):
            buf[r * fx.PAD_LP:r * fx.PAD_LP + fx.P] = _q(down[r])
        out[f"w_v_pad{suffix}"] = buf

        out[f"w_u{suffix}"] = _q(sd[f"{b}.up.weight"].reshape(-1))
        out[f"w_p{suffix}"] = _q(sd[f"{b}.mix.weight"].reshape(-1))
        out[f"b_p{suffix}"] = _q(sd[f"{b}.mix.bias"])
    return out


def write_hex(hexes: dict[str, list[int]], out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, vals in hexes.items():
        path = out / f"{name}.hex"
        path.write_text('\n'.join('%04x' % (v & 0xFFFF) for v in vals) + '\n')
        paths.append(path)
    return paths


def export(checkpoint: Path, out: Path) -> dict[str, list[int]]:
    sd = torch.load(checkpoint, map_location='cpu')
    hexes = tables(sd)
    write_hex(hexes, out)
    return hexes


if __name__ == '__main__':
    import sys

    dest = Path(sys.argv[2] if len(sys.argv) > 2 else 'build/export')
    hexes = export(Path(sys.argv[1]), dest)
    print(f'{len(hexes)} tables -> {dest}')
