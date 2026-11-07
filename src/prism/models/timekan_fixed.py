"""TimeKAN reference, float first, integer later.

Structure copied from the official model as it runs in our experiments:
lookback 96, pred 96, down_sampling_layers 1, down_sampling_window 2,
d_model 32, one encoder layer, begin_order 0. Channel independent, so the
whole model runs once per variable.

The two levels are 96 and 48 samples. Each level is instance-normalised, run
through the value embedding (a length-3 convolution), split into a low and a
high band by frequency interpolation, mixed by a Chebyshev KAN plus a depthwise
convolution, summed back, projected from 32 channels to one, and finally mapped
from 96 steps to 96 steps by a dense matrix.
"""
import numpy as np

LOOKBACK = 96
PRED = 96
DM = 32
DSW = 2
DSL = 1
ORDER = 0
SEED = 7


def rfft_interp(x, target):
    """Zero-pad the spectrum and scale, as FrequencyDecomp does."""
    src = x.shape[-1]
    spec = np.fft.rfft(x, axis=-1)
    out = np.zeros(x.shape[:-1] + (target // 2 + 1,), dtype=np.complex128)
    out[..., : spec.shape[-1]] = spec
    return np.fft.irfft(out, n=target, axis=-1) * (src / target)


def cheby(x, coeff):
    """coeff[i][o][k]; y[o] = sum_i sum_k coeff[i][o][k] cos(k acos(tanh(tanh(x_i))))."""
    u = np.tanh(np.tanh(x))
    a = np.arccos(np.clip(u, -1.0, 1.0))
    deg = coeff.shape[2]
    t = np.stack([np.cos(k * a) for k in range(deg)], axis=-1)  # (in, deg)
    return np.einsum("ik,iok->o", t, coeff)


def m_kan(x, coeff, conv):
    """x: (steps, DM). Chebyshev channel mixer plus a depthwise length-3 conv."""
    steps = x.shape[0]
    y1 = np.stack([cheby(x[s], coeff) for s in range(steps)])
    pad = np.pad(x, ((1, 1), (0, 0)))
    y2 = np.zeros_like(x)
    for k in range(3):
        y2 += pad[k : k + steps] * conv[:, k]
    return y1 + y2


def inst_norm(x):
    m = x.mean()
    s = x.std()
    return (x - m) / s, m, s


def build_weights(seed=SEED, steps0=LOOKBACK, steps1=LOOKBACK // DSW):
    rng = np.random.default_rng(seed)
    w = {}
    w["emb"] = rng.normal(0, 0.1, (DM, 1, 3))
    for tag, steps, order in (("low", steps1, ORDER), ("high", steps0, ORDER + 1)):
        w[f"kan_{tag}"] = rng.normal(0, 0.05, (DM, DM, order + 1))
        w[f"conv_{tag}"] = rng.normal(0, 0.1, (DM, 3))
    w["proj"] = rng.normal(0, 0.1, (1, DM))
    w["proj_b"] = rng.normal(0, 0.1, (1,))
    w["pred"] = rng.normal(0, 0.02, (PRED, LOOKBACK))
    w["pred_b"] = rng.normal(0, 0.02, (PRED,))
    return w


def embed(x, w):
    """Value embedding: Conv1d(1, DM, 3, padding=1, circular) over time."""
    pad = np.concatenate([x[-1:], x, x[:1]])
    out = np.zeros((x.shape[0], DM))
    for k in range(3):
        out += pad[k : k + x.shape[0]][:, None] * w["emb"][:, 0, k][None, :]
    return out


def forward(x, w):
    """x: (LOOKBACK,) one variable, already normalised by the caller."""
    levels = [x]
    for _ in range(DSL):
        n = levels[-1].shape[0] // DSW
        levels.append(levels[-1][: n * DSW].reshape(n, DSW).mean(axis=1))
    embs = []
    stats = []
    for lv in levels:
        n, m, s = inst_norm(lv)
        stats.append((m, s))
        embs.append(embed(n, w))
    # FrequencyDecomp: the coarse level is the low band, the difference is high
    low = embs[1]
    low_up = rfft_interp(low.T, embs[0].shape[0]).T
    high = embs[0] - low_up
    # FrequencyMixing
    low_mix = m_kan(low, w["kan_low"], w["conv_low"])
    high_mix = m_kan(high, w["kan_high"], w["conv_high"])
    high_out = high_mix + rfft_interp(low_mix.T, high_mix.shape[0]).T
    # dense 96 -> 96 on the time axis, then 32 channels -> 1
    y = (high_out.T @ w["pred"].T).T + w["pred_b"][:, None]  # (steps, DM)
    out = y @ w["proj"][0] + w["proj_b"][0]
    m, s = stats[0]
    return out * s + m


def main():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, LOOKBACK)
    w = build_weights()
    y = forward(x, w)
    print("out mean %.4f std %.4f first %s" % (y.mean(), y.std(), np.round(y[:4], 4)))

    ref = np.load("timekan_ref.npz") if __import__("os").path.exists("timekan_ref.npz") else None
    if ref is not None:
        got = forward(ref["x"].astype(np.float64), {k: ref[k].astype(np.float64) for k in ref.files if k not in ("x", "y")})
        want = ref["y"].astype(np.float64)
        print("max abs diff vs torch %.3e, corr %.6f" % (
            np.abs(got - want).max(),
            np.corrcoef(got, want)[0, 1]))


if __name__ == "__main__":
    main()


# ---- integer path ---------------------------------------------------------
# Same structure, everything in fixed point. The interpolation between levels
# is a fixed linear operator, so it is applied as a quantised matrix instead of
# running an FFT at inference time.

WQ = 14
DQ = 12


def qmat(m, q=WQ):
    a = np.asarray(m, dtype=np.float64)

    def rec(x):
        if x.ndim == 1:
            return [int(round(float(v) * (1 << q))) for v in x]
        return [rec(y) for y in x]

    return rec(a)


def interp_matrix(src, target, q=WQ):
    """The 96-by-48 operator that rfft-pad-irfft performs, in fixed point."""
    eye = np.eye(src)
    m = rfft_interp(eye, target)          # row s is the response to a unit at s
    return qmat(m.T, q)                   # so the operator is its transpose


def isqrt(n):
    if n <= 0:
        return 0
    r = int(n ** 0.5)
    while r * r > n:
        r -= 1
    while (r + 1) * (r + 1) <= n:
        r += 1
    return r


def clamp16(v):
    return 32767 if v > 32767 else (-32768 if v < -32768 else int(v))


def inorm_i(x, shift=DQ):
    """Integer instance normalisation, same recipe as the LayerNorm elsewhere."""
    n = len(x)
    s = sum(x)
    # truncation toward zero in both places, so the RTL can use plain signed
    # division and still land on the same integer
    mean = -((-s) // n) if s < 0 else s // n
    tot = sum((v - mean) * (v - mean) for v in x)
    var = -((-tot) // n) if tot < 0 else tot // n
    std = max(isqrt(var), 1)
    inv = (1 << 24) // std
    out = [clamp16(((v - mean) * inv) >> shift) for v in x]
    return out, mean, std, inv


def conv1d_i(x, w, q=WQ):
    """x: (steps, ch); w[ch][k]; zero padding of one on each side."""
    steps = len(x)
    ch = len(x[0])
    out = [[0] * ch for _ in range(steps)]
    for t in range(steps):
        for c in range(ch):
            acc = 0
            for k in range(3):
                idx = t + k - 1
                if 0 <= idx < steps:
                    acc += x[idx][c] * w[c][k]
            out[t][c] = clamp16(acc >> q)
    return out


def embedding_i(x, w, q=WQ):
    """Circular padding, as the token convolution uses."""
    steps = len(x)
    ch = len(w)
    out = [[0] * ch for _ in range(steps)]
    for t in range(steps):
        for c in range(ch):
            acc = 0
            for k in range(3):
                idx = (t + k - 1) % steps
                acc += x[idx] * w[c][0][k]
            out[t][c] = clamp16(acc >> q)
    return out


# ---- tables for the nonlinear part ----------------------------------------

def tables():
    tanh_t = [int(round(np.tanh(-8 + i / 16.0) * 4096)) for i in range(256)]
    acos_t = [int(round(np.arccos(-1 + i / 128.0) * 4096)) for i in range(256)]
    cos_t = [int(round(np.cos(2 * np.pi * i / 1024) * 4096)) for i in range(1024)]
    return tanh_t, acos_t, cos_t


def cheby_i(x, coeff, tab):
    """Same basis as the float path, through the tables.

    Products are summed before the shift, which is how one accumulator per
    output works in hardware.
    """
    tanh_t, acos_t, cos_t = tab
    out = []
    for o in range(len(coeff[0])):
        acc = 0
        for i, xv in enumerate(x):
            u1 = tanh_t[((xv + 32768) >> 8) & 255]
            u2 = tanh_t[((u1 + 32768) >> 8) & 255]
            ang = acos_t[((u2 + 4096) >> 5) & 255]
            for k in range(len(coeff[i][o])):
                tk = cos_t[(((k * ang) * 163) >> 12) & 1023]
                acc += tk * coeff[i][o][k]
        out.append(clamp16(acc >> WQ))
    return out


def forward_i(x, wi, tab, interp48):
    """Integer forward, mirroring forward()."""
    steps1 = len(x) // 2
    lvl0 = [clamp16(int(round(v * (1 << DQ)))) for v in x] if abs(x[0]) < 1e-9 or isinstance(x[0], float) else list(x)
    lvl1 = [int((lvl0[2 * t] + lvl0[2 * t + 1]) >> 1) for t in range(steps1)]

    n0, m0, s0, inv0 = inorm_i(lvl0)
    n1, m1, s1, inv1 = inorm_i(lvl1)
    e0 = embedding_i(n0, wi["emb"])
    e1 = embedding_i(n1, wi["emb"])

    low_up = [[0] * DM for _ in range(len(e0))]
    for t in range(len(e0)):
        for c in range(DM):
            acc = 0
            for s2 in range(steps1):
                acc += interp48[t][s2] * e1[s2][c]
            low_up[t][c] = clamp16(acc >> WQ)
    high = [[clamp16(e0[t][c] - low_up[t][c]) for c in range(DM)] for t in range(len(e0))]

    def m_kan_i(xin, coeff, cw):
        y1 = [cheby_i(xin[t], coeff, tab) for t in range(len(xin))]
        y2 = conv1d_i(xin, cw)
        return [[clamp16(y1[t][c] + y2[t][c]) for c in range(DM)] for t in range(len(xin))]

    low_mix = m_kan_i(e1, wi["kan_low"], wi["conv_low"])
    high_mix = m_kan_i(high, wi["kan_high"], wi["conv_high"])

    up = [[0] * DM for _ in range(len(high_mix))]
    for t in range(len(high_mix)):
        for c in range(DM):
            acc = 0
            for s2 in range(steps1):
                acc += interp48[t][s2] * low_mix[s2][c]
            up[t][c] = clamp16(acc >> WQ)
    mixed = [[clamp16(high_mix[t][c] + up[t][c]) for c in range(DM)] for t in range(len(high_mix))]

    # dense 96 -> 96 along time, then 32 channels -> 1
    dense = []
    for t in range(len(mixed)):
        row = [0] * DM
        for c in range(DM):
            acc = 0
            for s2 in range(len(mixed)):
                acc += wi["pred"][t][s2] * mixed[s2][c]
            row[c] = clamp16((acc >> WQ) + (wi["pred_b"][t] >> (WQ - DQ)))
        dense.append(row)
    out = []
    for t in range(len(dense)):
        acc = 0
        for c in range(DM):
            acc += wi["proj"][0][c] * dense[t][c]
        norm = (acc >> WQ) + (wi["proj_b"][0] >> (WQ - DQ))
        # back to the units of the input, using the level-0 statistics
        out.append(clamp16(((norm * s0) >> DQ) + m0))
    return out, dict(lvl0=lvl0, n0=n0, n1=n1, l1=lvl1, e0=e0, e1=e1, up=low_up,
                     high=high, low_mix=low_mix, high_mix=high_mix, mixed=mixed, dense=dense)


_W_EMB, _W_KANL, _W_KANH = 0, 96, 1120
_W_CONVL, _W_CONVH = 3168, 3264
_W_PRED, _W_PREDB, _W_PROJ, _W_PROJB = 3360, 12576, 12672, 12704
_W_TANH, _W_ACOS, _W_COS, _W_INTR = 12736, 12992, 13248, 14272
_WMEM = 18880


def _dump(name, vals, bits=16):
    mask = (1 << bits) - 1
    with open(name, 'w', encoding='ascii', newline='') as fh:
        for v in vals:
            fh.write('%0*x\n' % (bits // 4, int(v) & mask))


def _main():
    wf = build_weights()
    L, L1 = LOOKBACK, LOOKBACK // DSW

    w = np.zeros(_WMEM, dtype=np.int64)
    emb = qmat(wf['emb'])
    for o in range(DM):
        for k in range(3):
            w[_W_EMB + o * 3 + k] = emb[o][0][k]
    kanl = qmat(wf['kan_low'])
    for i in range(DM):
        for o in range(DM):
            w[_W_KANL + i * DM + o] = kanl[i][o][0]
    kanh = qmat(wf['kan_high'])
    for i in range(DM):
        for o in range(DM):
            for k in range(2):
                w[_W_KANH + i * DM * 2 + o * 2 + k] = kanh[i][o][k]
    for tag, base in (('conv_low', _W_CONVL), ('conv_high', _W_CONVH)):
        c = qmat(wf[tag])
        for o in range(DM):
            for k in range(3):
                w[base + o * 3 + k] = c[o][k]
    pred = qmat(wf['pred'])
    for t in range(L):
        for s in range(L):
            w[_W_PRED + t * L + s] = pred[t][s]
        w[_W_PREDB + t] = int(round(float(wf['pred_b'][t]) * (1 << WQ)))
    proj = qmat(wf['proj'])
    for c in range(DM):
        w[_W_PROJ + c] = proj[0][c]
    w[_W_PROJB] = int(round(float(wf['proj_b'][0]) * (1 << WQ)))

    tanh_t, acos_t, cos_t = tables()
    for k, v in enumerate(tanh_t):
        w[_W_TANH + k] = v
    for k, v in enumerate(acos_t):
        w[_W_ACOS + k] = v
    for k, v in enumerate(cos_t):
        w[_W_COS + k] = v
    intr = interp_matrix(L1, L)
    for t in range(L):
        for s in range(L1):
            w[_W_INTR + t * L1 + s] = intr[t][s]

    rng = np.random.default_rng(11)
    xf = rng.normal(0, 1, LOOKBACK)
    xq = [int(round(float(v) * (1 << DQ))) for v in xf]
    wi = {
        'emb': emb, 'kan_low': kanl, 'conv_low': qmat(wf['conv_low']),
        'kan_high': kanh, 'conv_high': qmat(wf['conv_high']),
        'proj': proj, 'proj_b': [int(round(float(v) * (1 << WQ))) for v in wf['proj_b']],
        'pred': pred, 'pred_b': [int(round(float(v) * (1 << WQ))) for v in wf['pred_b']],
    }
    y, _ = forward_i(xq, wi, tables(), intr)

    _dump('tk_w.hex', w)
    _dump('tk_x.hex', xq)
    _dump('tk_y_ref.hex', y)
    print('weights', len(w), 'input', len(xq), 'expected', len(y))


if __name__ == '__main__':
    _main()
