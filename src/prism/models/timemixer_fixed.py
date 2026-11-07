"""TimeMixer reference, float first, integer later.

Structure copied from the official model as it runs in our experiments:
lookback 96, pred 96, down_sampling_layers 3, down_sampling_window 2,
down_sampling_method avg, d_model 16, d_ff 32, two PDM blocks, moving-average
decomposition with kernel 25, channel independent. The whole model runs once
per variable.

Four scales (96, 48, 24, 12) are produced by average pooling, each instance
normalised with its own affine pair, then embedded by a length-3 circular
convolution from one input channel to d_model. Every PDM block decomposes each
scale into season and trend, mixes seasons bottom up and trends top down,
sums the two, adds the block input and runs a d_model->d_ff->d_model gate.
Each scale then gets its own temporal projection to the horizon, the d_model
axis collapses to one value, and the four predictions are summed and denormalised
with the level-0 statistics.
"""
import numpy as np

LOOKBACK = 96
PRED = 96
DM = 16
DFF = 32
BLOCKS = 2
DSL = 3
DSW = 2
MA = 25
EPS = 1e-5
SEED = 11

SCALES = [LOOKBACK // (DSW ** i) for i in range(DSL + 1)]


def gelu(x):
    from math import erf, sqrt
    v = np.vectorize(lambda u: 0.5 * u * (1.0 + erf(u / sqrt(2.0))))
    return v(x)


def pool_avg(x):
    """AvgPool1d(window, stride) over the time axis of a (T, C) array."""
    steps = x.shape[0] // DSW
    out = np.zeros((steps, x.shape[1]))
    for t in range(steps):
        out[t] = x[t * DSW:(t + 1) * DSW].mean(axis=0)
    return out


def moving_avg(x, kernel=MA):
    """Replicate padding then a running mean, over axis 0."""
    pad = (kernel - 1) // 2
    steps = x.shape[0]
    ext = np.concatenate([np.repeat(x[0:1], pad, axis=0), x,
                          np.repeat(x[-1:], pad, axis=0)], axis=0)
    out = np.zeros_like(x)
    for t in range(steps):
        out[t] = ext[t:t + kernel].mean(axis=0)
    return out


def instance_norm(x, w, b):
    m = x.mean()
    s = np.sqrt(x.var() + EPS)
    return ((x - m) / s) * w + b, m, s


def embed(x, w):
    """Circular length-3 convolution, one input channel, DM outputs."""
    steps = x.shape[0]
    out = np.zeros((steps, DM))
    for t in range(steps):
        for k in range(3):
            src = x[(t + k - 1) % steps, 0]
            out[t] += src * w[:, 0, k]
    return out


def linear(x, w, b):
    return x @ w.T + b


def season_mix(seasons, p):
    """Bottom-up: high resolution drives the next level down."""
    high = seasons[0]
    low = seasons[1]
    outs = [high]
    for i in range(len(seasons) - 1):
        res = linear(gelu(linear(high, p[f'down.{i}.0.w'], p[f'down.{i}.0.b'])),
                     p[f'down.{i}.2.w'], p[f'down.{i}.2.b'])
        low = low + res
        high = low
        if i + 2 <= len(seasons) - 1:
            low = seasons[i + 2]
        outs.append(high)
    return outs


def trend_mix(trends, p):
    """Top-down: the coarsest level drives the next level up."""
    rev = list(reversed(trends))
    low = rev[0]
    high = rev[1]
    outs = [low]
    for i in range(len(rev) - 1):
        res = linear(gelu(linear(low, p[f'up.{i}.0.w'], p[f'up.{i}.0.b'])),
                     p[f'up.{i}.2.w'], p[f'up.{i}.2.b'])
        high = high + res
        low = high
        if i + 2 <= len(rev) - 1:
            high = rev[i + 2]
        outs.append(low)
    outs.reverse()
    return outs


def block(x_list, p):
    """x_list entries are (T, DM); the mixers work on (DM, T)."""
    seasons, trends = [], []
    for x in x_list:
        trend = moving_avg(x)
        seasons.append((x - trend).T)
        trends.append(trend.T)
    smix = season_mix(seasons, p)
    tmix = trend_mix(trends, p)
    outs = []
    for ori, s, t in zip(x_list, smix, tmix):
        out = (s + t).T
        gate = linear(gelu(linear(out, p['out.0.w'], p['out.0.b'])),
                      p['out.2.w'], p['out.2.b'])
        outs.append(ori + gate)
    return outs


def forward(params, x, want_int=False):
    """x: (LOOKBACK,) float series for one variable."""
    x = np.asarray(x, dtype=np.float64).reshape(LOOKBACK, 1)
    scales = [x]
    for _ in range(DSL):
        scales.append(pool_avg(scales[-1]))

    normed, stats = [], []
    for i, s in enumerate(scales):
        n, m, sd = instance_norm(s, params['norm'][i][0], params['norm'][i][1])
        normed.append(n)
        stats.append((m, sd))

    enc = [embed(n, params['emb']) for n in normed]
    for k in range(BLOCKS):
        enc = block(enc, params['blocks'][k])

    total = np.zeros(PRED)
    for i, e in enumerate(enc):
        d = linear(e.T, params['predict'][i][0], params['predict'][i][1]).T
        d = linear(d, params['proj'][0], params['proj'][1])
        total += d[:, 0]

    m0, s0 = stats[0]
    w0, b0 = params['norm'][0]
    out = total * (w0 + 1e-10) + b0
    out = out * s0 + m0
    return out


def build_weights(seed=SEED):
    """Same shapes as the official state dict, in numpy."""
    rng = np.random.default_rng(seed)

    def mat(rows, cols, scale=0.3):
        return rng.normal(0.0, scale / np.sqrt(cols), (rows, cols))

    def vec(n, scale=0.1):
        return rng.normal(0.0, scale, (n,))

    p = dict(
        emb=rng.normal(0.0, 0.4, (DM, 1, 3)),
        norm=[(np.ones(1), np.zeros(1)) for _ in SCALES],
        blocks=[], predict=[], proj=None,
    )
    for _ in range(BLOCKS):
        b = {'out.0.w': mat(DFF, DM), 'out.0.b': vec(DFF),
             'out.2.w': mat(DM, DFF), 'out.2.b': vec(DM)}
        for i in range(DSL):
            b[f'down.{i}.0.w'] = mat(SCALES[i + 1], SCALES[i])
            b[f'down.{i}.0.b'] = vec(SCALES[i + 1])
            b[f'down.{i}.2.w'] = mat(SCALES[i + 1], SCALES[i + 1])
            b[f'down.{i}.2.b'] = vec(SCALES[i + 1])
            b[f'up.{i}.0.w'] = mat(SCALES[DSL - 1 - i], SCALES[DSL - i])
            b[f'up.{i}.0.b'] = vec(SCALES[DSL - 1 - i])
            b[f'up.{i}.2.w'] = mat(SCALES[DSL - 1 - i], SCALES[DSL - 1 - i])
            b[f'up.{i}.2.b'] = vec(SCALES[DSL - 1 - i])
        p['blocks'].append(b)
    p['predict'] = [(mat(PRED, SCALES[i]), vec(PRED)) for i in range(len(SCALES))]
    p['proj'] = (mat(1, DM), vec(1))
    return p


def sample_input(seed=SEED):
    rng = np.random.default_rng(seed + 1)
    t = np.arange(LOOKBACK)
    return (np.sin(2 * np.pi * t / 24.0) * 0.6
            + np.sin(2 * np.pi * t / 7.0) * 0.3
            + rng.normal(0.0, 0.05, LOOKBACK)
            + 1.5)


# ---- integer path ---------------------------------------------------------
# Same schedule, everything in fixed point: activations Q12, weights Q14,
# 44-bit accumulators, one saturation on the way back to Q12. The GELU table is
# indexed by the top bits of the Q12 input, so the reference and the RTL read
# the same 256 entries.

WQ = 14
DQ = 12
ACC = 44
GELU_STEP = 6
GELU_N = 512
GELU_OFF = 256


def clamp16(v):
    v = int(v)
    return 32767 if v > 32767 else (-32768 if v < -32768 else v)


def qw(v):
    return clamp16(round(float(v) * (1 << WQ)))


def qb(v):
    """Bias also at Q14; the datapath shifts it to Q12 when adding."""
    return clamp16(round(float(v) * (1 << WQ)))


def qmat(m):
    if hasattr(m, 'ndim'):
        m = m.tolist()

    def rec(x):
        if isinstance(x, list):
            return [rec(y) for y in x]
        return qw(x)

    return rec(m)


def qvec(v):
    return [qb(x) for x in v]


def gelu_table():
    import math
    out = []
    for i in range(GELU_N):
        x = (i - GELU_OFF) / (1 << GELU_STEP)
        out.append(clamp16(round(0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0))) * (1 << DQ))))
    return out


def gelu_lut(tab, x):
    i = (x >> GELU_STEP) + GELU_OFF
    i = 0 if i < 0 else (GELU_N - 1 if i >= GELU_N else i)
    return tab[i]


def isqrt(n):
    if n <= 0:
        return 0
    r = int(n ** 0.5)
    while r * r > n:
        r -= 1
    while (r + 1) * (r + 1) <= n:
        r += 1
    return r


def inorm_i(x):
    n = len(x)
    s = sum(x)
    mean = -((-s) // n) if s < 0 else s // n
    tot = sum((v - mean) * (v - mean) for v in x)
    var = -((-tot) // n) if tot < 0 else tot // n
    std = max(isqrt(var), 1)
    inv = (1 << 24) // std
    return [clamp16(((v - mean) * inv) >> DQ) for v in x], mean, std


def div25(s):
    """1/25 as a 24-bit reciprocal, exact enough to stay inside one LSB."""
    return (s * 671089) >> 24


def pool2(v):
    return [(v[2 * t] + v[2 * t + 1]) >> 1 for t in range(len(v) // 2)]


def embed_i(x, emb):
    steps = len(x)
    out = []
    for t in range(steps):
        row = []
        for c in range(DM):
            acc = 0
            for k in range(3):
                acc += x[(t + k - 1) % steps] * emb[c][0][k]
            row.append(clamp16(acc >> WQ))
        out.append(row)
    return out


def moving_avg_i(buf):
    steps = len(buf)
    pad = (MA - 1) // 2
    out = []
    for t in range(steps):
        row = []
        for c in range(DM):
            acc = 0
            for k in range(-pad, pad + 1):
                row_acc = t + k
                row_acc = 0 if row_acc < 0 else (steps - 1 if row_acc >= steps else row_acc)
                acc += buf[row_acc][c]
            row.append(clamp16(div25(acc)))
        out.append(row)
    return out


def addmat(a, b):
    return [[clamp16(u + v) for u, v in zip(ra, rb)] for ra, rb in zip(a, b)]


def submat(a, b):
    return [[clamp16(u - v) for u, v in zip(ra, rb)] for ra, rb in zip(a, b)]


def linear_time(buf, w, b):
    """y[o][c] = sum_t w[o][t]*buf[t][c], over the time axis."""
    t_in = len(buf)
    out = []
    for o in range(len(w)):
        row = []
        for c in range(DM):
            acc = 0
            for t in range(t_in):
                acc += w[o][t] * buf[t][c]
            row.append(clamp16((acc >> WQ) + (b[o] >> (WQ - DQ))))
        out.append(row)
    return out


def gelu_mat(buf, tab):
    return [[gelu_lut(tab, v) for v in row] for row in buf]


def season_mix_i(seasons, wi, tab):
    high = seasons[0]
    low = seasons[1]
    outs = [high]
    for i in range(len(seasons) - 1):
        h = gelu_mat(linear_time(high, wi[f'down.{i}.0.w'], wi[f'down.{i}.0.b']), tab)
        res = linear_time(h, wi[f'down.{i}.2.w'], wi[f'down.{i}.2.b'])
        low = addmat(low, res)
        high = low
        if i + 2 <= len(seasons) - 1:
            low = seasons[i + 2]
        outs.append(high)
    return outs


def trend_mix_i(trends, wi, tab):
    rev = list(reversed(trends))
    low = rev[0]
    high = rev[1]
    outs = [low]
    for i in range(len(rev) - 1):
        h = gelu_mat(linear_time(low, wi[f'up.{i}.0.w'], wi[f'up.{i}.0.b']), tab)
        res = linear_time(h, wi[f'up.{i}.2.w'], wi[f'up.{i}.2.b'])
        high = addmat(high, res)
        low = high
        if i + 2 <= len(rev) - 1:
            high = rev[i + 2]
        outs.append(low)
    outs.reverse()
    return outs


def gate_i(buf, wi, tab):
    """d_model -> d_ff -> d_model over the channel axis, per time step."""
    out = []
    for row in buf:
        hid = []
        for o in range(DFF):
            acc = 0
            for c in range(DM):
                acc += wi['out.0.w'][o][c] * row[c]
            hid.append(gelu_lut(tab, clamp16((acc >> WQ) + (wi['out.0.b'][o] >> (WQ - DQ)))))
        res = []
        for c in range(DM):
            acc = 0
            for o in range(DFF):
                acc += wi['out.2.w'][c][o] * hid[o]
            res.append(clamp16((acc >> WQ) + (wi['out.2.b'][c] >> (WQ - DQ))))
        out.append(res)
    return out


def block_i(xl, wi, tab):
    seasons, trends = [], []
    for x in xl:
        tr = moving_avg_i(x)
        seasons.append(submat(x, tr))
        trends.append(tr)
    smix = season_mix_i(seasons, wi, tab)
    tmix = trend_mix_i(trends, wi, tab)
    outs = []
    for ori, s, t in zip(xl, smix, tmix):
        out = addmat(s, t)
        outs.append(addmat(ori, gate_i(out, wi, tab)))
    return outs


def forward_i(wi, tab, xq):
    """xq: LOOKBACK integers at Q12."""
    scales = [list(xq)]
    for _ in range(DSL):
        scales.append(pool2(scales[-1]))

    normed, stats = [], []
    for s in scales:
        n, mean, std = inorm_i(s)
        normed.append(n)
        stats.append((mean, std))

    enc = [embed_i(n, wi['emb']) for n in normed]
    for k in range(BLOCKS):
        enc = block_i(enc, wi['blocks'][k], tab)

    acc = [0] * PRED
    for i, e in enumerate(enc):
        y = linear_time(e, wi['predict.{}.w'.format(i)], wi['predict.{}.b'.format(i)])
        for o in range(PRED):
            a = 0
            for c in range(DM):
                a += wi['proj.w'][0][c] * y[o][c]
            acc[o] += clamp16((a >> WQ) + (wi['proj.b'][0] >> (WQ - DQ)))

    mean0, std0 = stats[0]
    return [clamp16(((v * std0) >> DQ) + mean0) for v in acc]


def int_weights(p):
    """Quantise the float parameter tree and tag every block's matrices."""
    wi = dict(emb=qmat(p['emb']), blocks=[], proj=(qmat(p['proj'][0]), qvec(p['proj'][1])))
    for b in p['blocks']:
        nb = {}
        for key, mat in b.items():
            nb[key] = qmat(mat) if key.endswith('.w') else qvec(mat)
        wi['blocks'].append(nb)
    wi['predict'] = {}
    for i, (w, b) in enumerate(p['predict']):
        wi['predict'][f'predict.{i}.w'] = qmat(w)
        wi['predict'][f'predict.{i}.b'] = qvec(b)
    flat = dict(emb=wi['emb'], proj=wi['proj'])
    flat['proj.w'], flat['proj.b'] = wi['proj']
    flat['blocks'] = wi['blocks']
    for k in wi['predict']:
        flat[k] = wi['predict'][k]
    return flat


# ---- rom image ------------------------------------------------------------
# One flat weight memory, laid out in the order the datapath walks it. The RTL
# includes the generated header, so offsets cannot drift from this file.

def pack_rom(wi, tab):
    img = list(tab)
    off = {'gelu': (0, GELU_N, 1)}

    def put(name, mat, bias=None):
        rows = len(mat)
        cols = len(mat[0]) if rows else 0
        off[name] = (len(img), rows, cols)
        for row in mat:
            img.extend(row)
        if bias is not None:
            off[name + '_b'] = (len(img), len(bias), 1)
            img.extend(bias)

    put('emb', [row[0] for row in wi['emb']])
    for k in range(BLOCKS):
        b = wi['blocks'][k]
        put(f'b{k}_g1', b['out.0.w'], b['out.0.b'])
        put(f'b{k}_g2', b['out.2.w'], b['out.2.b'])
        for i in range(DSL):
            put(f'b{k}_down{i}_0', b[f'down.{i}.0.w'], b[f'down.{i}.0.b'])
            put(f'b{k}_down{i}_2', b[f'down.{i}.2.w'], b[f'down.{i}.2.b'])
            put(f'b{k}_up{i}_0', b[f'up.{i}.0.w'], b[f'up.{i}.0.b'])
            put(f'b{k}_up{i}_2', b[f'up.{i}.2.w'], b[f'up.{i}.2.b'])
    for i in range(DSL + 1):
        put(f'pred{i}', wi[f'predict.{i}.w'], wi[f'predict.{i}.b'])
    put('proj', wi['proj.w'], wi['proj.b'])
    return img, off


def write_hex(path, values):
    with open(path, 'w', encoding='ascii') as fh:
        for v in values:
            fh.write('%04x\n' % (int(v) & 0xFFFF))


def write_offsets(path, off):
    lines = ['// generated by timemixer_ref.py, do not edit',
             f'localparam integer WROM = {max(o[0] + o[1] * o[2] for o in off.values())};',
             f'localparam integer GELU_N = {GELU_N};',
             f'localparam integer GELU_OFF = {GELU_OFF};',
             f'localparam integer GELU_STEP = {GELU_STEP};']
    for name in sorted(off):
        base, rows, cols = off[name]
        lines.append(f'localparam integer O_{name.upper()} = {base};')
        if name.endswith('_b') or name == 'gelu':
            continue
        lines.append(f'localparam integer N_{name.upper()} = {rows};')
        lines.append(f'localparam integer C_{name.upper()} = {cols};')
    with open(path, 'w', encoding='ascii') as fh:
        fh.write('\n'.join(lines) + '\n')


def gate_rom(wi):
    """Pack the gate coefficients so one 256-bit word feeds all sixteen lanes.

    Lanes are output channels here, so a word holds w[o][c] for the sixteen o of
    one group, or w2[c][o] for the sixteen c of one time step.
    """
    words = [0] * 256
    for k in range(BLOCKS):
        b = wi['blocks'][k]
        w1, b1 = b['out.0.w'], b['out.0.b']
        w2, b2 = b['out.2.w'], b['out.2.b']
        for g in range(2):
            for c in range(DM):
                word = 0
                for n in range(DM):
                    word |= (w1[g * DM + n][c] & 0xFFFF) << (n * 16)
                words[k * 128 + g * DM + c] = word
            word = 0
            for n in range(DM):
                word |= (b1[g * DM + n] & 0xFFFF) << (n * 16)
            words[k * 128 + 64 + g] = word
        for o in range(DFF):
            word = 0
            for c in range(DM):
                word |= (w2[c][o] & 0xFFFF) << (c * 16)
            words[k * 128 + 32 + o] = word
        word = 0
        for c in range(DM):
            word |= (b2[c] & 0xFFFF) << (c * 16)
        words[k * 128 + 66] = word
    return words


def write_wide(path, words, width=256):
    with open(path, 'w', encoding='ascii') as fh:
        for w in words:
            fh.write('%064x\n' % (int(w) & ((1 << width) - 1)))


def emit(prefix='tm'):
    p = build_weights()
    wi = int_weights(p)
    tab = gelu_table()
    img, off = pack_rom(wi, tab)
    write_hex(f'{prefix}_w.hex', img)
    write_hex(f'{prefix}_x.hex', [clamp16(round(v * (1 << DQ)))
                                  for v in sample_input()])
    write_hex(f'{prefix}_y_ref.hex', forward_i(wi, tab,
              [clamp16(round(v * (1 << DQ))) for v in sample_input()]))
    write_wide(f'{prefix}_g.hex', gate_rom(wi))
    write_offsets(f'{prefix}_offsets.vh', off)
    return img, off


if __name__ == '__main__':
    img, off = emit()
    print(f'rom words {len(img)}  entries {len(off)}')
    print(f'out of {GELU_N}-entry table at 0, emb at {off["emb"][0]}')
