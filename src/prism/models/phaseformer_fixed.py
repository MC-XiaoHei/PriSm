"""Integer PhaseFormer: phase tokens, router attention, shared head.

One layer, latent width 32, four heads, four routers. The attention is
bidirectional: routers read the phase tokens and the tokens read the routers
back, which is what makes the routing data dependent rather than a fixed mix.
Every nonlinearity is a table or an integer routine, so the hardware can match
it exactly.
"""
import math
import pathlib

L = 96          # look-back
P = 24          # phases (the period)
HP = 4          # cycles per phase, i.e. forecast length / period
DM = 32         # latent width
NH = 4          # heads
DH = DM // NH   # 8
NR = 4          # routers
DFF = 32
Q = 12
WQ = 14
OUT = pathlib.Path('.')

# exp(-x) for x in [0, 8) step 1/32, Q12
EXP_STEP = 5      # x is measured in 1/32 units, i.e. >> 5 from Q12
EXP_N = 256
# gelu(x) = x * 0.5 * (1 + erf(x/sqrt2)), tabulated over [-4, 4) step 1/32
GELU_STEP = 5
GELU_N = 256
GELU_OFF = 128


def rr(v, bits=WQ):
    return int((v + (1 << (bits - 1))) >> bits)


def q(v, bits):
    return int(round(v * (1 << bits)))


def clamp16(v):
    return 32767 if v > 32767 else (-32768 if v < -32768 else v)


def isqrt(n):
    return int(math.isqrt(n)) if n > 0 else 0


def lcg(seed):
    s = seed
    while True:
        s = (1103515245 * s + 12345) & 0x7FFFFFFF
        yield s


def exp_table():
    return [min(4095, q(math.exp(-i / 32.0), Q)) for i in range(EXP_N)]


def gelu_table():
    out = []
    for i in range(GELU_N):
        x = (i - GELU_OFF) / 32.0
        v = 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))
        out.append(clamp16(q(v, Q)))
    return out


def exp_lut(tab, x):
    """x is Q12, non-positive."""
    if x >= 0:
        return tab[0]
    i = (-x) >> EXP_STEP
    return tab[i] if i < EXP_N else 0


def gelu_lut(tab, x):
    i = (x >> GELU_STEP) + GELU_OFF
    i = 0 if i < 0 else (GELU_N - 1 if i >= GELU_N else i)
    return tab[i]


def matvec(x, w, out_dim, in_dim):
    """w[o][i], Q14."""
    return [rr(sum(x[i] * w[o][i] for i in range(in_dim))) for o in range(out_dim)]


def softmax(tab, s):
    m = max(s)
    e = [exp_lut(tab, s[i] - m) for i in range(len(s))]
    tot = sum(e)
    if tot == 0:
        tot = 1
    # one reciprocal, then a multiply per element: the hardware divides once
    inv = (1 << (Q + 12)) // tot
    return [(v * inv) >> 12 for v in e]


def layernorm(x, g, b):
    n = len(x)
    mean = int(sum(x) / n)
    var = int(sum((v - mean) * (v - mean) for v in x) / n)
    std = isqrt(max(var, 0))
    if std == 0:
        std = 1
    inv = (1 << 24) // std
    return [clamp16(((((x[i] - mean) * inv) >> 12) * g[i]) >> 12) + b[i] for i in range(n)]


def make_weights():
    n = lcg(20261024)
    def qv(scale=0.5):
        return q(((next(n) % 2001) - 1000) / 1000.0 * scale, WQ)
    w = {}
    w['emb'] = [[qv(0.5) for _ in range(HP)] for _ in range(DM)]
    w['q'] = [[qv(0.25) for _ in range(DM)] for _ in range(DM)]
    w['k'] = [[qv(0.25) for _ in range(DM)] for _ in range(DM)]
    w['v'] = [[qv(0.25) for _ in range(DM)] for _ in range(DM)]
    w['rq'] = [[qv(0.25) for _ in range(DM)] for _ in range(NR)]
    w['o'] = [[qv(0.25) for _ in range(DM)] for _ in range(DM)]
    w['f1'] = [[qv(0.25) for _ in range(DM)] for _ in range(DFF)]
    w['f2'] = [[qv(0.25) for _ in range(DFF)] for _ in range(DM)]
    w['head'] = [[qv(0.25) for _ in range(DM)] for _ in range(HP)]
    w['ln1g'] = [q(1.0, Q)] * DM
    w['ln1b'] = [0] * DM
    w['ln2g'] = [q(1.0, Q)] * DM
    w['ln2b'] = [0] * DM
    w['lnhg'] = [q(1.0, Q)] * DM
    w['lnhb'] = [0] * DM
    # router tokens are learned constants
    w['rout'] = [[qv(0.5) for _ in range(DM)] for _ in range(NR)]
    return w


def run(xq, w, etab, gtab, dbg=None):
    # embed: token[p] = emb @ x[p][:HP]
    tok = []
    for p in range(P):
        x = [xq[p + c * P] for c in range(HP)]
        tok.append(matvec(x, w['emb'], DM, HP))

    # Q, K, V over the tokens
    qs = [matvec(t, w['q'], DM, DM) for t in tok]
    ks = [matvec(t, w['k'], DM, DM) for t in tok]
    vs = [matvec(t, w['v'], DM, DM) for t in tok]

    ctx_tok = [[0] * DM for _ in range(P)]
    ctx_rout = [[0] * DM for _ in range(NR)]
    for h in range(NH):
        lo, hi = h * DH, (h + 1) * DH
        # routers read the tokens
        for r in range(NR):
            sc = [sum(w['rout'][r][d] * ks[t][d] for d in range(lo, hi)) for t in range(P)]
            sc = [rr(v) for v in sc]
            pr = softmax(etab, sc)
            for d in range(lo, hi):
                ctx_rout[r][d] = rr(sum(pr[t] * vs[t][d] for t in range(P)))
        # tokens read the routers
        rctx = [matvec(w['rout'][r], w['v'], DM, DM) for r in range(NR)]
        for t in range(P):
            sc = [rr(sum(qs[t][d] * rctx[r][d] for d in range(lo, hi))) for r in range(NR)]
            pr = softmax(etab, sc)
            for d in range(lo, hi):
                ctx_tok[t][d] = rr(sum(pr[r] * rctx[r][d] for r in range(NR)))

    if dbg is not None:
        dbg['tok'] = tok
        dbg['q'] = qs
        dbg['k'] = ks
        dbg['v'] = vs
        dbg['rct'] = rctx
        dbg['ctxt'] = ctx_tok

    # merge heads and project out
    merged = [[ctx_tok[t][d] for d in range(DM)] for t in range(P)]
    attn = [matvec(m, w['o'], DM, DM) for m in merged]
    h1 = [[clamp16(tok[p][d] + attn[p][d]) for d in range(DM)] for p in range(P)]
    if dbg is not None:
        dbg['h1pre'] = [list(r) for r in h1]
    h1 = [layernorm(h1[p], w['ln1g'], w['ln1b']) for p in range(P)]
    if dbg is not None:
        dbg['h1'] = h1

    # feed forward
    h2 = []
    for p in range(P):
        f = matvec(h1[p], w['f1'], DFF, DM)
        if dbg is not None:
            dbg.setdefault('f1', []).append(list(f))
        f = [gelu_lut(gtab, v) for v in f]
        if dbg is not None:
            dbg.setdefault('f1a', []).append(list(f))
        o = matvec(f, w['f2'], DM, DFF)
        h2.append([clamp16(h1[p][d] + o[d]) for d in range(DM)])
    h2 = [layernorm(h2[p], w['ln2g'], w['ln2b']) for p in range(P)]

    if dbg is not None:
        dbg['h2'] = h2

    # shared head, one row per phase
    y = [matvec(h2[p], w['head'], HP, DM) for p in range(P)]
    out = [0] * (HP * P)
    for p in range(P):
        for o in range(HP):
            out[o * P + p] = y[p][o]
    return out


def emit(name, vals):
    (OUT / name).write_text('\n'.join('%04x' % (v & 0xFFFF) for v in vals) + '\n')


def load_hex(path):
    vals = []
    for tok in pathlib.Path(path).read_text().split():
        v = int(tok, 16) & 0xFFFF
        vals.append(v - 0x10000 if v >= 0x8000 else v)
    return vals


if __name__ == '__main__':
    n = lcg(7)
    xq = []
    level = 0
    for t in range(L):
        level += (next(n) % 401) - 200
        xq.append(level + int(3000 * math.sin(2 * math.pi * t / P)) + (next(n) % 2001) - 1000)
    w = make_weights()
    etab, gtab = exp_table(), gelu_table()
    out = run(xq, w, etab, gtab)
    emit('pf_x_in.hex', xq)
    emit('pf_y_ref.hex', out)
    emit('pf_exp.hex', etab)
    emit('pf_gelu.hex', gtab)
    for key in ('emb', 'q', 'k', 'v', 'o', 'f1', 'f2', 'head', 'ln1g', 'ln2g', 'lnhg'):
        e = w[key]
        flat = e if isinstance(e[0], int) else [e[r][c] for r in range(len(e)) for c in range(len(e[0]))]
        emit('pf_%s.hex' % key, flat)
    emit('pf_rq.hex', [w['rq'][r][d] for r in range(NR) for d in range(DM)])
    emit('pf_rout.hex', [w['rout'][r][d] for r in range(NR) for d in range(DM)])
    print('out', out[:6])
