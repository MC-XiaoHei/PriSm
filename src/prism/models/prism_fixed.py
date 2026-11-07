"""Fixed-point reference for the model, used to verify the RTL.

Everything the hardware does, in the same integer arithmetic, in the same order.
A mismatch means a control-flow error rather than a rounding rule disagreeing.

    normalise   local mean/variance over the last 32 steps, floored against the
                whole window and against an absolute 0.05
    fold        96 time steps -> 24 phases x 4 cycles
    in_proj     cycles -> d, one vector per phase
    block x2    depthwise (phase, variate) conv, rank-1 phase mix, 1x1, residual,
                activation between the rank mix and the pointwise projection
    out_proj    d -> out cycles, one vector per phase
    skip        cycles -> out cycles straight from the folded input
    denormalise

Fixed point: data Q12, weights Q14. Every product is Q12*Q14 = Q26; the
accumulator stays at Q26 and rounds once on the way back to Q12.
The three projections are affine: each carries a Q14 bias that joins the
accumulator at Q26 before that single rounding.
"""
import math
import pathlib

Q = 12          # data fraction bits
WQ = 14         # weight fraction bits
L = 96          # look-back
P = 24          # phase length (the dataset period)
D = 16          # feature width
KL = 5          # phase-axis kernel
KV = 1          # variate-axis kernel
RZ = 1          # rank of the phase-axis mix
BLOCKS = 2
HP = 4          # pred_len / P
LP = L // P     # cycles per phase
PAD_KL = 1 << (KL - 1).bit_length()
PAD_LP = 1 << (P - 1).bit_length()

OUT = pathlib.Path('.')


def q(v, bits):
    return int(round(v * (1 << bits)))


def clamp(v):
    return 32767 if v > 32767 else (-32768 if v < -32768 else v)


def rr(v):
    """Round half up then shift, matching the saturator in the RTL."""
    return clamp((v + (1 << (WQ - 1))) >> WQ)


def isqrt(n):
    """Integer square root, restoring algorithm, as the RTL computes it."""
    if n <= 0:
        return 0
    res, bit = 0, 1 << 30
    while bit > n:
        bit >>= 2
    while bit:
        if n >= res + bit:
            n -= res + bit
            res = (res >> 1) + bit
        else:
            res >>= 1
        bit >>= 2
    return res


def lcg(seed):
    s = [seed]

    def nxt():
        s[0] = (s[0] * 1103515245 + 12345) & 0x7FFFFFFF
        return s[0]
    return nxt


def emit(name, vals):
    (OUT / name).write_text('\n'.join('%04x' % (v & 0xFFFF) for v in vals) + '\n')


def load_hex(path):
    src = pathlib.Path(path)
    if not src.is_absolute():
        src = OUT / src
    vals = []
    for tok in src.read_text().split():
        v = int(tok, 16) & 0xFFFF
        vals.append(v - 0x10000 if v >= 0x8000 else v)
    return vals


def make_weights():
    """Block weights come off disk, because the core reads those files.

    The projections are not implemented as tables in hardware yet, so they are
    generated here and written out for the wrapper.
    """
    n = lcg(424242)
    def qv(scale=0.5):
        return q(((n() % 2001) - 1000) / 1000.0 * scale, WQ)

    w = {}
    wi = load_hex('w_in.hex')
    wo = load_hex('w_out.hex')
    ws = load_hex('w_skip.hex')
    w['in'] = [[wi[c * D + d] for d in range(D)] for c in range(LP)]
    w['out'] = [[wo[d * HP + o] for o in range(HP)] for d in range(D)]
    w['skip'] = [[ws[c * HP + o] for o in range(HP)] for c in range(LP)]
    w['b_in'] = load_hex('b_in.hex')
    w['b_out'] = load_hex('b_out.hex')

    w['blocks'] = [_bank(suffix) for suffix in ('', '_1')]
    return w


def _bank(suffix):
    """One pass's lattice weights: the core holds two banks in one array."""
    dw = load_hex(f'w_dw_pad{suffix}.hex')
    wv = load_hex(f'w_v_pad{suffix}.hex')
    wu = load_hex(f'w_u{suffix}.hex')
    wp = load_hex(f'w_p{suffix}.hex')
    bp = load_hex(f'b_p{suffix}.hex')
    return {
        'dw': [[[dw[d * PAD_KL + t] for t in range(KL)] for d in range(D)]],
        'v': [[wv[r * PAD_LP + p] for p in range(P)] for r in range(RZ)],
        'u': [[wu[p * RZ + r] for r in range(RZ)] for p in range(P)],
        'p': [[wp[fo * D + fi] for fi in range(D)] for fo in range(D)],
        'b_p': bp,
    }


def normalise(xq):
    """Statistics over the last 32 steps and over the whole window.

    The local scale is floored against a tenth of the window scale and against
    0.05, so a near-flat channel cannot blow the division up. Division truncates
    toward zero, which is what the restoring divider produces.
    """
    win = 32
    acc1 = sum(xq)
    acc2 = sum(v * v for v in xq)
    acc1w = sum(xq[-win:])
    acc2w = sum(v * v for v in xq[-win:])

    mean_all = int(acc1 / L)
    var_all = int(acc2 / L) - mean_all * mean_all
    mean_w = int(acc1w / win)
    var_loc = int(acc2w / win) - mean_w * mean_w

    std_loc = isqrt(max(var_loc, 0))
    std_all = isqrt(max(var_all, 0))
    std = max(std_loc, std_all // 10, (5 << Q) // 100)
    if std == 0:
        std = 1
    # the window is centred on the local mean, as the float model does
    return mean_w, std, (1 << 24) // std


def fold(x):
    """[L] -> [P][LP]: lattice[p][c] is time step c*P + p."""
    return [[x[c * P + p] for c in range(LP)] for p in range(P)]


def block(z, w, bank=0):
    h = [[0] * D for _ in range(P)]
    for p in range(P):
        for d in range(D):
            acc = 0
            for t in range(KL):
                acc += z[(p + t - KL // 2) % P][d] * w['blocks'][bank]['dw'][0][d][t]
            h[p][d] = rr(acc)

    # the down result has to live somewhere else: writing it back into g index
    # r made every phase after the first read a value that had already been
    # scaled by the up projection
    gd = [[0] * D for _ in range(RZ)]
    for r in range(RZ):
        for d in range(D):
            acc = 0
            for p in range(P):
                acc += h[p][d] * w['blocks'][bank]['v'][r][p]
            gd[r][d] = rr(acc)

    g = [[0] * D for _ in range(P)]
    for p in range(P):
        for d in range(D):
            g[p][d] = rr(sum(gd[r][d] * w['blocks'][bank]['u'][p][r] for r in range(RZ)))

    h = [[max(0, clamp(h[p][d] + g[p][d])) for d in range(D)] for p in range(P)]

    out = [[0] * D for _ in range(P)]
    for p in range(P):
        for d in range(D):
            acc = 0
            for k in range(D):
                acc += h[p][k] * w['blocks'][bank]['p'][d][k]
            out[p][d] = rr(z[p][d] * (1 << WQ) + acc + (w['blocks'][bank]['b_p'][d] << 12))
    return out


def run(xq, w):
    mean, std, inv = normalise(xq)
    zn = [((v - mean) * inv) >> 12 for v in xq]
    zk = fold(zn)

    z = [[0] * D for _ in range(P)]
    for p in range(P):
        for d in range(D):
            acc = 0
            for c in range(LP):
                acc += zk[p][c] * w['in'][c][d]
            z[p][d] = rr(acc + (w['b_in'][d] << 12))

    for bank in range(BLOCKS):
        z = block(z, w, bank)

    y = [[0] * HP for _ in range(P)]
    for p in range(P):
        for o in range(HP):
            acc = 0
            for d in range(D):
                acc += z[p][d] * w['out'][d][o]
            acc2 = 0
            for c in range(LP):
                acc2 += zk[p][c] * w['skip'][c][o]
            y[p][o] = rr(acc + (w['b_out'][o] << 12)) + rr(acc2)

    yq = [0] * (HP * P)
    for p in range(P):
        for o in range(HP):
            yq[o * P + p] = y[p][o]
    return yq, mean, std, inv


def _projections() -> None:
    """The input, output and skip projections, drawn from the same stream as before."""
    n = lcg(424242)

    def qv(scale=0.5):
        return q(((n() % 2001) - 1000) / 1000.0 * scale, WQ)

    wi = [[qv(0.6) for _ in range(D)] for _ in range(LP)]
    wo = [[qv(0.6) for _ in range(HP)] for _ in range(D)]
    ws = [[qv(0.5) for _ in range(HP)] for _ in range(LP)]
    emit('w_in.hex', [wi[c][d] for c in range(LP) for d in range(D)])
    emit('w_out.hex', [wo[d][o] for d in range(D) for o in range(HP)])
    emit('w_skip.hex', [ws[c][o] for c in range(LP) for o in range(HP)])


def _biases() -> None:
    n = lcg(777001)

    def bv():
        return q(((n() % 2001) - 1000) / 1000.0 * 0.2, WQ)

    emit('b_in.hex', [bv() for _ in range(D)])
    emit('b_out.hex', [bv() for _ in range(HP)])
    emit('b_p.hex', [bv() for _ in range(D)])
    emit('b_p_1.hex', [bv() for _ in range(D)])


def _lattice_weights() -> None:
    n = lcg(424242)
    pad_kl = 1 << (KL - 1).bit_length()
    pad_lp = 1 << (P - 1).bit_length()

    def qv(scale=0.5):
        return q(((n() % 2001) - 1000) / 1000.0 * scale, WQ)

    dw = [0] * (D * pad_kl)
    for d in range(D):
        for k in range(KL):
            dw[d * pad_kl + k] = qv()
    wv = [0] * (RZ * pad_lp)
    for r in range(RZ):
        for ph in range(P):
            wv[r * pad_lp + ph] = qv()
    wu = [0] * (P * RZ)
    for ph in range(P):
        for r in range(RZ):
            wu[ph * RZ + r] = qv()
    wp = [0] * (D * D)
    for fo in range(D):
        for fi in range(D):
            wp[fo * D + fi] = qv(0.4)

    emit('w_dw_pad.hex', dw)
    emit('w_v_pad.hex', wv)
    emit('w_u.hex', wu)
    emit('w_p.hex', wp)

    # second pass: an independent draw, because the two blocks are not tied
    dw = [0] * (D * pad_kl)
    for d in range(D):
        for k in range(KL):
            dw[d * pad_kl + k] = qv()
    wv = [0] * (RZ * pad_lp)
    for r in range(RZ):
        for ph in range(P):
            wv[r * pad_lp + ph] = qv()
    wu = [0] * (P * RZ)
    for ph in range(P):
        for r in range(RZ):
            wu[ph * RZ + r] = qv()
    wp = [0] * (D * D)
    for fo in range(D):
        for fi in range(D):
            wp[fo * D + fi] = qv(0.4)

    emit('w_dw_pad_1.hex', dw)
    emit('w_v_pad_1.hex', wv)
    emit('w_u_1.hex', wu)
    emit('w_p_1.hex', wp)


def _main() -> None:
    import math

    _projections()
    _lattice_weights()
    _biases()
    n = lcg(20261022)
    xq, level = [], 0
    for t in range(L):
        level += (n() % 401) - 200
        xq.append(level + int(3000 * math.sin(2 * math.pi * t / P)) + (n() % 2001) - 1000)

    w = make_weights()
    yq, mean, std, inv = run(xq, w)

    emit('x_in.hex', xq)
    emit('y_ref.hex', yq)
    emit('w_out.hex', [w['out'][d][o] for d in range(D) for o in range(HP)])
    emit('w_skip.hex', [w['skip'][c][o] for c in range(LP) for o in range(HP)])
    print('x', min(xq), max(xq), 'mean', mean, 'std', std)


if __name__ == '__main__':
    _main()
