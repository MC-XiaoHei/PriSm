"""Integer FITS: instance norm, low-pass the spectrum, inverse transform.

The transform is a direct sum over the bins that survive the low-pass, which is
exactly what a mixed-radix FFT would compute; 96 = 2^5*3 has no radix-2
factorisation, so the sum is done literally and the hardware matches it term for
term.
"""
import math
import pathlib

L = 96          # look-back
H = 96          # forecast length
NBIN = 15       # bins kept: k = 0 .. NBIN-1
Q = 12          # data fraction bits
WQ = 14         # constant fraction bits
OUT = pathlib.Path('.')


def rr(v, bits=WQ):
    return int((v + (1 << (bits - 1))) >> bits)


def q(v, bits):
    return int(round(v * (1 << bits)))


def isqrt(n):
    if n <= 0:
        return 0
    x = int(math.isqrt(n))
    return x


def clamp16(v):
    return 32767 if v > 32767 else (-32768 if v < -32768 else v)


def normalise(xq):
    mean = int(sum(xq) / L)
    var = int(sum(v * v for v in xq) / L) - mean * mean
    std = isqrt(max(var, 0))
    if std == 0:
        std = 1
    return mean, std, (1 << 24) // std


def cos_sin_tables():
    """Constants for the bins that survive, k >= 1."""
    c = [[q(math.cos(2 * math.pi * k * n / L), WQ) for n in range(L)] for k in range(NBIN)]
    s = [[q(math.sin(2 * math.pi * k * n / L), WQ) for n in range(L)] for k in range(NBIN)]
    return c, s


def forward(seq, c, s):
    """Bins at Q14: the accumulator is scaled down once so the inverse taps can
    use a 32x16 multiply instead of a 44x16 one."""
    xre = [0] * NBIN
    xim = [0] * NBIN
    xre[0] = rr(sum(seq) << WQ, 12)          # cos is exactly one
    for k in range(1, NBIN):
        xre[k] = rr(sum(seq[n] * c[k][n] for n in range(L)), 12)
        xim[k] = rr(-sum(seq[n] * s[k][n] for n in range(L)), 12)
    return xre, xim


def inverse(xre, xim, c, s):
    """y[n] at Q12: the sum is Q28, so 16 bits come off before the divide."""
    out = []
    for n in range(L):
        tot = xre[0] << WQ
        for k in range(1, NBIN):
            tot += 2 * (xre[k] * c[k][n] - xim[k] * s[k][n])
        # the hardware divides magnitudes and puts the sign back, so the
        # quotient truncates toward zero rather than flooring
        v = rr(tot, 16)
        out.append((abs(v) // L) * (1 if v >= 0 else -1))
    return out


def run(xq):
    mean, std, inv = normalise(xq)
    seq = [((v - mean) * inv) >> 12 for v in xq]
    c, s = cos_sin_tables()
    xre, xim = forward(seq, c, s)
    y = inverse(xre, xim, c, s)
    # y is at Q26; back to Q12 before the window scale goes on
    out = [clamp16(((y[n] * std) >> 12) + mean) for n in range(L)]
    return out[-H:], mean, std, inv


def emit(name, vals):
    (OUT / name).write_text('\n'.join('%04x' % (v & 0xFFFF) for v in vals) + '\n')


def load_hex(path):
    vals = []
    for tok in pathlib.Path(path).read_text().split():
        v = int(tok, 16) & 0xFFFF
        vals.append(v - 0x10000 if v >= 0x8000 else v)
    return vals


if __name__ == '__main__':
    n = 20261024
    xq = []
    level = 0
    for t in range(L):
        level += (n % 401) - 200
        n = (1103515245 * n + 12345) & 0x7fffffff
        xq.append(level + int(3000 * math.sin(2 * math.pi * t / 24)) + (n % 2001) - 1000)
    out, mean, std, inv = run(xq)
    emit('fits_x_in.hex', xq)
    emit('fits_y_ref.hex', out)
    c, s = cos_sin_tables()
    emit('fits_cos.hex', [c[k][t] for k in range(NBIN) for t in range(L)])
    emit('fits_sin.hex', [s[k][t] for k in range(NBIN) for t in range(L)])
    print('mean', mean, 'std', std, 'inv', inv)
    print('out', out[:5])
