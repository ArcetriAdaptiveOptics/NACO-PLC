#!/usr/bin/env python3
"""
Check and visualize the PLC modal projection (VecMulMatFlat).

The PLC (SampleUdpEchoServer) receives a UDP packet with nModes float32
values, computes Commands = M2C * Modes, and sends back the nActs float32
commands. M2C is read at startup from m2cFilename on the PLC
(C:\\TwinCAT\\3.1\\Boot\\eye185.bin), a raw little-endian float32 file,
row-major, nActs rows x nModes columns (numpy: m2c.astype('<f4').tofile()).

Sub-commands:

  makebin   write a test matrix in the PLC file format
            python mvm_check.py makebin --kind random m2c.bin

  check     send a vector to the PLC, compare the reply with numpy, plot.
            --matrix must be the same file the PLC loaded (m2cFilename)
            python mvm_check.py check --host 192.168.1.10 --matrix m2c.bin

  emulate   offline: run the old (buggy) and the fixed ST algorithm on the
            same byte layout the PLC uses, and plot them against numpy
            python mvm_check.py emulate --matrix m2c.bin

A random (non-symmetric) matrix is the best test: an identity hides
transposition errors, and the PLC falls back to identity if the file
cannot be loaded (check bLoadError / nLoadErrId in the PLC).
"""

import argparse
import socket
import sys
import time

import numpy as np

N_MODES = 185
N_ACTS = 185
DTYPE = '<f4'


def make_matrix(kind, n_acts, n_modes, seed=0):
    if kind == 'eye':
        return np.eye(n_acts, n_modes, dtype=DTYPE)
    if kind == 'shift':
        # commands[j] = modes[j+1]: the output is the input moved left by one
        return np.eye(n_acts, n_modes, k=1, dtype=DTYPE)
    if kind == 'random':
        rng = np.random.default_rng(seed)
        return rng.standard_normal((n_acts, n_modes)).astype(DTYPE)
    raise ValueError(kind)


def make_vector(kind, n, seed=1):
    if kind == 'random':
        return np.random.default_rng(seed).standard_normal(n).astype(DTYPE)
    if kind == 'ramp':
        return np.linspace(-1, 1, n, dtype=DTYPE)
    if kind == 'ones':
        return np.ones(n, dtype=DTYPE)
    if kind.startswith('poke:'):
        v = np.zeros(n, dtype=DTYPE)
        v[int(kind[5:])] = 1.0
        return v
    raise ValueError(kind)


def load_matrix(path, n_acts, n_modes):
    m = np.fromfile(path, dtype=DTYPE)
    if m.size != n_acts * n_modes:
        sys.exit(f'{path}: {m.size} float32 values, expected {n_acts}*{n_modes}')
    return m.reshape(n_acts, n_modes)


def reference(m2c, vec):
    """Return (float64 result, float32 result summed in the PLC order, tolerance).

    The tolerance is the float32 rounding bound of a sequential dot product,
    (n+1) * eps * sum_i |M_ji * v_i|, doubled for safety: it scales with the
    size of the terms, not of the result, which can be small after cancellation.
    """
    m64 = m2c.astype(np.float64)
    v64 = vec.astype(np.float64)
    ref64 = m64 @ v64
    ref32 = np.zeros(m2c.shape[0], dtype=np.float32)
    for i in range(m2c.shape[1]):
        ref32 += m2c[:, i] * vec[i]
    eps = 2.0 ** -24
    tol = 2 * (m2c.shape[1] + 1) * eps * (np.abs(m64) @ np.abs(v64)) + np.finfo(np.float32).tiny
    return ref64, ref32, tol


def emulate_old(m2c, vec):
    """Original VecMulMatFlat: 'p := p + 1' advances ONE BYTE in TwinCAT."""
    vbytes = vec.tobytes()
    mbytes = m2c.tobytes()
    n_cols, n_vec = m2c.shape

    def real_at(buf, off):
        b = buf[off:off + 4]
        return np.frombuffer(b, dtype=DTYPE)[0] if len(b) == 4 else np.float32(0)

    res = np.zeros(n_cols, dtype=np.float32)
    pm = 0
    with np.errstate(all='ignore'):
        for j in range(n_cols):
            s = np.float32(0)
            pv = 0
            for _ in range(n_vec):
                s = np.float32(s + real_at(vbytes, pv) * real_at(mbytes, pm))
                pv += 1
                pm += 1
            res[j] = s
    return res


def emulate_new(m2c, vec):
    """Fixed VecMulMatFlat: res[j] = sum_i pMat[j*nVec + i] * pVec[i]."""
    flat = m2c.ravel()
    n_cols, n_vec = m2c.shape
    res = np.zeros(n_cols, dtype=np.float32)
    for j in range(n_cols):
        s = np.float32(0)
        for i in range(n_vec):
            s = np.float32(s + flat[j * n_vec + i] * vec[i])
        res[j] = s
    return res


def plc_roundtrip(host, port, vec, n_acts, timeout):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.settimeout(timeout)
        t0 = time.perf_counter()
        s.sendto(vec.astype(DTYPE).tobytes(), (host, port))
        data, _ = s.recvfrom(65536)
        dt = time.perf_counter() - t0
    if len(data) != n_acts * 4:
        sys.exit(f'PLC replied {len(data)} bytes ({data[:16]!r}), expected {n_acts * 4}:'
                 ' is the fixed PLC code running?')
    return np.frombuffer(data, dtype=DTYPE).copy(), dt


def report(name, got, m2c, vec, diagnose=True):
    ref64, ref32, tol = reference(m2c, vec)
    diff = got.astype(np.float64) - ref64
    bad = ~(np.abs(diff) <= tol)            # NaN/inf count as bad
    ok = not bad.any()
    with np.errstate(all='ignore'):
        rel = np.nanmax(np.abs(diff) / tol)
    print(f'{name:>8s}: max|res-numpy| = {np.nanmax(np.abs(diff)):.3e}, '
          f'max error / float32 bound = {rel:.3g}, '
          f'max|res-float32 same order| = {np.nanmax(np.abs(got - ref32)):.3e}  '
          f'-> {"OK" if ok else f"WRONG ({bad.sum()}/{bad.size} entries)"}')
    if not ok and diagnose:
        explain(got, m2c, vec, ref64, tol, bad)
    return ok


def explain(got, m2c, vec, ref64, tol, bad):
    """Print the bad entries and test the usual explanations."""
    idx = np.flatnonzero(bad)
    for j in idx[:5]:
        print(f'            [{j:3d}] got {got[j]: .7g}  expected {ref64[j]: .7g}  '
              f'diff {got[j] - ref64[j]: .3g}  tol {tol[j]:.2g}')
    if idx.size > 5:
        print(f'            ... bad indices: {idx[:20].tolist()}{" ..." if idx.size > 20 else ""}')

    def close(a, b):
        with np.errstate(all='ignore'):
            a = np.asarray(a, dtype=np.float64)
        return a.shape == b.shape and np.all(np.abs(a - b) <= tol + 1e-6 * np.abs(b))

    n_acts, n_modes = m2c.shape
    candidates = {
        'the identity: the PLC uses an identity (e.g. eye185.bin, or the fallback when'
        ' the load fails: check bLoadError/nLoadErrId) but --matrix is a different file':
            np.eye(n_acts, n_modes) @ vec,
        'zero': np.zeros(n_acts),
        'the old VecMulMatFlat: the fixed PLC code is not running': emulate_old(m2c, vec),
    }
    if n_acts == n_modes:
        candidates['M2C transposed: the file on the PLC is column-major (e.g. IDL/Fortran'
                   ' order, or numpy .T), or --matrix is not the file on the PLC'] = m2c.T @ vec
    with np.errstate(all='ignore'):
        candidates['byte-swapped floats in the reply'] = ref64.astype('>f4').view('<f4')
    found = False
    for what, cand in candidates.items():
        if close(cand, got):
            print(f'            -> the PLC reply equals {what}')
            found = True
    if not found and idx[0] > 0 and not bad[:idx[0]].any():
        k = idx[0]
        print(f'            -> entries 0..{k - 1} are right and the first wrong one is {k}: '
              f'the matrix on the PLC differs from row {k} on '
              f'(byte {k * n_modes * 4} of the file). Partial load? Check bLoadError, '
              f'fbLoad.cbRead, and that --matrix is the same file copied to the PLC')
    elif not found:
        print('            -> no known pattern: compare --matrix with the file on the PLC '
              '(size must be exactly %d bytes, float32 little-endian, row-major)'
              % (n_acts * n_modes * 4))


def plot(vec, ref64, tol, results, title, matrix_name, out=None):
    import matplotlib
    if out:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    ax[0].plot(vec, '.-', color='0.4')
    ax[0].set_ylabel('input modes')

    ax[1].plot(ref64, '-', color='k', lw=2, label=f'numpy  M2C @ modes  (M2C: {matrix_name})')
    for (name, res), c in zip(results.items(), ('tab:red', 'tab:blue')):
        shown = np.where(np.isfinite(res), res, np.nan)
        nbad = np.count_nonzero(np.isnan(shown))
        if nbad:
            name += f' ({nbad} NaN/inf not shown)'
        ax[1].plot(shown, 'o', ms=3, color=c, label=name)
        ax[2].plot(shown - ref64, 'o', ms=3, color=c, label=name)
    lim = 3 * np.abs(ref64).max()
    if lim > 0:
        ax[1].set_ylim(-lim, lim)
    ax[1].set_ylabel('commands')
    ax[1].legend(loc='upper right')
    ax[2].fill_between(np.arange(len(tol)), -tol, tol, color='0.85', label='float32 bound')
    ax[2].set_yscale('symlog', linthresh=max(tol.max(), 1e-12))
    ax[2].axhline(0, color='k', lw=0.5)
    ax[2].set_ylabel('result - numpy')
    ax[2].set_xlabel('index')
    ax[2].legend(loc='upper right')
    fig.suptitle(title)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=100)
        print(f'plot saved to {out}')
    else:
        plt.show()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--modes', type=int, default=N_MODES)
    p.add_argument('--acts', type=int, default=N_ACTS)
    sub = p.add_subparsers(dest='cmd', required=True)

    pm = sub.add_parser('makebin', help='write a test matrix file')
    pm.add_argument('out')
    pm.add_argument('--kind', choices=['random', 'eye', 'shift'], default='random')
    pm.add_argument('--seed', type=int, default=0)

    for name in ('check', 'emulate'):
        pc = sub.add_parser(name)
        if name == 'check':
            pc.add_argument('--matrix', required=True,
                            help='the same matrix file the PLC loaded (m2cFilename)')
        else:
            pc.add_argument('--matrix', help='matrix file (default: random, seed 0)')
        pc.add_argument('--vector', default='random',
                        help='random | ramp | ones | poke:K (default random)')
        pc.add_argument('--seed', type=int, default=1, help='seed for the random vector')
        pc.add_argument('--plot', nargs='?', const='', default=None, metavar='PNG',
                        help='show a plot (or save it to PNG)')
        if name == 'check':
            pc.add_argument('--host', required=True)
            pc.add_argument('--port', type=int, default=10000)
            pc.add_argument('--timeout', type=float, default=1.0)
            pc.add_argument('--count', type=int, default=1,
                            help='repeat with COUNT different random vectors')

    a = p.parse_args()

    if a.cmd == 'makebin':
        m = make_matrix(a.kind, a.acts, a.modes, a.seed)
        m.astype(DTYPE).tofile(a.out)
        print(f'wrote {a.out}: {a.kind} {a.acts}x{a.modes} float32 row-major, {m.nbytes} bytes')
        return

    m2c = (load_matrix(a.matrix, a.acts, a.modes) if a.matrix
           else make_matrix('random', a.acts, a.modes, 0))
    print(f'matrix: {a.matrix or "random (seed 0)"}')
    vec = make_vector(a.vector, a.modes, a.seed)
    ref64, _, tol = reference(m2c, vec)

    if a.cmd == 'emulate':
        results = {'old ST code': emulate_old(m2c, vec), 'fixed ST code': emulate_new(m2c, vec)}
        for k, r in results.items():
            report(k, r, m2c, vec, diagnose=False)
        title = 'VecMulMatFlat emulation'
    else:
        all_ok = True
        for n in range(a.count):
            v = vec if n == 0 else make_vector('random', a.modes, a.seed + n)
            got, dt = plc_roundtrip(a.host, a.port, v, a.acts, a.timeout)
            all_ok &= report(f'#{n}', got, m2c, v)
            print(f'          round trip {dt * 1e3:.2f} ms')
            if n == 0:
                results = {'PLC': got}
        title = f'PLC {a.host}:{a.port}  ' + ('OK' if all_ok else 'MISMATCH')

    if a.plot is not None:
        plot(vec, ref64, tol, results, title, a.matrix or 'random, seed 0', a.plot or None)


if __name__ == '__main__':
    main()
