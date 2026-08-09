#!/usr/bin/env python3
"""Can the VAE's last stage be truncated and linearly projected back?

The final encoder stage is 8 ConvNeXt blocks at dim 2048: 52% of the encoder's MACs
and 89% of its FFN parameters. If block K's output linearly predicts block 8's, the
remaining blocks can be replaced by one 2048x2048 matrix -- the trick from the H3
text-encoder swap, applied within a single model where alignment is trivial.

Usage:
    VIBEASR_TAP_STAGE=/tmp/tap.bin ./build/bin/asr_infer ... # collect, repeat per clip
    python3 bench/stage6_calib.py /tmp/tap.bin

Calibration is ridge regression, as in the H3 work: accumulate X'X and X'Y over the
corpus, solve once. No gradients.

Reported per candidate K:
  cosine     mean per-frame cosine between predicted and true block-8 output, on a
             held-out split -- fit on 80% of clips, scored on the other 20%
  identity   the same, for W = I (truncate and change nothing). This is the control
             that matters: if identity already scores well the blocks were near
             no-ops and the matrix earns nothing.
  zero       W = 0. Floor.
"""

import struct
import sys

import numpy as np


def load(path):
    """-> list of clips, each an array [n_blocks, frames, dim]. Encoders interleave."""
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    recs, off = [], 0
    total = len(raw)
    while off < total:
        b, dim, fr = struct.unpack("<3i", raw[off:off + 12].tobytes())
        off += 12
        n = dim * fr
        vals = np.frombuffer(raw[off:off + 4 * n].tobytes(), dtype=np.float32).reshape(fr, dim)
        off += 4 * n
        recs.append((b, vals))
    # Split into runs of increasing block index; each run is one encoder pass.
    passes, cur = [], []
    for b, v in recs:
        if b == 0 and cur:
            passes.append(cur)
            cur = []
        cur.append(v)
    if cur:
        passes.append(cur)
    return [np.stack(p) for p in passes if len(p) == len(passes[0])]


def fit_ridge(X, Y, lam):
    d = X.shape[1]
    XtX = X.T @ X
    XtY = X.T @ Y
    return np.linalg.solve(XtX + lam * np.eye(d, dtype=np.float64), XtY)


def cosines(P, Y):
    pn = np.linalg.norm(P, axis=1)
    yn = np.linalg.norm(Y, axis=1)
    ok = (pn > 1e-9) & (yn > 1e-9)
    return float(np.mean(np.sum(P[ok] * Y[ok], axis=1) / (pn[ok] * yn[ok])))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/tap.bin"
    lam = float(sys.argv[2]) if len(sys.argv) > 2 else 1e2

    passes = load(path)
    nb, _, dim = passes[0].shape
    print("%d encoder passes, %d blocks, dim %d, %d frames total"
          % (len(passes), nb, dim, sum(p.shape[1] for p in passes)))

    # Hold out whole passes, not frames: frames within a clip are highly correlated
    # and a frame-level split would report a number the deployment never sees.
    # The two encoders (acoustic, semantic) have different weights and interleave in
    # the dump. They must be calibrated separately -- one matrix serving both is not
    # what deployment would use, and pooling them understates what is achievable.
    for name, sel in (("acoustic", 0), ("semantic", 1)):
        # Select on the GLOBAL pass index. Slicing first and enumerating the slice
        # restarts the counter at 0, so an odd cut silently swaps the two encoders
        # and scores each against the other's targets.
        own = [i for i in range(len(passes)) if i % 2 == sel]
        ocut = int(len(own) * 0.8)
        tr = [passes[i] for i in own[:ocut]]
        te = [passes[i] for i in own[ocut:]]
        if not tr or not te:
            continue
        Ytr = np.concatenate([p[nb - 1] for p in tr]).astype(np.float64)
        Yte = np.concatenate([p[nb - 1] for p in te]).astype(np.float64)
        print("\n%s encoder  (%d train / %d test passes, %d train frames)"
              % (name, len(tr), len(te), Ytr.shape[0]))
        print("  %-6s %10s %10s   %s" % ("keep", "learned", "identity", "verdict"))
        for k in range(1, nb):
            Xtr = np.concatenate([p[k - 1] for p in tr]).astype(np.float64)
            Xte = np.concatenate([p[k - 1] for p in te]).astype(np.float64)
            W = fit_ridge(Xtr, Ytr, lam)
            c_learn = cosines(Xte @ W, Yte)
            c_ident = cosines(Xte, Yte)
            gain = c_learn - c_ident
            verdict = ("matrix earns %+.3f" % gain) if gain > 0.005 else "matrix adds nothing"
            print("  %-6s %10.4f %10.4f   %s" % ("%d/8" % k, c_learn, c_ident, verdict))


if __name__ == "__main__":
    main()
