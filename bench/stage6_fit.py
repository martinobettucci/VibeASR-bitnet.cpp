#!/usr/bin/env python3
"""Fit the stage-6 truncation projections and write them for the runtime.

    python3 bench/stage6_fit.py /tmp/tap2.bin --keep 4 --out models/vibeasr/stage6_proj.bin

Reads block outputs recorded by VIBEASR_TAP_STAGE (see bench/stage6_calib.py for the
study that motivates this), fits an affine map from block K's output to block 8's by
ridge regression, and writes it in the I8_S layout the runtime already uses for every
other weight: int8 values with one float scale appended, so the projection runs on
the same fused kernel as the FFN linears it replaces.

File format (little endian), matching what vae.cpp's loader expects:

    magic   "V6PJ"        4 bytes
    keep    int32         blocks retained before the projection
    dim     int32         2048
    then, acoustic followed by semantic:
        int8[dim*dim]     weights, ggml [in, out] order
        float32           dequant scale (max_abs / 127), as quantize_i8_s stores it
        float32[dim]      bias
"""

import argparse
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage6_calib import cosines, load  # noqa: E402


def fit_affine(X, Y, lam):
    """Ridge with an intercept: the blocks carry biases, so an affine map is the
    right family. The intercept column is left unpenalised."""
    n, d = X.shape
    Xa = np.hstack([X, np.ones((n, 1))])
    G = Xa.T @ Xa
    reg = np.eye(d + 1) * lam
    reg[d, d] = 0.0
    Wa = np.linalg.solve(G + reg, Xa.T @ Y)
    return Wa[:d], Wa[d]


def quantize_i8s(W):
    """Match quantize_i8_s in src/ggml-lm-mad.cpp: symmetric per-tensor, clamp +-127,
    and store max_abs/127 as the dequant scale."""
    amax = float(np.abs(W).max())
    scale = 127.0 / amax if amax > 0 else 1.0
    q = np.clip(np.rint(W * scale), -127, 127).astype(np.int8)
    return q, np.float32(1.0 / scale)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tap")
    ap.add_argument("--keep", type=int, default=4)
    ap.add_argument("--lam", type=float, default=1e2)
    ap.add_argument("--out", default="models/vibeasr/stage6_proj.bin")
    ap.add_argument("--holdout", type=float, default=0.15)
    args = ap.parse_args()

    passes = load(args.tap)
    nb, _, dim = passes[0].shape
    print("%d passes, %d blocks, dim %d" % (len(passes), nb, dim))
    if not 1 <= args.keep < nb:
        sys.exit("--keep must be in [1, %d)" % nb)

    blobs = []
    for name, sel in (("acoustic", 0), ("semantic", 1)):
        # Select on the GLOBAL pass index; see stage6_calib.py. Slicing then
        # enumerating restarts the counter and swaps encoders on an odd cut.
        own = [i for i in range(len(passes)) if i % 2 == sel]
        ocut = int(len(own) * (1.0 - args.holdout))
        tr = [passes[i] for i in own[:ocut]]
        te = [passes[i] for i in own[ocut:]]
        Xtr = np.concatenate([p[args.keep - 1] for p in tr]).astype(np.float64)
        Ytr = np.concatenate([p[nb - 1] for p in tr]).astype(np.float64)
        W, b = fit_affine(Xtr, Ytr, args.lam)

        Xte = np.concatenate([p[args.keep - 1] for p in te]).astype(np.float64)
        Yte = np.concatenate([p[nb - 1] for p in te]).astype(np.float64)
        c_f32 = cosines(Xte @ W + b, Yte)

        q, scale = quantize_i8s(W)
        c_i8 = cosines(Xte @ (q.astype(np.float64) * float(scale)) + b, Yte)
        print("  %-9s keep %d/%d: %d train frames, cosine %.4f f32 -> %.4f after int8"
              % (name, args.keep, nb, Xtr.shape[0], c_f32, c_i8))

        # ggml wants [in, out] with ne0 = in contiguous; numpy W is [in, out] C-order,
        # so transposing before serialising puts element (in=i, out=j) at j*in + i.
        blobs.append(q.T.copy().tobytes())
        blobs.append(struct.pack("<f", float(scale)))
        blobs.append(b.astype(np.float32).tobytes())

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(b"V6PJ")
        f.write(struct.pack("<ii", args.keep, dim))
        for blob in blobs:
            f.write(blob)
    print("wrote %s (%.1f MB)" % (args.out, os.path.getsize(args.out) / 1e6))


if __name__ == "__main__":
    main()
