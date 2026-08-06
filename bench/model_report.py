#!/usr/bin/env python3
"""Report where a GGUF's bytes go, and the effective bits per weight.

    python3 bench/model_report.py models/vibeasr/*.gguf

Sizes come from the tensor offsets recorded in the file, so the numbers hold for
types whose packing ggml_row_size does not describe (I2_S, I8_S, TL2) rather than
relying on a nominal type size.
"""

import collections
import os
import struct
import sys

# GGML type id -> name, for the ids this project uses.
TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K",
    24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
    30: "BF16", 36: "I2_S", 37: "I8_S", 38: "TL1", 39: "TL2",
}

TERNARY_ENTROPY = 1.5849625007211562  # log2(3): the floor for a ternary weight


def read_gguf(path):
    f = open(path, "rb")
    if f.read(4) != b"GGUF":
        sys.exit("%s: not a GGUF" % path)
    struct.unpack("<I", f.read(4))
    n_tensors = struct.unpack("<Q", f.read(8))[0]
    n_kv = struct.unpack("<Q", f.read(8))[0]

    def rstr():
        n = struct.unpack("<Q", f.read(8))[0]
        return f.read(n).decode("utf-8", "replace")

    def rval(t):
        simple = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
                  6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
        if t in simple:
            fmt = simple[t]
            return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
        if t == 8:
            return rstr()
        if t == 9:
            et = struct.unpack("<I", f.read(4))[0]
            n = struct.unpack("<Q", f.read(8))[0]
            return [rval(et) for _ in range(n)]
        raise ValueError("gguf value type %d" % t)

    kv = {}
    for _ in range(n_kv):
        k = rstr()
        kv[k] = rval(struct.unpack("<I", f.read(4))[0])

    tensors = []
    for _ in range(n_tensors):
        name = rstr()
        nd = struct.unpack("<I", f.read(4))[0]
        dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
        tt = struct.unpack("<I", f.read(4))[0]
        off = struct.unpack("<Q", f.read(8))[0]
        tensors.append([name, dims, tt, off, 0])

    align = kv.get("general.alignment", 32)
    pos = f.tell()
    if pos % align:
        pos += align - pos % align
    data_off = pos
    total = os.path.getsize(path)
    f.close()

    # Size each tensor from the gap to the next one; the last runs to end of file.
    order = sorted(range(len(tensors)), key=lambda i: tensors[i][3])
    for k, i in enumerate(order):
        end = tensors[order[k + 1]][3] if k + 1 < len(order) else total - data_off
        tensors[i][4] = end - tensors[i][3]
    return kv, tensors, data_off, total


def classify(name):
    if "token_embd" in name:
        return "token embedding"
    if name.startswith("output.") and "norm" not in name:
        return "output projection"
    if "norm" in name or name.endswith(".bias"):
        return "norms / biases"
    return "transformer body"


def main():
    for path in sys.argv[1:]:
        kv, tensors, data_off, total = read_gguf(path)
        print("=" * 78)
        print("%s   %.1f MB   %d tensors" % (os.path.basename(path), total / 1e6, len(tensors)))
        print("=" * 78)

        groups = collections.OrderedDict()
        for name, dims, tt, off, size in tensors:
            n = 1
            for d in dims:
                n *= d
            key = (classify(name), TYPES.get(tt, "type%d" % tt))
            g = groups.setdefault(key, [0, 0, 0])
            g[0] += n
            g[1] += size
            g[2] += 1

        print("%-20s %-7s %6s %14s %10s %8s" %
              ("component", "type", "count", "weights", "MB", "bits/wt"))
        print("-" * 70)
        tw = tb = 0
        for (comp, ty), (n, size, cnt) in groups.items():
            print("%-20s %-7s %6d %14d %10.1f %8.2f" % (comp, ty, cnt, n, size / 1e6, 8.0 * size / n))
            tw += n
            tb += size
        print("-" * 70)
        print("%-20s %-7s %6d %14d %10.1f %8.2f" % ("total", "", len(tensors), tw, tb / 1e6, 8.0 * tb / tw))

        ternary = sum(s for nm, d, tt, o, s in tensors if TYPES.get(tt) in ("I2_S", "TL1", "TL2"))
        ternary_w = 0
        for nm, dims, tt, o, s in tensors:
            if TYPES.get(tt) in ("I2_S", "TL1", "TL2"):
                n = 1
                for d in dims:
                    n *= d
                ternary_w += n
        if ternary_w:
            bpw = 8.0 * ternary / ternary_w
            floor = ternary_w * TERNARY_ENTROPY / 8 / 1e6
            print("\nternary weights: %d at %.3f bits/wt (%.1f MB)" % (ternary_w, bpw, ternary / 1e6))
            print("  log2(3) = %.3f bits/wt would be %.1f MB -- %.1f MB of padding (%.1f%%)"
                  % (TERNARY_ENTROPY, floor, ternary / 1e6 - floor,
                     100.0 * (1 - floor / (ternary / 1e6))))
        print()


if __name__ == "__main__":
    main()
