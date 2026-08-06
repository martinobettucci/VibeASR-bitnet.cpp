"""Seekable file object that reads a remote file with parallel HTTP range requests.

Pulling a full FLEURS test parquet (0.5-1 GB) over one long-lived connection runs at
about 1 MB/s from here, while several short ranged GETs in flight together sustain
~60 MB/s. This wraps a URL so pyarrow can open it directly and only the byte ranges it
asks for get fetched -- in parallel chunks.
"""

import io
import os
import urllib.request
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

CHUNK = 8 << 20     # block size; also the size of each ranged GET
WORKERS = 8         # blocks fetched concurrently -- this is where the speedup lives
CACHE_BLOCKS = 24   # LRU bound, keeps peak memory near 200 MB
RETRIES = 4


class RangeFile(io.RawIOBase):
    def __init__(self, url, token=None):
        self.token = token
        self.pos = 0
        self.url, self.size = self._resolve(url)
        self._pool = ThreadPoolExecutor(WORKERS)
        self._blocks = OrderedDict()

    def _req(self, url, **kw):
        r = urllib.request.Request(url, **kw)
        if self.token:
            r.add_header("Authorization", "Bearer " + self.token)
        return r

    def _resolve(self, url):
        """Follow redirects once so every later range GET hits the CDN directly."""
        with urllib.request.urlopen(self._req(url, method="HEAD"), timeout=60) as r:
            return r.url, int(r.headers["Content-Length"])

    def _fetch_block(self, idx):
        start = idx * CHUNK
        stop = min(start + CHUNK, self.size) - 1
        for attempt in range(RETRIES):
            try:
                r = self._req(self.url)
                r.add_header("Range", "bytes=%d-%d" % (start, stop))
                with urllib.request.urlopen(r, timeout=180) as f:
                    return idx, f.read()
            except Exception:
                if attempt == RETRIES - 1:
                    raise
        raise AssertionError("unreachable")

    def _ensure(self, first, last):
        """Make blocks [first, last] resident, fetching misses WORKERS at a time.

        pyarrow reads a parquet column chunk in many modest sequential steps. Fetching
        only what each step asks for would serialise the network, so on any miss we pull
        a whole WORKERS-wide window ahead in parallel.
        """
        missing = [i for i in range(first, last + 1) if i not in self._blocks]
        if not missing:
            return
        nblocks = (self.size + CHUNK - 1) // CHUNK
        ahead = min(missing[-1] + WORKERS, nblocks - 1)
        missing += [i for i in range(missing[-1] + 1, ahead + 1) if i not in self._blocks]
        for idx, data in self._pool.map(self._fetch_block, missing):
            self._blocks[idx] = data
        for i in range(first, last + 1):
            self._blocks.move_to_end(i)
        keep = set(range(first, last + 1))
        while len(self._blocks) > CACHE_BLOCKS:
            idx, data = self._blocks.popitem(last=False)
            if idx in keep:  # never drop a block the caller is about to read
                self._blocks[idx] = data
                self._blocks.move_to_end(idx)
                break

    # -- file protocol --
    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, offset, whence=os.SEEK_SET):
        if whence == os.SEEK_SET:
            self.pos = offset
        elif whence == os.SEEK_CUR:
            self.pos += offset
        else:
            self.pos = self.size + offset
        return self.pos

    def tell(self):
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        out = bytearray()
        pos, end = self.pos, self.pos + n
        span = (CACHE_BLOCKS - WORKERS) * CHUNK  # cap a single pass so the LRU holds it
        while pos < end:
            stop = min(pos + span, end)
            first, last = pos // CHUNK, (stop - 1) // CHUNK
            self._ensure(first, last)
            for i in range(first, last + 1):
                blk = self._blocks[i]
                lo = max(pos, i * CHUNK) - i * CHUNK
                hi = min(stop, (i + 1) * CHUNK) - i * CHUNK
                out += blk[lo:hi]
            pos = stop
        self.pos += len(out)
        return bytes(out)

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)

    def close(self):
        self._pool.shutdown(wait=False)
        super().close()


def fleurs_url(lang, split="test"):
    return "https://huggingface.co/api/datasets/google/fleurs/parquet/%s/%s/0.parquet" % (lang, split)
