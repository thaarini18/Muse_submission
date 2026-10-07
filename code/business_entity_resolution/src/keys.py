"""Blocking keys: atomic tokens + composite (conjunctive) keys, hashed into 2^26 buckets.

Composite keys (e.g. name-token x house-number, house-number x street-word) are what keep the
inverted-index lists short: a surname or a house number alone is common, their conjunction is not.
"""
import zlib
from array import array
from multiprocessing import Pool

import numpy as np

from normalize import skeleton, ADDR_STOP, GENERIC

NBITS = 26
MASK = (1 << NBITS) - 1
EXTRA_NAME_STOP = {"com", "www", "net", "org", "in", "co", "praivet", "privet", "pryvett", "limitd", "limited",
                   "private", "prvt", "ltd", "llp", "elelpi", "elelpii", "pvt"}


def _h(s: str) -> int:
    return zlib.crc32(s.encode()) & MASK


def record_keys(nm_core: str, ad_tok: str, nm_ascii: str):
    core = [t for t in nm_core.split() if t not in EXTRA_NAME_STOP]
    N = []
    for t in core:
        s = skeleton(t)
        if len(s) >= 2 and s not in N:
            N.append(s)
    N_spec = [s for s, t in zip(N, core) if t not in GENERIC] or N
    concat = "".join(core)
    P = [concat[:6]] if len(concat) >= 4 else []
    atoks = ad_tok.split()
    D, W = [], []
    for t in atoks:
        if t.isdigit():
            t = t.lstrip("0") or "0"
            if t not in D:
                D.append(t)
        elif len(t) >= 3 and t not in ADDR_STOP:
            s = skeleton(t)
            if s not in W:
                W.append(s)
    D = D[:3]
    W = W[:8]
    keys = set()
    for s in N:
        keys.add("n" + s)
    for p in P:
        keys.add("p" + p)
    for d in D:
        keys.add("d" + d)
    for w in W:
        keys.add("w" + w)
    n4 = N[:4]
    for i in range(len(n4)):
        for j in range(i + 1, len(n4)):
            a, b = sorted((n4[i], n4[j]))
            keys.add("nn" + a + "|" + b)
    D2 = D[:2]
    for d in D2:
        for s in N_spec[:4]:
            keys.add("nd" + s + "|" + d)
        for p in P:
            keys.add("pd" + p + "|" + d)
        for w in W:
            keys.add("dw" + d + "|" + w)
    for p in P:
        for w in W:
            keys.add("pw" + p + "|" + w)
    for s in N_spec[:2]:
        for w in W:
            keys.add("nw" + s + "|" + w)
    return [_h(k) for k in keys]


def _chunk_keys(args):
    cores, addrs, asciis = args
    lens = array("i")
    out = array("i")
    for c, a, n in zip(cores, addrs, asciis):
        ks = record_keys(c, a, n)
        lens.append(len(ks))
        out.extend(ks)
    return np.frombuffer(lens, np.int32).copy(), np.frombuffer(out, np.int32).copy()


def build_keys(table, workers=2, chunk=100000):
    """table: pyarrow table with nm_core, ad_tok, nm_ascii. Returns (indptr int64, indices int32)."""
    n = table.num_rows
    lens_all, keys_all = [], []
    with Pool(workers, maxtasksperchild=10) as p:
        jobs = ((table["nm_core"].slice(i, chunk).to_pylist(), table["ad_tok"].slice(i, chunk).to_pylist(),
                 table["nm_ascii"].slice(i, chunk).to_pylist()) for i in range(0, n, chunk))
        for l, k in p.imap(_chunk_keys, jobs):
            lens_all.append(l)
            keys_all.append(k)
    lens = np.concatenate(lens_all)
    indptr = np.zeros(n + 1, np.int64)
    np.cumsum(lens, out=indptr[1:])
    return indptr, np.concatenate(keys_all)
