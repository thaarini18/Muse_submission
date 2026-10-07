"""Stage 2: candidate generation (blocking).

Per country partition:
  1. every record is represented by a bag of blocking tokens (phonetic name tokens,
     concatenated-name prefix, address numbers, phonetic address words)  -- see prep.py
  2. an inverted index over Source2+Source3 is queried with each Source1 record via a
     sparse matrix product A(S1 x V, idf weights) @ B(V x R, binary); tokens that are too frequent
     (df > DF_CAP) are ignored -> the cost is bounded, this is plain inverted-index blocking
  3. keep the top-K records per S1 entity (score = summed idf of shared tokens) and apply a
     relative-score cut (score >= REL * best score)
  4. reverse filter: every S2/S3 record belongs to at most one real entity, so each record is kept
     only for its top-M Source1 entities.
Output: DataFrame (s1_idx, r_idx, bscore, brank, brank_rev) with global row indices.
"""
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pcmp
import scipy.sparse as sp
from sparse_dot_topn import sp_matmul_topn
from keys import NBITS


def _sub_csr(indptr, indices, rows):
    """Row subset (rows sorted ascending) of a CSR given as (indptr, indices); memory-light."""
    n = len(indptr) - 1
    lens = np.diff(indptr)
    row_mask = np.zeros(n, bool)
    row_mask[rows] = True
    ent_mask = np.repeat(row_mask, lens)
    sub_ptr = np.zeros(len(rows) + 1, np.int64)
    np.cumsum(lens[rows], out=sub_ptr[1:])
    return sub_ptr, indices[ent_mask]


def block_country(s1_keys, r_keys, k=20, df_cap=2000, n_threads=2, s1_chunk=100000, r_chunk=1000000,
                  mode="sum", min_mass=30.0):
    """s1_keys / r_keys: (indptr, bucket indices) for one country. Returns (s1_local, r_local, score).

    mode="sum":     score = summed idf of shared keys
    mode="contain": score = summed idf of shared keys / max(total idf of the S2/S3 record, min_mass)
                    (how much of a short record - e.g. one with an empty address - is explained by S1)

    Exact top-k of A @ B.T computed over R in column blocks (top-k of the union of per-block top-k),
    which keeps peak memory bounded regardless of how many records a country has."""
    r_ptr, r_idx = r_keys
    s_ptr, s_idx = s1_keys
    n_r, n_s = len(r_ptr) - 1, len(s_ptr) - 1
    df = np.bincount(r_idx, minlength=1 << NBITS).astype(np.int32)
    keep = (df > 0) & (df <= df_cap)
    kept = np.flatnonzero(keep)
    V = len(kept)
    remap = np.full(1 << NBITS, V, np.int32)          # dropped buckets -> dummy column V
    remap[kept] = np.arange(V, dtype=np.int32)
    idf = np.log1p(n_r / df[kept]).astype(np.float32)
    idf_ext = np.append(idf, np.float32(0))
    del df, keep, kept
    # S1 side (idf weights), dropped buckets removed entirely
    sc_ = remap[s_idx]
    srow = np.repeat(np.arange(n_s, dtype=np.int32), np.diff(s_ptr))
    m = sc_ < V
    A = sp.csr_matrix((idf[sc_[m]], (srow[m], sc_[m])), shape=(n_s, V + 1))
    del sc_, srow, m
    A.sum_duplicates()
    # running dense top-k per S1 row
    best_v = np.zeros((n_s, k), np.float32)
    best_r = np.full((n_s, k), -1, np.int32)
    for rs in range(0, n_r, r_chunk):
        re_ = min(n_r, rs + r_chunk)
        p0, p1 = r_ptr[rs], r_ptr[re_]
        cols = remap[r_idx[p0:p1]]
        B = sp.csr_matrix((np.ones(len(cols), np.float32), cols, (r_ptr[rs:re_ + 1] - p0)),
                          shape=(re_ - rs, V + 1))
        del cols
        B.sum_duplicates()
        B.data[:] = 1.0
        if mode == "contain":
            mass = np.asarray(B @ idf_ext).ravel()
            B.data[:] = np.repeat(1.0 / np.maximum(mass, min_mass), np.diff(B.indptr)).astype(np.float32)
        BT = B.T.tocsr()
        del B
        for st in range(0, n_s, s1_chunk):
            en = min(n_s, st + s1_chunk)
            C = sp_matmul_topn(A[st:en], BT, top_n=k, n_threads=n_threads)
            cnt = np.diff(C.indptr)
            pos = np.arange(C.nnz) - np.repeat(C.indptr[:-1], cnt)
            rows = np.repeat(np.arange(en - st), cnt)
            nv = np.zeros((en - st, k), np.float32)
            nr = np.full((en - st, k), -1, np.int32)
            nv[rows, pos] = C.data
            nr[rows, pos] = C.indices + rs
            allv = np.concatenate([best_v[st:en], nv], axis=1)
            allr = np.concatenate([best_r[st:en], nr], axis=1)
            top = np.argsort(-allv, axis=1, kind="stable")[:, :k]
            best_v[st:en] = np.take_along_axis(allv, top, axis=1)
            best_r[st:en] = np.take_along_axis(allr, top, axis=1)
            del C, nv, nr, allv, allr, top
        del BT
    del A
    m = (best_r >= 0) & (best_v > 0)
    a = np.repeat(np.arange(n_s, dtype=np.int32), k).reshape(n_s, k)[m]
    return a, best_r[m], best_v[m]


def country_codes(col, countries):
    """Integer code per row (index into `countries`, -1 if absent) without materialising Python strings."""
    return pcmp.index_in(col, value_set=pa.array(countries)).fill_null(-1).to_numpy().astype(np.int16)


def _group_rank(g, score):
    """Rank (1 = best) of score within group g, via lexsort (memory-light)."""
    order = np.lexsort((-score, g))
    gs = g[order]
    start = np.r_[0, np.flatnonzero(gs[1:] != gs[:-1]) + 1]
    run = np.arange(len(g)) - np.repeat(start, np.diff(np.r_[start, len(g)]))
    rank = np.empty(len(g), np.int32)
    rank[order] = run + 1
    gmax = np.empty(len(g), np.float32)
    gmax[order] = np.repeat(score[order][start], np.diff(np.r_[start, len(g)]))
    return rank, gmax


def generate_candidates(s1, r, s1_keys, r_keys, k=20, k2=10, df_cap=2000, rev_m=0, n_threads=2, verbose=True):
    """s1, r: pyarrow tables (country column); *_keys: (indptr, indices). Returns candidate DataFrame."""
    t0 = time.time()
    countries = pcmp.unique(s1["country"]).to_pylist()
    s1_country = country_codes(s1["country"], countries)
    r_country = country_codes(r["country"], countries)
    parts = []
    for ci, c in enumerate(countries):
        i1 = np.flatnonzero(s1_country == ci)
        ir = np.flatnonzero(r_country == ci)
        if len(ir) == 0:
            continue
        sk, rk_ = _sub_csr(*s1_keys, i1), _sub_csr(*r_keys, ir)
        res = {}
        for mode, kk, pre in (("sum", k, "b"), ("contain", k2, "c")):
            a, b, sc = block_country(sk, rk_, k=kk, df_cap=df_cap, n_threads=n_threads, mode=mode)
            rank, _ = _group_rank(a, sc)
            rrank, _ = _group_rank(b, sc)
            if rev_m:
                m = rrank <= rev_m
                a, b, sc, rank, rrank = a[m], b[m], sc[m], rank[m], rrank[m]
            res[pre] = ((a.astype(np.int64) << 32) | b.astype(np.int64), sc, rank, rrank)
            del a, b, sc, rank, rrank
        del sk, rk_
        allk = np.unique(np.concatenate([res["b"][0], res["c"][0]]))
        d = {"s1": i1[(allk >> 32).astype(np.int64)].astype(np.int32),
             "r": ir[(allk & 0xFFFFFFFF).astype(np.int64)].astype(np.int32)}
        for pre in "bc":
            kx, sc, rank, rrank = res[pre]
            pos = np.searchsorted(allk, kx)
            col_s = np.zeros(len(allk), np.float32); col_s[pos] = sc
            col_r = np.full(len(allk), 999, np.int16); col_r[pos] = rank
            col_v = np.full(len(allk), 999, np.int16); col_v[pos] = rrank
            d[pre + "score"], d[pre + "rank"], d[pre + "rank_rev"] = col_s, col_r, col_v
        del res, allk
        d = pd.DataFrame(d)
        parts.append(d)
        if verbose:
            print(f"  block {c}: S1={len(i1)} R={len(ir)} kept={len(d)} ({time.time() - t0:.0f}s)", flush=True)
    return pd.concat(parts, ignore_index=True)
