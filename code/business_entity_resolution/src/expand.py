"""Second-round candidate generation ("record linking through siblings").

Round 1 finds, for most entities, a few confident Source2/3 matches ("seeds", stage-1 p >= SEED_THR).
Other fragments of the same business often resemble a seed more than they resemble the Source1 record
(same native-script name, same truncated address, same name-only record).  So each Source1 query is
expanded with the blocking keys of its seeds (pseudo-relevance feedback) and retrieval is run again.

Two extra retrieval passes, per country:
  * expanded query  -> sum-idf top-K_PRF and containment top-K_PRF_C over all Source2/3 records
  * original query  -> containment top-K_EMPTY over Source2/3 records with an EMPTY address only
    (name-only records are otherwise crowded out by address-bearing look-alikes)
Only pairs not already in the round-1 candidate rule set are returned.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp

from block import block_country, _sub_csr, _group_rank, country_codes

SEED_THR = 0.9
K_PRF, K_PRF_C, K_EMPTY = 10, 5, 3


def _expanded_queries(q_rows, s1_keys, seeds, r_keys):
    """Key lists for queries q_rows (global S1 idx): own keys + keys of their seed records."""
    sp_, si_ = s1_keys
    rp_, ri_ = r_keys
    own_ptr, own_idx = _sub_csr(sp_, si_, q_rows)
    rows = [np.repeat(np.arange(len(q_rows)), np.diff(own_ptr))]
    cols = [np.asarray(own_idx)]
    if len(seeds):
        pos = np.searchsorted(q_rows, seeds.s1.values)
        sr = seeds.r.values.astype(np.int64)
        order = np.argsort(sr, kind="stable")
        sr_sorted, pos_sorted = sr[order], pos[order]
        u, inv = np.unique(sr_sorted, return_inverse=True)
        uptr, uidx = _sub_csr(rp_, ri_, u)
        lens = np.diff(uptr)[inv]
        starts = uptr[:-1][inv]
        rows.append(np.repeat(pos_sorted, lens))
        cols.append(np.asarray(uidx)[np.repeat(starts, lens) + np.arange(lens.sum()) -
                                     np.repeat(np.cumsum(lens) - lens, lens)])
    rows, cols = np.concatenate(rows).astype(np.int32), np.concatenate(cols).astype(np.int32)
    m = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(q_rows), 1 << 26))
    m.sum_duplicates()
    return m.indptr.astype(np.int64), m.indices.astype(np.int32)


def expand_candidates(s1_country_col, r_country_col, s1_keys, r_keys, r_empty, seeds, query_s1,
                      exclude_keys, df_cap=2000, verbose=True, log=print, batch=250000):
    """seeds: DataFrame(s1, r) confident round-1 matches; query_s1: sorted global S1 idx to expand;
    r_empty: bool array (Source2/3 record has empty address); exclude_keys: sorted int64 (s1<<24|r)
    of round-1 rule candidates. Returns DataFrame s1, r, pscore, prank, pcscore, pcrank, escore, erank,
    p_rank_rev (rev rank among expanded queries)."""
    countries = sorted(set(pd.unique(pd.Series(s1_country_col.to_pylist()))))
    s1c = country_codes(s1_country_col, countries)
    rc = country_codes(r_country_col, countries)
    out = []
    for ci, c in enumerate(countries):
        qc = query_s1[s1c[query_s1] == ci]
        ir = np.flatnonzero(rc == ci)
        if len(qc) == 0 or len(ir) == 0:
            continue
        rk_c = _sub_csr(*r_keys, ir)
        ie = ir[r_empty[ir]]
        rk_e = _sub_csr(*r_keys, ie) if len(ie) else None
        for bs in range(0, len(qc), batch):             # query batches bound peak memory
            q = qc[bs:bs + batch]
            sd = seeds[np.isin(seeds.s1.values, q)]
            qk = _expanded_queries(q, s1_keys, sd, r_keys)
            res = {}
            for mode, k, pre in (("sum", K_PRF, "p"), ("contain", K_PRF_C, "pc")):
                a, b, v = block_country(qk, rk_c, k=k, df_cap=df_cap, mode=mode)
                res[pre] = (q[a], ir[b], v)
            del qk
            # empty-address pass: original (unexpanded) keys vs empty-address records only
            if rk_e is not None:
                a, b, v = block_country(_sub_csr(*s1_keys, q), rk_e, k=K_EMPTY, df_cap=df_cap, mode="contain")
                res["e"] = (q[a], ie[b], v)
            parts = []
            for pre, (a, b, v) in res.items():
                rank, _ = _group_rank(a, v)
                parts.append(pd.DataFrame({"s1": a.astype(np.int32), "r": b.astype(np.int32),
                                           pre + "score": v.astype(np.float32),
                                           pre + "rank": rank.astype(np.float32)}))
            d = parts[0]
            for p in parts[1:]:
                d = d.merge(p, on=["s1", "r"], how="outer")
            key = (d.s1.values.astype(np.int64) << 24) | d.r.values.astype(np.int64)
            d = d[~np.isin(key, exclude_keys)]
            out.append(d)
            if verbose:
                log(f"  expand {c}: queries {bs}-{bs + len(q)} of {len(qc)}, seeds={len(sd)}, new pairs={len(d)}")
        del rk_c, rk_e
    d = pd.concat(out, ignore_index=True)
    for pre in ("p", "pc", "e"):
        if pre + "score" not in d:
            d[pre + "score"], d[pre + "rank"] = 0.0, 99.0
        d[pre + "score"] = d[pre + "score"].fillna(0).astype(np.float32)
        d[pre + "rank"] = d[pre + "rank"].fillna(99).astype(np.float32)
    best = np.maximum(d.pscore.values, d.pcscore.values * 50)
    d["p_rank_rev"] = _group_rank(d.r.values.astype(np.int64), best.astype(np.float32))[0].astype(np.float32)
    return d.reset_index(drop=True)
