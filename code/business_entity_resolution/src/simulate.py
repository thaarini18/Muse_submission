"""Retrain on a TEST-LIKE decoy rate.

The test set has ~5.8 Source2/3 records per Source1 entity vs ~4.7 in training, i.e. many more records
whose business is not in Source1 (decoys). We reproduce that in training by hiding a random HIDE_FRAC
of training Source1 entities that are not used for fitting: their Source2/3 records stay in the corpus
but no longer have an owner, exactly like test decoys. Everything downstream of blocking (reverse
ranks, stage 1, context, round 2, final matcher, threshold) is recomputed in that world.

Run:  python src/simulate.py --data <dataset> --work work --sim sim --out <dir>   (after run_pipeline.py)
"""
import argparse
import gc
import json
import os
import time
import multiprocessing as mp

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.compute as pcmp
import lightgbm as lgb

from common import load_split, f05_macro
from features import skeleton_idf, compute_features, sibling_features, FEATURE_NAMES
from pairs import select_candidates, add_block_feats, ALL_FEATS, BLOCK_FEATS
from block import _group_rank
from model import train_stage1, context_features, FILTER_THR
from expand import SEED_THR
from round2 import (train_new_filter, assemble_final, train_final, NEW_FILTER_THR, NEW_FEATS,
                    NEW_PRECUT_REV, rule_counts)
from decide import decide
from run_pipeline import _perm, _fold, _key, N_DEV, N_VALID, _write_lists

HIDE_FRAC = 0.20
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def P(a, *x):
    return os.path.join(a.sim, *x)


def W(a, *x):
    return os.path.join(a.work, *x)


def hidden_mask(n1):
    perm = _perm(n1)
    pool = perm[N_DEV:]
    rng = np.random.default_rng(7)
    h = rng.choice(pool, int(HIDE_FRAC * n1), replace=False)
    m = np.zeros(n1, bool)
    m[h] = True
    return m


def _rerank_rev(rows_key_r, score, present):
    rank = np.full(len(score), 999, np.int16)
    idx = np.flatnonzero(present)
    if len(idx):
        rk, _ = _group_rank(rows_key_r[idx].astype(np.int64), score[idx].astype(np.float32))
        rank[idx] = rk
    return rank


# ---------------------------------------------------------------- S1: candidates in the simulated world
def s1_candidates(a):
    out = P(a, "cand_train_sel.parquet")
    if os.path.exists(out):
        return
    n1 = pq.ParquetFile(W(a, "train_s1.parquet")).metadata.num_rows
    hid = hidden_mask(n1)
    np.save(P(a, "hidden.npy"), hid)
    c = pq.read_table(W(a, "cand_train.parquet")).to_pandas()
    c = c[~hid[c.s1.values]].reset_index(drop=True)
    r = c.r.values
    c["brank_rev"] = _rerank_rev(r, c.bscore.values, c.brank.values < 999)
    c["crank_rev"] = _rerank_rev(r, c.cscore.values, c.crank.values < 999)
    _, idsr = load_split(a.work, "train", columns=["entity_id"])
    r_is_s3 = pcmp.starts_with(idsr["entity_id"], "S3-").to_numpy(zero_copy_only=False)
    del idsr, _
    sel = add_block_feats(select_candidates(c), r_is_s3)
    sel.to_parquet(out)
    log(f"sim candidates: {len(sel)} ({len(sel) / (~hid).sum():.2f}/S1), hidden {hid.sum()}")


# ---------------------------------------------------------------- S2: dev features in the simulated world
def s2_dev(a):
    fx = P(a, "dev_X.npy")
    if os.path.exists(fx):
        return
    n1 = pq.ParquetFile(W(a, "train_s1.parquet")).metadata.num_rows
    perm = _perm(n1)
    sel = pq.read_table(P(a, "cand_train_sel.parquet")).to_pandas()
    sel = sel[np.isin(sel.s1.values, perm[:N_DEV])].reset_index(drop=True)
    oldP = pd.read_parquet(W(a, "dev_pairs.parquet"))
    oldX = np.load(W(a, "dev_X.npy"), mmap_mode="r")
    ok = _key(oldP.s1, oldP.r)
    nk = _key(sel.s1, sel.r)
    pos = pd.Series(np.arange(len(ok)), index=ok).reindex(nk).values
    have = ~np.isnan(pos)
    nf = len(FEATURE_NAMES)
    miss = np.flatnonzero(~have)
    log(f"dev pairs {len(sel)}: {have.sum()} reused, {len(miss)} need features")
    fm = P(a, "dev_missing_X.npy")
    if len(miss) and not os.path.exists(fm):
        # separate child: string tables are large; free them before assembling the big matrix
        p = mp.get_context("spawn").Process(target=_missing_feats, args=(a, sel.iloc[miss][["s1", "r"]].reset_index(drop=True)))
        p.start(); p.join()
        if p.exitcode != 0:
            raise RuntimeError("missing features failed")
    X = np.lib.format.open_memmap(fx + ".tmp.npy", mode="w+", dtype=np.float32, shape=(len(sel), len(ALL_FEATS)))
    hv = np.flatnonzero(have)
    for st in range(0, len(hv), 1_000_000):
        idx = hv[st:st + 1_000_000]
        X[idx, :nf] = oldX[pos[idx].astype(np.int64), :nf]
    if len(miss):
        X[miss, :nf] = np.load(fm)
    X[:, nf:] = sel[BLOCK_FEATS].to_numpy(np.float32)
    X.flush()
    del X
    gt = pd.read_parquet(W(a, "gt_train.parquet"))
    Pd = sel[["s1", "r"]].copy()
    Pd["y"] = np.isin(nk, _key(gt.s1, gt.r)).astype(np.int8)
    Pd["valid"] = np.isin(Pd.s1.values, perm[:N_VALID])
    Pd.to_parquet(P(a, "dev_pairs.parquet"))
    os.replace(fx + ".tmp.npy", fx)


def _missing_feats(a, pairs):
    s1, r = load_split(a.work, "train", columns=["nm_ascii", "nm_core", "ad_tok", "business_name"])
    nidf, aidf = skeleton_idf(r["nm_core"]), skeleton_idf(r["ad_tok"], addr=True)
    np.save(P(a, "dev_missing_X.npy"), compute_features(s1, r, pairs, nidf, aidf))


# ---------------------------------------------------------------- S3: stage 1 + round-1 scores for everyone
def s3_round1(a):
    if os.path.exists(P(a, "r1_all.parquet")):
        return
    Pd = pd.read_parquet(P(a, "dev_pairs.parquet"))
    X = np.load(P(a, "dev_X.npy"), mmap_mode="r")
    trn = ~Pd.valid.values
    f = _fold(Pd.s1.values)
    models = []
    for m in ((trn & (f == 1)), (trn & (f == 0)), trn):
        models.append(train_stage1(np.asarray(X[m]), Pd.y.values[m]))
    for b, n in zip(models, ("f0", "f1", "full")):
        b.save_model(P(a, f"stage1_{n}.txt"))
    log("stage1 models trained")
    p1 = np.empty(len(Pd), np.float32)
    for i, m in enumerate([trn & (f == 0), trn & (f == 1), ~trn]):
        p1[m] = models[i].predict(np.asarray(X[m]), num_threads=2)
    keep = p1 >= FILTER_THR
    dev1 = Pd[keep].reset_index(drop=True)
    dev1["p1"] = p1[keep]
    dev1["nfe"] = X[keep, ALL_FEATS.index("num_first_eq")]
    dev1["leg"] = X[keep, ALL_FEATS.index("legal_eq")]
    np.save(P(a, "dev_r1_X.npy"), np.asarray(X[keep]))
    dev1.to_parquet(P(a, "dev_r1.parquet"))
    del X
    gc.collect()
    # everyone else (not dev, not hidden), chunked
    n1 = pq.ParquetFile(W(a, "train_s1.parquet")).metadata.num_rows
    perm = _perm(n1)
    skip = np.load(P(a, "hidden.npy"))
    skip[perm[:N_DEV]] = True
    sel_path = P(a, "cand_train_sel.parquet")
    _, rtok = load_split(a.work, "train", columns=["nm_core", "ad_tok"])
    nidf, aidf = skeleton_idf(rtok["nm_core"]), skeleton_idf(rtok["ad_tok"], addr=True)
    del rtok, _
    s1, r = load_split(a.work, "train", columns=["nm_ascii", "nm_core", "ad_tok", "business_name"])
    from pairs import build_matrix
    parts = []
    bounds = np.linspace(0, n1, 13).astype(np.int64)
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        c = pq.read_table(sel_path, filters=[("s1", ">=", int(lo)), ("s1", "<", int(hi))]).to_pandas()
        c = c[~skip[c.s1.values]]
        Pc, Xc = build_matrix(s1, r, c, nidf, aidf)
        pc = models[2].predict(Xc, num_threads=2).astype(np.float32)
        m = pc >= FILTER_THR
        o = Pc[m].reset_index(drop=True)
        o["p1"] = pc[m]
        o["nfe"] = Xc[m, ALL_FEATS.index("num_first_eq")]
        o["leg"] = Xc[m, ALL_FEATS.index("legal_eq")]
        parts.append(o)
        log(f"  round1 S1 [{lo},{hi}): {len(Pc)} -> {m.sum()}")
        del Pc, Xc, c
        gc.collect()
    other = pd.concat(parts, ignore_index=True)
    r1 = pd.concat([dev1.drop(columns=["y", "valid"]), other], ignore_index=True)
    r1.to_parquet(P(a, "r1_all.parquet"))


# ---------------------------------------------------------------- S4: round 2 in the simulated world
def s4_round2(a):
    if os.path.exists(P(a, "dev_r2_Xn.npy")):
        return
    hid = np.load(P(a, "hidden.npy"))
    n1 = len(hid)
    perm = _perm(n1)
    is_dev = np.zeros(n1, bool)
    is_dev[perm[:N_DEV]] = True
    r1 = pd.read_parquet(P(a, "r1_all.parquet"))
    dev1 = pd.read_parquet(P(a, "dev_r1.parquet"))
    seeds = r1[r1.p1 >= SEED_THR][["s1", "r"]]
    # round-2 candidates: reuse retrieval output, drop hidden owners, recompute reverse rank
    new = pd.read_parquet(W(a, "new_train.parquet"))
    new = new[~hid[new.s1.values]].reset_index(drop=True)
    best = np.maximum(new.pscore.values, new.pcscore.values * 50).astype(np.float32)
    new["p_rank_rev"] = _group_rank(new.r.values.astype(np.int64), best)[0].astype(np.float32)
    new = new[new.p_rank_rev <= NEW_PRECUT_REV]
    new.to_parquet(P(a, "new_train.parquet"))
    new_dev = new[is_dev[new.s1.values]].reset_index(drop=True)
    # reuse pair features of the old dev round-2 table where possible
    oP = pd.read_parquet(W(a, "dev_r2_Pn.parquet"))
    oX = np.load(W(a, "dev_r2_Xn.npy"), mmap_mode="r")
    pos = pd.Series(np.arange(len(oP)), index=_key(oP.s1, oP.r)).reindex(_key(new_dev.s1, new_dev.r)).values
    have = ~np.isnan(pos)
    new_dev = new_dev[have].reset_index(drop=True)
    Xn = np.asarray(oX[pos[have].astype(np.int64)]).copy()
    del oX
    # refresh columns that depend on the simulated world
    Xn[:, NEW_FEATS.index("p_rank_rev")] = new_dev.p_rank_rev.values
    sel = pq.read_table(P(a, "cand_train_sel.parquet"), columns=["s1", "r"]).to_pandas()
    nr = pq.ParquetFile(W(a, "train_s2.parquet")).metadata.num_rows + pq.ParquetFile(W(a, "train_s3.parquet")).metadata.num_rows
    s1_cnt, r_cnt = rule_counts(sel.s1.values, sel.r.values, n1, nr)
    del sel
    Xn[:, NEW_FEATS.index("rule_s1_ncand")] = s1_cnt[new_dev.s1.values]
    Xn[:, NEW_FEATS.index("rule_r_ncand")] = r_cnt[new_dev.r.values]
    _, r = load_split(a.work, "train", columns=["nm_core", "ad_tok"])
    sibn = sibling_features(r, new_dev[["s1", "r"]], seeds)
    Xn[:, NEW_FEATS.index("sib_n"):] = sibn
    sib1 = sibling_features(r, dev1[["s1", "r"]], seeds)
    np.save(P(a, "dev_r2_sib1.npy"), sib1)
    new_dev[["s1", "r"]].to_parquet(P(a, "dev_r2_Pn.parquet"))
    np.save(P(a, "dev_r2_Xn.npy"), Xn)
    log(f"round-2 dev pairs {len(new_dev)} (reused features; {(~have).sum()} dropped)")


# ---------------------------------------------------------------- S5: filters + final matcher + threshold
def s5_final(a):
    if os.path.exists(P(a, "final.txt")):
        return
    hid = np.load(P(a, "hidden.npy"))
    n1 = len(hid)
    perm = _perm(n1)
    valid_ids = perm[:N_VALID]
    dev1 = pd.read_parquet(P(a, "dev_r1.parquet"))
    r1 = pd.read_parquet(P(a, "r1_all.parquet"))
    sib1 = np.load(P(a, "dev_r2_sib1.npy"))
    Pn = pd.read_parquet(P(a, "dev_r2_Pn.parquet"))
    Xn = np.load(P(a, "dev_r2_Xn.npy"))
    gt = pd.read_parquet(W(a, "gt_train.parquet"))
    yn = np.isin(_key(Pn.s1, Pn.r), _key(gt.s1, gt.r)).astype(np.int8)
    vn = np.isin(Pn.s1.values, valid_ids)
    fn = _fold(Pn.s1.values)
    pn = np.empty(len(Pn), np.float32)
    ms = []
    for m_tr, m_pr in (((~vn) & (fn == 1), (~vn) & (fn == 0)), ((~vn) & (fn == 0), (~vn) & (fn == 1)), (~vn, vn)):
        b = train_new_filter(Xn[m_tr], yn[m_tr])
        pn[m_pr] = b.predict(Xn[m_pr], num_threads=2)
        ms.append(b)
    ms[2].save_model(P(a, "newfilter.txt"))
    k2 = pn >= NEW_FILTER_THR
    U = pd.concat([r1[["s1", "r"]], Pn[k2][["s1", "r"]]], ignore_index=True)
    up = np.concatenate([r1.p1.values, pn[k2]]).astype(np.float32)
    unfe = np.concatenate([r1.nfe.values, Xn[k2, NEW_FEATS.index("num_first_eq")]]).astype(np.float32)
    uleg = np.concatenate([r1.leg.values, Xn[k2, NEW_FEATS.index("legal_eq")]]).astype(np.float32)
    C = context_features(U, None, up, nfe=unfe, leg=uleg)
    C = np.vstack([C[:len(dev1)], C[len(r1):]])
    del U, up, unfe, uleg, r1
    gc.collect()
    X1 = np.load(P(a, "dev_r1_X.npy"))
    Xf = assemble_final(X1, sib1, Xn[k2], C)
    del X1, Xn, C
    yf = np.concatenate([dev1.y.values, yn[k2]])
    vf = np.concatenate([dev1.valid.values, vn[k2]])
    Pf = pd.concat([dev1[["s1", "r"]], Pn[k2][["s1", "r"]]], ignore_index=True)
    Xv, yv, Xt, yt = Xf[vf], yf[vf], Xf[~vf], yf[~vf]
    del Xf
    gc.collect()
    log(f"final matcher: {len(yt)} train rows, {len(yv)} valid rows")
    b2 = train_final(Xt, yt, 5000, valid=(Xv, yv))
    b2.save_model(P(a, "final.txt"))
    pv = b2.predict(Xv, num_threads=2)
    Pv = Pf[vf].reset_index(drop=True)
    truth = gt[np.isin(gt.s1.values, valid_ids)]
    rep = {"hidden_frac": HIDE_FRAC, "valid_cands_per_s1": float(len(yv) / N_VALID),
           "candidate_recall": float(yv.sum() / len(truth)), "best_iteration": b2.best_iteration}
    for thr in (0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95):
        rep[f"F0.5@{thr}"] = f05_macro(decide(Pv, pv, thr), truth, valid_ids)
    # the OLD models evaluated in the simulated world would need a separate run; report only new
    log(json.dumps(rep, indent=1))
    with open(P(a, "validation_report.json"), "w") as f:
        json.dump(rep, f, indent=1)


# ---------------------------------------------------------------- S6: rescore the test set
def s6_test(a):
    b1 = lgb.Booster(model_file=P(a, "stage1_full.txt"))
    nfm = lgb.Booster(model_file=P(a, "newfilter.txt"))
    b2 = lgb.Booster(model_file=P(a, "final.txt"))
    rparts = sorted(f for f in os.listdir(a.work) if f.startswith("r1_test_part") and f.endswith(".parquet"))
    r1s, keepm = [], []
    for f in rparts:
        X = np.load(W(a, f.replace(".parquet", "_X.npy")))
        p1 = b1.predict(X, num_threads=2).astype(np.float32)
        d = pd.read_parquet(W(a, f))
        d["p1"] = p1
        m = p1 >= FILTER_THR
        r1s.append(d[m])
        keepm.append(m)
        del X
    r1 = pd.concat(r1s, ignore_index=True)
    seeds = r1[r1.p1 >= SEED_THR][["s1", "r"]]
    _, r = load_split(a.work, "test", columns=["nm_core", "ad_tok"])
    sib1 = sibling_features(r, r1[["s1", "r"]], seeds)
    log(f"test round-1 rows {len(r1)}; sibling features done")
    p2parts = sorted(f for f in os.listdir(a.work) if f.startswith("test_r2_part") and f.endswith(".parquet"))
    Pn = pd.concat([pd.read_parquet(W(a, f)) for f in p2parts], ignore_index=True).drop(columns=["pn"])
    Xn = np.concatenate([np.load(W(a, f.replace(".parquet", "_X.npy"))) for f in p2parts])
    Xn[:, NEW_FEATS.index("sib_n"):] = sibling_features(r, Pn[["s1", "r"]], seeds)
    del r
    pn = nfm.predict(Xn, num_threads=2).astype(np.float32)
    k2 = pn >= NEW_FILTER_THR
    Pn, Xn, pn = Pn[k2].reset_index(drop=True), Xn[k2], pn[k2]
    U = pd.concat([r1[["s1", "r"]], Pn[["s1", "r"]]], ignore_index=True)
    up = np.concatenate([r1.p1.values, pn]).astype(np.float32)
    unfe = np.concatenate([r1.nfe.values, Xn[:, NEW_FEATS.index("num_first_eq")]]).astype(np.float32)
    uleg = np.concatenate([r1.leg.values, Xn[:, NEW_FEATS.index("legal_eq")]]).astype(np.float32)
    C = context_features(U, None, up, nfe=unfe, leg=uleg)
    p = np.empty(len(U), np.float32)
    st = 0
    for f, m in zip(rparts, keepm):
        X1 = np.load(W(a, f.replace(".parquet", "_X.npy")))[m]
        n = len(X1)
        p[st:st + n] = b2.predict(assemble_final(X1, sib1[st:st + n], Xn[:0], C[st:st + n]), num_threads=2)
        st += n
        del X1
        gc.collect()
    p[st:] = b2.predict(assemble_final(np.zeros((0, len(ALL_FEATS)), np.float32), sib1[:0], Xn, C[st:]),
                        num_threads=2)
    U.to_parquet(P(a, "test_pairs.parquet"))
    np.save(P(a, "test_p.npy"), p)
    ids1, idsr = load_split(a.work, "test", columns=["entity_id"])
    s1_ids = ids1["entity_id"].to_numpy(zero_copy_only=False)
    r_ids = idsr["entity_id"].to_numpy(zero_copy_only=False)
    rep = json.load(open(P(a, "validation_report.json")))
    thr = max((float(k.split("@")[1]) for k in rep if k.startswith("F0.5@")), key=lambda t: rep[f"F0.5@{t}"])
    os.makedirs(a.out, exist_ok=True)
    _write_lists(os.path.join(a.out, "candidate_pairs.tsv"), "candidate_entity_ids", s1_ids, r_ids, U)
    for t in sorted({thr, 0.8, 0.9}):
        m = decide(U, p, t)
        _write_lists(os.path.join(a.out, f"matching_results_thr{int(round(t * 100)):02d}.tsv"),
                     "matched_entity_ids", s1_ids, r_ids, m)
        log(f"test thr {t}: {len(m) / len(s1_ids):.3f} matches/S1")
    log(f"test candidates {len(U) / len(s1_ids):.2f}/S1; best validation thr {thr}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--work", default="work")
    ap.add_argument("--sim", default="sim")
    ap.add_argument("--out", default="output_sim")
    a = ap.parse_args()
    os.makedirs(a.sim, exist_ok=True)
    for fn in (s1_candidates, s2_dev, s3_round1, s4_round2, s5_final, s6_test):
        log(fn.__name__)
        p = mp.get_context("spawn").Process(target=fn, args=(a,))
        p.start()
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"{fn.__name__} failed ({p.exitcode})")
    log("done")


if __name__ == "__main__":
    main()
