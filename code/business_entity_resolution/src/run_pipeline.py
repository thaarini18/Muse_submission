"""End-to-end pipeline: raw TSVs -> blocking -> two-round candidate generation -> matcher -> output TSVs.

Usage (from code/business_entity_resolution/):
    python src/run_pipeline.py --data ../../dataset --work work --out ../../output

Steps (each cached in --work, so an interrupted run resumes):
  prep     normalise names / addresses (prep.py)
  block    hashed blocking keys (keys.py) + inverted-index retrieval per country (block.py)
  train    round-1 features for 800k training entities; stage-1 filter (2-fold OOF + full);
           stage-1 scores for ALL training entities; round-2 expansion (expand.py);
           sibling + round-2 features (round2.py); new-pair filter; final matcher
           (early-stopped on 200k held-out entities -> validation_report.json)
  test     same scoring path on the test set; writes candidate_pairs.tsv and matching_results.tsv
"""
import argparse
import gc
import json
import os
import time
import multiprocessing as mp

import numpy as np
import pandas as pd
import pyarrow.compute as pcmp
import pyarrow.parquet as pq
import lightgbm as lgb

from common import load_split, load_gt_pairs, f05_macro
from prep import prep_file
from keys import build_keys
from block import generate_candidates
from features import skeleton_idf
from pairs import select_candidates, add_block_feats, build_matrix
from model import train_stage1, context_features, FILTER_THR
from pairs import ALL_FEATS
from expand import expand_candidates, SEED_THR
from features import sibling_features
from round2 import (new_pair_matrix, rule_counts, train_new_filter, assemble_final, train_final,
                    NEW_PRECUT_REV, NEW_FILTER_THR, NEW_FEATS)
import pyarrow as pa
from decide import decide

N_DEV = 800_000             # Source1 training entities whose pairs are used to fit the models
N_VALID = 200_000           # of which held out for early stopping / threshold / report
MATCH_THR = 0.65            # final probability threshold (tuned for macro F0.5 on the held-out split)
SEED = 0


def log(msg, t0=[time.time()]):
    print(f"[{time.time() - t0[0]:7.0f}s] {msg}", flush=True)


def step_prep(args, split):
    for s in (1, 2, 3):
        dst = os.path.join(args.work, f"{split}_s{s}.parquet")
        if not os.path.exists(dst):
            prep_file(os.path.join(args.data, split, f"{split}_source{s}.tsv"), dst)


def step_keys(args, split):
    names = ["sp", "si", "rp", "ri"]
    paths = [os.path.join(args.work, f"keys_{split}_{n}.npy") for n in names]
    if not all(os.path.exists(p) for p in paths):
        s1, r = load_split(args.work, split, columns=["nm_core", "ad_tok", "nm_ascii"])
        s1k = build_keys(s1)
        np.save(paths[0], s1k[0]); np.save(paths[1], s1k[1]); del s1k, s1
        rk = build_keys(r)
        np.save(paths[2], rk[0]); np.save(paths[3], rk[1]); del rk, r
        gc.collect()
    ld = lambda p: np.load(p, mmap_mode="r")
    return (ld(paths[0]), ld(paths[1])), (ld(paths[2]), ld(paths[3]))


def step_block(args, split):
    out = os.path.join(args.work, f"cand_{split}.parquet")
    if not os.path.exists(out):
        s1k, rk = step_keys(args, split)
        s1, r = load_split(args.work, split, columns=["country"])
        cand = generate_candidates(s1, r, s1k, rk, k=20, k2=10, df_cap=2000, rev_m=5)
        cand.to_parquet(out)
        del cand, s1k, rk
        gc.collect()
    return out


# ============================================================================ training
# Source1 training entities used to fit models: DEV = first N_DEV of a fixed permutation;
# VALID = first N_VALID of those (early stopping / threshold / report), TRN = the rest.
# Every other training entity is still scored by stage 1 so that the competition between entities
# (who else claims a Source2/3 record) is the same in training as on the full test set.

def _perm(n):
    return np.random.default_rng(SEED).permutation(n)


def _fold(s1):
    return (np.asarray(s1).astype(np.int64) * 2654435761 % 2**32) % 2


def _key(a, b):
    return (np.asarray(a).astype(np.int64) << 24) | np.asarray(b).astype(np.int64)


def _cand_sel(args, split):
    cf = os.path.join(args.work, f"cand_{split}_sel.parquet")
    if not os.path.exists(cf):
        _, idsr = load_split(args.work, split, columns=["entity_id"])
        r_is_s3 = pcmp.starts_with(idsr["entity_id"], "S3-").to_numpy(zero_copy_only=False)
        del idsr, _
        cand = select_candidates(pq.read_table(os.path.join(args.work, f"cand_{split}.parquet")).to_pandas())
        add_block_feats(cand, r_is_s3).to_parquet(cf)
    return cf


def _train_dev_features(args):
    """Round-1 features for the DEV entities (with labels)."""
    fx, fp = os.path.join(args.work, "dev_X.npy"), os.path.join(args.work, "dev_pairs.parquet")
    if os.path.exists(fx):
        return
    cf = _cand_sel(args, "train")
    s1, r = load_split(args.work, "train", columns=["entity_id", "nm_ascii", "nm_core", "ad_tok", "business_name"])
    gt = load_gt_pairs(os.path.join(args.data, "train", "train_ground_truth.tsv"), s1["entity_id"], r["entity_id"])
    s1 = s1.drop_columns(["entity_id"]); r = r.drop_columns(["entity_id"])
    nidf, aidf = skeleton_idf(r["nm_core"]), skeleton_idf(r["ad_tok"], addr=True)
    perm = _perm(s1.num_rows)
    cand = pq.read_table(cf).to_pandas()
    cand = cand[np.isin(cand.s1.values, perm[:N_DEV])]
    P, X = build_matrix(s1, r, cand, nidf, aidf)
    P["y"] = np.isin(_key(P.s1, P.r), _key(gt.s1, gt.r)).astype(np.int8)
    P["valid"] = np.isin(P.s1.values, perm[:N_VALID])
    P.to_parquet(fp); np.save(fx, X)
    gt.to_parquet(os.path.join(args.work, "gt_train.parquet"))


def _train_stage1(args):
    names = [os.path.join(args.work, f"stage1_{n}.txt") for n in ("f0", "f1", "full")]
    if all(os.path.exists(n) for n in names):
        return [lgb.Booster(model_file=n) for n in names]
    P = pd.read_parquet(os.path.join(args.work, "dev_pairs.parquet"))
    X = np.load(os.path.join(args.work, "dev_X.npy"), mmap_mode="r")
    trn = ~P.valid.values
    f = _fold(P.s1.values)
    models = []
    for m in ((trn & (f == 1)), (trn & (f == 0)), trn):   # f0 model predicts fold 0 (trained on fold 1)
        models.append(train_stage1(np.asarray(X[m]), P.y.values[m]))
        log(f"  stage1 model trained on {m.sum()} pairs")
    for b, n in zip(models, names):
        b.save_model(n)
    return models


def _stage1_dev_scores(P, X, models):
    """OOF for TRN entities, full model for VALID entities."""
    f = _fold(P.s1.values)
    p = np.empty(len(P), np.float32)
    for i, m in enumerate([~P.valid.values & (f == 0), ~P.valid.values & (f == 1), P.valid.values]):
        if m.any():
            p[m] = models[i].predict(np.asarray(X[m]), num_threads=2)
    return p


def _round1_scores(args, split, model, skip_s1=None, keep_x=False, n_chunks=12):
    """Stage-1 scores for every rule candidate (chunked over Source1); keeps p1 >= FILTER_THR.
    Writes r1_{split}_part{i}.parquet (s1, r, p1, nfe, leg) and, if keep_x, the filtered X."""
    cf = _cand_sel(args, split)
    s1n = pq.ParquetFile(os.path.join(args.work, f"{split}_s1.parquet")).metadata.num_rows
    bounds = np.linspace(0, s1n, n_chunks + 1).astype(np.int64)
    todo = [i for i in range(n_chunks) if not os.path.exists(os.path.join(args.work, f"r1_{split}_part{i:02d}.parquet"))]
    if not todo:
        return
    _, rtok = load_split(args.work, split, columns=["nm_core", "ad_tok"])
    nidf, aidf = skeleton_idf(rtok["nm_core"]), skeleton_idf(rtok["ad_tok"], addr=True)
    del rtok, _
    gc.collect()
    s1, r = load_split(args.work, split, columns=["nm_ascii", "nm_core", "ad_tok", "business_name"])
    nfe_i, leg_i = ALL_FEATS.index("num_first_eq"), ALL_FEATS.index("legal_eq")
    for i in todo:
        lo, hi = bounds[i], bounds[i + 1]
        c = pq.read_table(cf, filters=[("s1", ">=", int(lo)), ("s1", "<", int(hi))]).to_pandas()
        if skip_s1 is not None:
            c = c[~skip_s1[c.s1.values]]
        P, X = build_matrix(s1, r, c, nidf, aidf)
        del c
        p1 = model.predict(X, num_threads=2).astype(np.float32)
        m = p1 >= FILTER_THR
        out = P[m].reset_index(drop=True)
        out["p1"] = p1[m]; out["nfe"] = X[m, nfe_i]; out["leg"] = X[m, leg_i]
        if keep_x:
            np.save(os.path.join(args.work, f"r1_{split}_part{i:02d}_X.npy"), X[m])
        out.to_parquet(os.path.join(args.work, f"r1_{split}_part{i:02d}.parquet"))
        log(f"  {split} S1 [{lo},{hi}): {len(P)} -> {m.sum()}")
        del P, X
        gc.collect()


def _load_r1(args, split):
    parts = sorted(f for f in os.listdir(args.work) if f.startswith(f"r1_{split}_part") and f.endswith(".parquet"))
    return pd.concat([pd.read_parquet(os.path.join(args.work, f)) for f in parts], ignore_index=True)


def _expand(args, split, r1):
    """Round-2 retrieval for ALL Source1 entities of the split (seeds = round-1 p1 >= SEED_THR)."""
    out = os.path.join(args.work, f"new_{split}.parquet")
    if os.path.exists(out):
        return pd.read_parquet(out)
    ld = lambda n: np.load(os.path.join(args.work, f"keys_{split}_{n}.npy"), mmap_mode="r")
    s1k, rk = (ld("sp"), ld("si")), (ld("rp"), ld("ri"))
    s1c = pq.read_table(os.path.join(args.work, f"{split}_s1.parquet"), columns=["country"])["country"].combine_chunks()
    rt = pa.concat_tables([pq.read_table(os.path.join(args.work, f"{split}_s{i}.parquet"), columns=["country", "ad_tok"])
                           for i in (2, 3)]).combine_chunks()
    r_empty = pcmp.equal(rt["ad_tok"], "").to_numpy(zero_copy_only=False)
    rc = rt["country"]
    del rt
    rule = pq.read_table(_cand_sel(args, split), columns=["s1", "r"]).to_pandas()
    excl = np.sort(_key(rule.s1, rule.r))
    del rule
    seeds = r1[r1.p1 >= SEED_THR][["s1", "r"]]
    new = expand_candidates(s1c, rc, s1k, rk, r_empty, seeds, np.arange(len(s1c)), excl, log=log)
    new = new[new.p_rank_rev <= NEW_PRECUT_REV].reset_index(drop=True)
    new.to_parquet(out)
    return new


def _pair_tables(args, split):
    s1, r = load_split(args.work, split, columns=["entity_id", "nm_ascii", "nm_core", "ad_tok", "business_name"])
    r_is_s3 = pcmp.starts_with(r["entity_id"], "S3-").to_numpy(zero_copy_only=False).astype(np.float32)
    s1 = s1.drop_columns(["entity_id"]); r = r.drop_columns(["entity_id"])
    return s1, r, r_is_s3


def _round2_features(args, split, r1_rows, new, seeds, tag):
    """sibling features for round-1 rows + full features for round-2 rows; cached by tag."""
    fs, fn, fp = [os.path.join(args.work, f"{tag}_{x}") for x in ("sib1.npy", "Xn.npy", "Pn.parquet")]
    if os.path.exists(fn):
        return np.load(fs), pd.read_parquet(fp), np.load(fn)
    s1, r, r_is_s3 = _pair_tables(args, split)
    nidf, aidf = skeleton_idf(r["nm_core"]), skeleton_idf(r["ad_tok"], addr=True)
    rule = pq.read_table(_cand_sel(args, split), columns=["s1", "r"]).to_pandas()
    s1_cnt, r_cnt = rule_counts(rule.s1.values, rule.r.values, s1.num_rows, r.num_rows)
    del rule
    sib1 = sibling_features(r, r1_rows[["s1", "r"]], seeds)
    np.save(fs, sib1)
    log(f"  {tag}: sibling features for {len(r1_rows)} round-1 pairs")
    Pn, Xn = new_pair_matrix(s1, r, new, seeds, nidf, aidf, s1_cnt, r_cnt, r_is_s3)
    Pn.to_parquet(fp); np.save(fn, Xn)
    log(f"  {tag}: features for {len(Pn)} round-2 pairs")
    return sib1, Pn, Xn


def _t_round1(args):
    """Round-1 stage-1 scores for all training entities (dev: OOF / valid; others: full model)."""
    if os.path.exists(os.path.join(args.work, "dev_r1_X.npy")):
        return
    _train_dev_features(args)
    m0, m1, mfull = _train_stage1(args)
    n1 = pq.ParquetFile(os.path.join(args.work, "train_s1.parquet")).metadata.num_rows
    is_dev = np.zeros(n1, bool); is_dev[_perm(n1)[:N_DEV]] = True
    _round1_scores(args, "train", mfull, skip_s1=is_dev)
    P = pd.read_parquet(os.path.join(args.work, "dev_pairs.parquet"))
    X = np.load(os.path.join(args.work, "dev_X.npy"), mmap_mode="r")
    p1 = _stage1_dev_scores(P, X, [m0, m1, mfull])
    keep = p1 >= FILTER_THR
    dev1 = P[keep].reset_index(drop=True)
    dev1["p1"] = p1[keep]
    dev1["nfe"] = X[keep, ALL_FEATS.index("num_first_eq")]; dev1["leg"] = X[keep, ALL_FEATS.index("legal_eq")]
    np.save(os.path.join(args.work, "dev_r1_X.npy"), np.asarray(X[keep]))
    dev1.to_parquet(os.path.join(args.work, "dev_r1.parquet"))


def _train_r1_all(args):
    dev1 = pd.read_parquet(os.path.join(args.work, "dev_r1.parquet"))
    other1 = _load_r1(args, "train")
    return dev1, pd.concat([dev1.drop(columns=["y", "valid"]), other1], ignore_index=True)


def _t_expand(args):
    _, r1 = _train_r1_all(args)
    _expand(args, "train", r1)


def _t_round2(args):
    dev1, r1 = _train_r1_all(args)
    new = pd.read_parquet(os.path.join(args.work, "new_train.parquet"))
    n1 = pq.ParquetFile(os.path.join(args.work, "train_s1.parquet")).metadata.num_rows
    is_dev = np.zeros(n1, bool); is_dev[_perm(n1)[:N_DEV]] = True
    seeds = r1[r1.p1 >= SEED_THR][["s1", "r"]]
    _round2_features(args, "train", dev1, new[is_dev[new.s1.values]].reset_index(drop=True), seeds, "dev_r2")


def _t_final(args):
    dev1, r1 = _train_r1_all(args)
    sib1, Pn, Xn = _round2_features(args, "train", None, None, None, "dev_r2")
    n1 = pq.ParquetFile(os.path.join(args.work, "train_s1.parquet")).metadata.num_rows
    perm = _perm(n1)
    gt = pd.read_parquet(os.path.join(args.work, "gt_train.parquet"))
    yn = np.isin(_key(Pn.s1, Pn.r), _key(gt.s1, gt.r)).astype(np.int8)
    valid_ids = perm[:N_VALID]
    vn = np.isin(Pn.s1.values, valid_ids)
    # new-pair filter: OOF on TRN, full model for VALID
    fn = _fold(Pn.s1.values)
    pn = np.empty(len(Pn), np.float32)
    models = []
    for m_tr, m_pr in (((~vn) & (fn == 1), (~vn) & (fn == 0)), ((~vn) & (fn == 0), (~vn) & (fn == 1)), (~vn, vn)):
        b = train_new_filter(Xn[m_tr], yn[m_tr])
        pn[m_pr] = b.predict(Xn[m_pr], num_threads=2)
        models.append(b)
    models[2].save_model(os.path.join(args.work, "newfilter.txt"))
    k2 = pn >= NEW_FILTER_THR
    log(f"round-2 dev pairs {len(Pn)} -> {k2.sum()} after new-pair filter")
    # context over the union: round-1 survivors of EVERY training entity + dev round-2 survivors
    U = pd.concat([r1[["s1", "r"]], Pn[k2][["s1", "r"]]], ignore_index=True)
    up = np.concatenate([r1.p1.values, pn[k2]]).astype(np.float32)
    unfe = np.concatenate([r1.nfe.values, Xn[k2, NEW_FEATS.index("num_first_eq")]]).astype(np.float32)
    uleg = np.concatenate([r1.leg.values, Xn[k2, NEW_FEATS.index("legal_eq")]]).astype(np.float32)
    C = context_features(U, None, up, nfe=unfe, leg=uleg)
    C = np.vstack([C[:len(dev1)], C[len(r1):]])            # dev1 rows are the first rows of r1
    n_r1 = len(r1)
    del U, up, unfe, uleg, r1
    gc.collect()
    X1 = np.load(os.path.join(args.work, "dev_r1_X.npy"))
    Xf = assemble_final(X1, sib1, Xn[k2], C)
    del X1, Xn, C
    yf = np.concatenate([dev1.y.values, yn[k2]])
    vf = np.concatenate([dev1.valid.values, vn[k2]])
    Pf = pd.concat([dev1[["s1", "r"]], Pn[k2][["s1", "r"]]], ignore_index=True)
    log(f"final matcher: {(~vf).sum()} train rows, {vf.sum()} valid rows (context over {n_r1} round-1 pairs)")
    Xv, yv = Xf[vf], yf[vf]
    Xt, yt = Xf[~vf], yf[~vf]
    del Xf
    gc.collect()
    b2 = train_final(Xt, yt, 5000, valid=(Xv, yv))
    b2.save_model(os.path.join(args.work, "final.txt"))
    pv = b2.predict(Xv, num_threads=2)
    Pv = Pf[vf].reset_index(drop=True)
    truth = gt[np.isin(gt.s1.values, valid_ids)]
    rep = {"valid_entities": int(N_VALID), "final_cands_per_s1": float(vf.sum() / N_VALID),
           "candidate_recall": float(yv.sum() / len(truth)), "best_iteration": b2.best_iteration}
    for thr in (0.5, 0.6, 0.65, 0.7, 0.75, 0.8):
        rep[f"F0.5@{thr}"] = f05_macro(decide(Pv, pv, thr), truth, valid_ids)
    log(json.dumps(rep, indent=1))
    with open(os.path.join(args.work, "validation_report.json"), "w") as f:
        json.dump(rep, f, indent=1)


def step_train(args):
    if os.path.exists(os.path.join(args.work, "final.txt")):
        return
    for fn in (_t_round1, _t_expand, _t_round2, _t_final):
        log(f"  {fn.__name__}")
        _in_subprocess(fn, args)


# ============================================================================ test
def _test_round1(args):
    _round1_scores(args, "test", lgb.Booster(model_file=os.path.join(args.work, "stage1_full.txt")), keep_x=True)


def _test_round2(args, chunk=1_000_000):
    """Sibling features for round-1 survivors; round-2 pairs: features + new-pair filter, chunk-wise
    (only survivors are kept on disk)."""
    r1 = _load_r1(args, "test")
    new = _expand(args, "test", r1)
    seeds = r1[r1.p1 >= SEED_THR][["s1", "r"]]
    s1, r, r_is_s3 = _pair_tables(args, "test")
    fs = os.path.join(args.work, "test_r2_sib1.npy")
    if not os.path.exists(fs):
        np.save(fs, sibling_features(r, r1[["s1", "r"]], seeds))
        log(f"  test: sibling features for {len(r1)} round-1 pairs")
    del r1
    gc.collect()
    nidf, aidf = skeleton_idf(r["nm_core"]), skeleton_idf(r["ad_tok"], addr=True)
    rule = pq.read_table(_cand_sel(args, "test"), columns=["s1", "r"]).to_pandas()
    s1_cnt, r_cnt = rule_counts(rule.s1.values, rule.r.values, s1.num_rows, r.num_rows)
    del rule
    nfm = lgb.Booster(model_file=os.path.join(args.work, "newfilter.txt"))
    for i, st in enumerate(range(0, len(new), chunk)):
        fp = os.path.join(args.work, f"test_r2_part{i:02d}.parquet")
        if os.path.exists(fp):
            continue
        Pn, Xn = new_pair_matrix(s1, r, new.iloc[st:st + chunk].reset_index(drop=True), seeds, nidf, aidf,
                                 s1_cnt, r_cnt, r_is_s3)
        pn = nfm.predict(Xn, num_threads=2).astype(np.float32)
        k = pn >= NEW_FILTER_THR
        np.save(fp.replace(".parquet", "_X.npy"), Xn[k])
        Pn[k].reset_index(drop=True).assign(pn=pn[k]).to_parquet(fp)
        log(f"  test round-2 pairs {st}-{st + len(Pn)}: {k.sum()} kept")
        del Pn, Xn
        gc.collect()


def _test_final(args):
    r1 = _load_r1(args, "test")
    parts = sorted(f for f in os.listdir(args.work) if f.startswith("r1_test_part") and f.endswith("_X.npy"))
    sib1 = np.load(os.path.join(args.work, "test_r2_sib1.npy"))
    p2parts = sorted(f for f in os.listdir(args.work) if f.startswith("test_r2_part") and f.endswith(".parquet"))
    Pn = pd.concat([pd.read_parquet(os.path.join(args.work, f)) for f in p2parts], ignore_index=True)
    Xn = np.concatenate([np.load(os.path.join(args.work, f.replace(".parquet", "_X.npy"))) for f in p2parts])
    pn = Pn.pop("pn").values
    U = pd.concat([r1[["s1", "r"]], Pn[["s1", "r"]]], ignore_index=True)
    up = np.concatenate([r1.p1.values, pn]).astype(np.float32)
    unfe = np.concatenate([r1.nfe.values, Xn[:, NEW_FEATS.index("num_first_eq")]]).astype(np.float32)
    uleg = np.concatenate([r1.leg.values, Xn[:, NEW_FEATS.index("legal_eq")]]).astype(np.float32)
    C = context_features(U, None, up, nfe=unfe, leg=uleg)
    del up, unfe, uleg
    b2 = lgb.Booster(model_file=os.path.join(args.work, "final.txt"))
    p = np.empty(len(U), np.float32)
    st = 0
    for f in parts:                                           # round-1 rows, chunk by chunk
        X1 = np.load(os.path.join(args.work, f))
        n = len(X1)
        Xf = assemble_final(X1, sib1[st:st + n], Xn[:0], C[st:st + n])
        p[st:st + n] = b2.predict(Xf, num_threads=2)
        st += n
        del X1, Xf
        gc.collect()
    Xf = assemble_final(np.zeros((0, len(ALL_FEATS)), np.float32), sib1[:0], Xn, C[st:])
    p[st:] = b2.predict(Xf, num_threads=2)
    np.save(os.path.join(args.work, "test_final_p.npy"), p)
    U.to_parquet(os.path.join(args.work, "test_final_pairs.parquet"))
    ids1, idsr = load_split(args.work, "test", columns=["entity_id"])
    s1_ids = ids1["entity_id"].to_numpy(zero_copy_only=False)
    r_ids = idsr["entity_id"].to_numpy(zero_copy_only=False)
    matches = decide(U, p, MATCH_THR)
    write_outputs(args.out, s1_ids, r_ids, U, matches)
    log(f"test: round-1 {len(r1)} + round-2 {len(Pn)} = {len(U)} candidates ({len(U) / len(s1_ids):.2f}/S1); "
        f"matches {len(matches)} ({len(matches) / len(s1_ids):.2f}/S1)")


def _in_subprocess(fn, *a):
    """Run a memory-heavy step in a fresh (spawned) process so its memory is fully returned afterwards."""
    p = mp.get_context("spawn").Process(target=fn, args=a)
    p.start()
    p.join()
    if p.exitcode != 0:
        raise RuntimeError(f"{fn.__name__} failed with exit code {p.exitcode}")


def step_test(args):
    for fn in (_test_round1, _test_round2, _test_final):
        log(f"  {fn.__name__}")
        _in_subprocess(fn, args)


def _write_lists(path, header, s1_ids, r_ids, pairs):
    pairs = pairs.sort_values(["s1", "r"])
    lists = pairs.groupby("s1")["r"].apply(lambda x: ",".join(r_ids[x.values]))
    col = pd.Series("", index=np.arange(len(s1_ids)))
    col.loc[lists.index] = lists.values
    df = pd.DataFrame({"source1_entity_id": s1_ids, header: col.values})
    df.to_csv(path, sep="\t", index=False)


def write_outputs(out_dir, s1_ids, r_ids, cand_pairs, match_pairs):
    os.makedirs(out_dir, exist_ok=True)
    _write_lists(os.path.join(out_dir, "candidate_pairs.tsv"), "candidate_entity_ids", s1_ids, r_ids,
                 cand_pairs[["s1", "r"]].drop_duplicates())
    _write_lists(os.path.join(out_dir, "matching_results.tsv"), "matched_entity_ids", s1_ids, r_ids,
                 match_pairs[["s1", "r"]].drop_duplicates())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset dir containing train/ and test/")
    ap.add_argument("--work", default="work")
    ap.add_argument("--out", default="output")
    args = ap.parse_args()
    os.makedirs(args.work, exist_ok=True)
    for split in ("train", "test"):
        log(f"prep {split}"); step_prep(args, split)
    for split in ("train", "test"):
        log(f"blocking {split}"); step_block(args, split)
    log("training"); step_train(args)
    log("test inference"); step_test(args)
    log("done")


if __name__ == "__main__":
    main()
