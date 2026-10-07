"""Candidate-set selection and feature-matrix assembly shared by training and inference."""
import numpy as np
import pandas as pd

from features import compute_features, FEATURE_NAMES

# final candidate rule (this is what goes into candidate_pairs.tsv and into the model)
K_SUM, K_CONTAIN, REV_M = 20, 10, 2

BLOCK_FEATS = ["bscore", "brank", "brank_rev", "cscore", "crank", "crank_rev", "b_rel", "c_rel", "b_rel_rev",
               "c_rel_rev", "s1_ncand", "r_ncand", "is_s3"]
ALL_FEATS = FEATURE_NAMES + BLOCK_FEATS


def select_candidates(cand: pd.DataFrame, k=K_SUM, kc=K_CONTAIN, rev=REV_M):
    m = ((cand.brank.values <= k) & (cand.brank_rev.values <= rev)) | \
        ((cand.crank.values <= kc) & (cand.crank_rev.values <= rev))
    return pd.DataFrame({col: cand[col].values[m] for col in cand.columns}, copy=False)


def _gmax(idx, val, size):
    m = np.full(size, -np.inf, np.float32)
    np.maximum.at(m, idx, val)
    return m[idx]


def add_block_feats(c: pd.DataFrame, r_is_s3: np.ndarray):
    """Blocking-context features; numpy group ops (memory-light). Must be run on the FULL candidate set."""
    s1 = c.s1.values.astype(np.int64)
    r = c.r.values.astype(np.int64)
    n1, nr = int(s1.max()) + 1, int(r.max()) + 1
    out = {"s1": c.s1.values.astype(np.int32), "r": c.r.values.astype(np.int32)}
    for col in ("bscore", "brank", "brank_rev", "cscore", "crank", "crank_rev"):
        out[col] = c[col].values.astype(np.float32)
    for pre in "bc":
        s = out[pre + "score"]
        mx = _gmax(s1, s, n1)
        out[pre + "_rel"] = np.where(mx > 0, s / np.where(mx > 0, mx, 1), 0).astype(np.float32)
        mr = _gmax(r, s, nr)
        out[pre + "_rel_rev"] = np.where(mr > 0, s / np.where(mr > 0, mr, 1), 0).astype(np.float32)
    out["s1_ncand"] = np.bincount(s1, minlength=n1)[s1].astype(np.float32)
    out["r_ncand"] = np.bincount(r, minlength=nr)[r].astype(np.float32)
    out["is_s3"] = r_is_s3[r].astype(np.float32)
    del s1, r
    return pd.DataFrame({k: out[k] for k in ["s1", "r"] + BLOCK_FEATS}, copy=False)


def build_matrix(s1, r, c, name_idf, addr_idf, pool=None):
    """c: candidates with block features already added (add_block_feats on the FULL candidate set)."""
    X = np.empty((len(c), len(ALL_FEATS)), np.float32)
    X[:, len(FEATURE_NAMES):] = c[BLOCK_FEATS].to_numpy(np.float32)
    P = c[["s1", "r"]].reset_index(drop=True)
    del c
    compute_features(s1, r, P, name_idf, addr_idf, out=X, pool=pool)
    return P, X
