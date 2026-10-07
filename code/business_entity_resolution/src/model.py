"""Two-stage matcher.

Stage 1 ("filter"): a small LightGBM over all pair features scores the blocking output; pairs with
p1 < FILTER_THR are dropped. What survives is the final candidate set (candidate_pairs.tsv).
Stage 2 ("matcher"): a larger LightGBM over pair features + context features derived from stage-1
scores of the competing candidates (same Source1 entity / same Source2-3 record).
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

from pairs import ALL_FEATS

FILTER_THR = 0.01
STAGE1_PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=200,
                     feature_fraction=0.8, bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0,
                     num_threads=2, verbose=-1, seed=1)
STAGE1_ROUNDS = 250
STAGE2_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
                     feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                     num_threads=2, verbose=-1, seed=2)
CTX_FEATS = ["p1", "p1_rank_s1", "p1_rel_s1", "p1_max_other_s1", "p1_n_high_s1", "p1_sum_s1",
             "p1_rank_r", "p1_best_other_r", "p1_margin_r", "n_cand_s1_f", "n_cand_r_f",
             "num_agree_high", "legal_agree_high"]
STAGE2_FEATS = ALL_FEATS + CTX_FEATS


def train_stage1(X, y, rounds=STAGE1_ROUNDS):
    return lgb.train(STAGE1_PARAMS, lgb.Dataset(X, y, feature_name=ALL_FEATS), rounds)


def oof_stage1(P, X, y, n_folds=2, rounds=STAGE1_ROUNDS):
    """Out-of-fold stage-1 predictions (folds by Source1 entity)."""
    fold = (P.s1.values.astype(np.int64) * 2654435761 % 2**32) % n_folds
    p = np.zeros(len(P), np.float32)
    for f in range(n_folds):
        m = fold == f
        b = train_stage1(X[~m], y[~m], rounds)
        p[m] = b.predict(X[m], num_threads=2)
    return p


def _grp_rank(key, val):
    order = np.lexsort((-val, key))
    ks = key[order]
    start = np.r_[0, np.flatnonzero(ks[1:] != ks[:-1]) + 1]
    lens = np.diff(np.r_[start, len(key)])
    rank = np.empty(len(key), np.float32)
    rank[order] = np.arange(len(key)) - np.repeat(start, lens) + 1
    return rank


def context_features(P, X, p1, nfe=None, leg=None):
    """Context over the filtered candidate set. P: s1, r; X: base features; p1: stage-1 prob."""
    d = pd.DataFrame({"s1": P.s1.values, "r": P.r.values, "p1": p1})
    g1 = d.groupby("s1")["p1"]
    mx = g1.transform("max").values
    # second best within s1 (max of the others)
    rank_s1 = _grp_rank(d.s1.values, p1)
    d["_top"] = np.where(rank_s1 == 1, -1.0, p1)
    second = d.groupby("s1")["_top"].transform("max").values
    max_other = np.where(rank_s1 == 1, np.maximum(second, 0), mx)
    n_high = d.assign(h=(p1 >= 0.5).astype(np.float32)).groupby("s1")["h"].transform("sum").values
    sum_s1 = g1.transform("sum").values
    rank_r = _grp_rank(d.r.values, p1)
    d["_rt"] = np.where(rank_r == 1, -1.0, p1)
    r_second = d.groupby("r")["_rt"].transform("max").values
    r_max = d.groupby("r")["p1"].transform("max").values
    best_other_r = np.where(rank_r == 1, np.maximum(r_second, 0), r_max)
    n_s1 = d.groupby("s1")["r"].transform("size").values
    n_r = d.groupby("r")["s1"].transform("size").values
    # agreement with the confident candidates of the same S1 entity:
    #   share of confident (p1>=0.5, excluding self) siblings with the same first-number / legal-form
    #   equality w.r.t. S1 as this pair
    if nfe is None:
        nfe = X[:, ALL_FEATS.index("num_first_eq")]
        leg = X[:, ALL_FEATS.index("legal_eq")]
    hi = (p1 >= 0.5).astype(np.float32)
    d["_hn"] = hi * (nfe == 1)
    d["_hl"] = hi * (leg == 1)
    hn = d.groupby("s1")["_hn"].transform("sum").values - d["_hn"].values
    hl = d.groupby("s1")["_hl"].transform("sum").values - d["_hl"].values
    others_hi = n_high - hi
    num_agree = np.where(others_hi > 0, hn / np.maximum(others_hi, 1), -1)
    legal_agree = np.where(others_hi > 0, hl / np.maximum(others_hi, 1), -1)
    C = np.column_stack([p1, rank_s1, p1 / np.maximum(mx, 1e-6), max_other, n_high, sum_s1, rank_r,
                         best_other_r, p1 - best_other_r, n_s1, n_r, num_agree, legal_agree]).astype(np.float32)
    return C


def train_stage2(X2, y, rounds, valid=None):
    dt = lgb.Dataset(X2, y, feature_name=STAGE2_FEATS)
    cbs = [lgb.log_evaluation(200)]
    vs = []
    if valid is not None:
        vs = [lgb.Dataset(valid[0], valid[1], reference=dt)]
        cbs.append(lgb.early_stopping(100))
    return lgb.train(STAGE2_PARAMS, dt, rounds, valid_sets=vs, callbacks=cbs)
