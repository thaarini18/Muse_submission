"""Round-2 feature assembly shared by training and inference.

new pairs      = expansion candidates (expand.py), pre-cut by reverse rank
new-filter     = LightGBM over NEW_FEATS, keeps p >= NEW_FILTER_THR (last blocking stage for new pairs)
final matcher  = LightGBM over FINAL_FEATS on round-1 survivors + round-2 survivors
"""
import numpy as np
import lightgbm as lgb

from features import compute_features, sibling_features, FEATURE_NAMES, SIB_NAMES
from pairs import ALL_FEATS, BLOCK_FEATS
from model import CTX_FEATS

NEW_PRECUT_REV = 3
NEW_FILTER_THR = 0.05
EXP_FEATS = ["pscore", "prank", "pcscore", "pcrank", "escore", "erank", "p_rank_rev"]
RULE_FEATS = ["rule_s1_ncand", "rule_r_ncand", "is_s3_n"]
NEW_FEATS = FEATURE_NAMES + EXP_FEATS + RULE_FEATS + SIB_NAMES
FINAL_FEATS = ALL_FEATS + EXP_FEATS + SIB_NAMES + ["is_new"] + CTX_FEATS
NEW_PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=63, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0,
                  num_threads=2, verbose=-1, seed=3)
NEW_ROUNDS = 400
FINAL_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
                    feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                    num_threads=2, verbose=-1, seed=4)
_EXP_DEFAULT = np.array([0, 99, 0, 99, 0, 99, 99], np.float32)
_BLOCK_DEFAULT = {"bscore": 0, "brank": 999, "brank_rev": 999, "cscore": 0, "crank": 999, "crank_rev": 999,
                  "b_rel": 0, "c_rel": 0, "b_rel_rev": 0, "c_rel_rev": 0}


def rule_counts(rule_s1, rule_r, n1, nr):
    return np.bincount(rule_s1, minlength=n1), np.bincount(rule_r, minlength=nr)


def new_pair_matrix(s1, r, new, seeds, name_idf, addr_idf, s1_cnt, r_cnt, r_is_s3):
    """Features for round-2 pairs. new: DataFrame from expand_candidates (already pre-cut)."""
    n = len(new)
    X = np.empty((n, len(NEW_FEATS)), np.float32)
    P = new[["s1", "r"]].reset_index(drop=True)
    compute_features(s1, r, P, name_idf, addr_idf, out=X)
    o = len(FEATURE_NAMES)
    X[:, o:o + len(EXP_FEATS)] = new[EXP_FEATS].to_numpy(np.float32)
    o += len(EXP_FEATS)
    X[:, o] = s1_cnt[P.s1.values]; X[:, o + 1] = r_cnt[P.r.values]; X[:, o + 2] = r_is_s3[P.r.values]
    o += 3
    X[:, o:] = sibling_features(r, P, seeds)
    return P, X


def train_new_filter(X, y, rounds=NEW_ROUNDS):
    return lgb.train(NEW_PARAMS, lgb.Dataset(X, y, feature_name=NEW_FEATS), rounds)


def oof_new_filter(P, X, y, n_folds=2):
    fold = (P.s1.values.astype(np.int64) * 2654435761 % 2**32) % n_folds
    p = np.zeros(len(P), np.float32)
    for f in range(n_folds):
        m = fold == f
        p[m] = train_new_filter(X[~m], y[~m]).predict(X[m], num_threads=2)
    return p


def assemble_final(X1, sib1, X2n, C):
    """Final-matcher rows: round-1 survivors (X1: ALL_FEATS, sib1) stacked over round-2 survivors
    (X2n: NEW_FEATS); C: context features for the same rows in the same order."""
    n1, n2 = len(X1), len(X2n)
    nb, ne, ns = len(ALL_FEATS), len(EXP_FEATS), len(SIB_NAMES)
    X = np.empty((n1 + n2, len(FINAL_FEATS)), np.float32)
    X[:n1, :nb] = X1
    X[:n1, nb:nb + ne] = _EXP_DEFAULT
    X[:n1, nb + ne:nb + ne + ns] = sib1
    X[:n1, nb + ne + ns] = 0
    nf = len(FEATURE_NAMES)
    X[n1:, :nf] = X2n[:, :nf]
    for j, col in enumerate(BLOCK_FEATS):
        k = nf + j
        if col in _BLOCK_DEFAULT:
            X[n1:, k] = _BLOCK_DEFAULT[col]
        elif col == "s1_ncand":
            X[n1:, k] = X2n[:, NEW_FEATS.index("rule_s1_ncand")]
        elif col == "r_ncand":
            X[n1:, k] = X2n[:, NEW_FEATS.index("rule_r_ncand")]
        elif col == "is_s3":
            X[n1:, k] = X2n[:, NEW_FEATS.index("is_s3_n")]
    X[n1:, nb:nb + ne] = X2n[:, nf:nf + ne]
    X[n1:, nb + ne:nb + ne + ns] = X2n[:, NEW_FEATS.index(SIB_NAMES[0]):]
    X[n1:, nb + ne + ns] = 1
    X[:, nb + ne + ns + 1:] = C
    return X


def train_final(X, y, rounds, valid=None):
    dt = lgb.Dataset(X, y, feature_name=FINAL_FEATS)
    cbs = [lgb.log_evaluation(200)]
    vs = []
    if valid is not None:
        vs = [lgb.Dataset(valid[0], valid[1], reference=dt)]
        cbs.append(lgb.early_stopping(100))
    return lgb.train(FINAL_PARAMS, dt, rounds, valid_sets=vs, callbacks=cbs)
