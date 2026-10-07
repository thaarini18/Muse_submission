"""Stage 5d: group-aware LightGBM stacker (final decision scores).

    python src/ce_stack.py --pkg ce_pkg --scores scores

Inputs: the ce_prep package (val_pairs / test_pairs / val_truth / val_meta) and the three cross-encoder score
files per split written by ce_train.py (<tag>_val.npy, <tag>_test.npy for tags ce, xlmr, sib).

1. Logistic-regression blend of logit(p_lgb) and the three cross-encoder logits (+ products), fitted on the
   held-out VALID entities' uncertain pairs (0.02 < p_lgb < 0.98); other pairs keep p_lgb.
2. Group features on the blended probability: rank / count / best / sum / confident count / best competitor
   within the Source1 entity and within the record, plus source features (S3 flag, confident S2 / S3 counts per
   entity, best same-source probability, gap to the entity's top candidate).
3. A LightGBM stacker trained on the VALID uncertain pairs rescores the uncertain test pairs.
   Held-out (5-fold by entity): macro F0.5 0.98138 at threshold 0.75 with the empty-entity rescue.
Writes <scores>/stack_test_p.npy (probability for every row of test_pairs.parquet).
"""
import argparse, os
import numpy as np, pandas as pd, pyarrow.parquet as pq, lightgbm as lgb
from sklearn.linear_model import LogisticRegression

ap = argparse.ArgumentParser()
ap.add_argument("--pkg", required=True); ap.add_argument("--scores", required=True)
ap.add_argument("--work", default=None, help="run_pipeline work dir (only to read the Source2 sizes); optional")
a = ap.parse_args()
D, S = a.pkg, a.scores
T = ["ce", "xlmr", "sib"]
lg = lambda p: np.log(np.clip(p, 1e-5, 1 - 1e-5) / (1 - np.clip(p, 1e-5, 1 - 1e-5)))
F = lambda pl, c: np.column_stack([lg(pl)] + [lg(x) for x in c] + [lg(pl) * lg(x) for x in c])
PRM = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.9,
           verbose=-1, num_threads=2)
# number of Source2 records per split (Source2 rows come first in the record index)
NS2 = {"val": 5034616, "test": 4887273}
if a.work:
    NS2 = {"val": pq.ParquetFile(os.path.join(a.work, "train_s2.parquet")).metadata.num_rows,
           "test": pq.ParquetFile(os.path.join(a.work, "test_s2.parquet")).metadata.num_rows}


def load(split):
    P = pd.read_parquet(os.path.join(D, f"{split}_pairs.parquet"),
                        columns=["s1", "r", "p_lgb"] + (["y"] if split == "val" else []))
    m = ((P.p_lgb > .02) & (P.p_lgb < .98)).values
    return P, m, [np.load(os.path.join(S, f"{t}_{split}.npy")) for t in T]


def grp(key, p):
    o = np.lexsort((-p, key)); k = key[o]; st = np.r_[True, k[1:] != k[:-1]]
    gid = np.cumsum(st) - 1; first = np.flatnonzero(st); size = np.diff(np.r_[first, len(k)])
    rk = np.arange(len(k)) - first[gid] + 1
    mx = p[o][first]; sec = np.where(size > 1, p[o][np.minimum(first + 1, len(k) - 1)], 0.0)
    sm = np.add.reduceat(p[o], first); nc = np.add.reduceat((p[o] > 0.8).astype(float), first)
    out = np.empty((len(k), 6)); out[o, 0] = rk; out[o, 1] = size[gid]; out[o, 2] = mx[gid]; out[o, 3] = sm[gid]
    out[o, 4] = nc[gid]; out[o, 5] = np.where(rk == 1, sec[gid], mx[gid])
    return out


def feats(P, m, C, lr, ns2):
    p = P.p_lgb.values.astype(float).copy(); p[m] = lr.predict_proba(F(P.p_lgb.values[m], C))[:, 1]
    gs = grp(P.s1.values.astype(np.int64), p); gr = grp(P.r.values.astype(np.int64), p)
    X = pd.DataFrame({"lp3": lg(p), "pl": P.p_lgb.values, "rk": gs[:, 0], "n": gs[:, 1], "smax": gs[:, 2],
                      "ssum": gs[:, 3], "nconf": gs[:, 4], "other_max": gs[:, 5], "rrk": gr[:, 0], "rn": gr[:, 1],
                      "r_other": gr[:, 5]})
    for t, c in zip(T, C):
        v = np.full(len(P), np.nan); v[m] = c; X[t] = v
    s3 = (P.r.values >= ns2).astype(float); X["is_s3"] = s3
    conf = (p > 0.8).astype(float)
    df = pd.DataFrame({"k": P.s1.values.astype(np.int64), "c2": conf * (1 - s3), "c3": conf * s3,
                       "m": p * (1 - s3), "m3": p * s3})
    g = df.groupby("k")
    X["n2conf"] = g.c2.transform("sum").values; X["n3conf"] = g.c3.transform("sum").values
    X["best_same_src"] = np.where(s3 == 1, g.m3.transform("max").values, g.m.transform("max").values)
    X["gap_top"] = X.smax - p
    return X, p


FE = ["lp3", "pl", "ce", "xlmr", "sib", "rk", "n", "smax", "ssum", "nconf", "other_max", "rrk", "rn", "r_other",
      "is_s3", "n2conf", "n3conf", "best_same_src", "gap_top"]

val, mv, V = load("val")
lr = LogisticRegression(max_iter=2000).fit(F(val.p_lgb.values[mv], V), val.y.values[mv])
X, _ = feats(val, mv, V, lr, NS2["val"])
booster = lgb.train(PRM, lgb.Dataset(X.loc[mv, FE], val.y.values[mv]), 400)
test, mt, Tt = load("test")
XT, pt = feats(test, mt, Tt, lr, NS2["test"])
pt[mt] = booster.predict(XT.loc[mt, FE])
np.save(os.path.join(S, "stack_test_p.npy"), pt)
print("wrote", os.path.join(S, "stack_test_p.npy"))
