"""Stage 3c (v2, used for the submission): the address-pass LightGBM is retrained on held-out entities
(4-fold CV evaluation: +0.00085 macro F0.5 at p >= 0.7, vs +0.00061 for the model trained on other entities).
Original description follows.

Stage 3c: score address-pass pairs (addr_pass.py) with a small LightGBM and evaluate on VALID entities.

    python src/addr_score.py --work work --sim sim --pkg ce_pkg --addr addr
Writes addr/addr_model.txt, addr/test_addr_scored.parquet (s1, r, p) and prints the held-out F0.5 change.
"""
import argparse, os
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq, lightgbm as lgb
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

ap = argparse.ArgumentParser()
for k in ("work", "sim", "pkg", "addr"):
    ap.add_argument(f"--{k}", required=True)
ap.add_argument("--thr", type=float, default=0.85)
ap.add_argument("--test_only", action="store_true")
a = ap.parse_args()
key = lambda s, r: np.asarray(s, np.int64) * (1 << 25) + np.asarray(r, np.int64)
COLS = ["nm_ascii", "nm_core", "nm_skel", "ad_tok", "business_name"]


class Rows:
    """Arrow table; .iloc[idx] converts only the requested rows to pandas."""
    def __init__(self, t):
        self.t = t
        self.iloc = self

    def __getitem__(self, idx):
        return self.t.take(pa.array(np.asarray(idx, np.int64))).to_pandas()


def load(split):
    rd = lambda src: pq.read_table(os.path.join(a.work, f"{split}_{src}.parquet"), columns=COLS)
    return Rows(rd("s1").combine_chunks()), Rows(pa.concat_tables([rd("s2"), rd("s3")]).combine_chunks())


def nums(s):
    return {t for t in s.split() if any(ch.isdigit() for ch in t)}


def featurize(P, S, R, main):
    """P: s1, r, cos. main: s1, r, p (the main candidate set with its final probability)."""
    P = P.copy()
    g = P.groupby("s1").cos
    P["rank"] = g.rank(ascending=False, method="first"); P["gap"] = g.transform("max") - P.cos; P["n_q"] = g.transform("size")
    gr = P.groupby("r").cos
    P["rrank"] = gr.rank(ascending=False, method="first"); P["rgap"] = gr.transform("max") - P.cos; P["n_r"] = gr.transform("size")
    A = S.iloc[P.s1.values].reset_index(drop=True); B = R.iloc[P.r.values].reset_index(drop=True)
    f = lambda sc, x, y: cpdist(A[x].fillna("").tolist(), B[y].fillna("").tolist(), scorer=sc, workers=2)
    P["n_tset"] = f(fuzz.token_set_ratio, "nm_ascii", "nm_ascii")
    P["n_ratio"] = f(fuzz.ratio, "nm_core", "nm_core")
    P["n_sk"] = f(fuzz.token_set_ratio, "nm_skel", "nm_skel")
    P["n_part"] = f(fuzz.partial_ratio, "nm_core", "nm_core")
    P["a_tset"] = f(fuzz.token_set_ratio, "ad_tok", "ad_tok")
    P["r_nonascii"] = B.business_name.fillna("").map(lambda s: any(ord(c) > 127 for c in s)).values.astype(np.int8)
    P["r_nm_len"] = B.nm_core.fillna("").str.len().values
    na = [nums(x) for x in A.ad_tok.fillna("")]; nb = [nums(x) for x in B.ad_tok.fillna("")]
    P["num_j"] = [len(x & y) / max(1, len(x | y)) for x, y in zip(na, nb)]
    P["num_extra_r"] = [len(y - x) for x, y in zip(na, nb)]
    # main-set context
    m = main[main.p >= main.groupby("r").p.transform("max")]
    acc = m[m.p >= a.thr]
    P["s1_n_acc"] = acc.groupby("s1").size().reindex(P.s1.values).fillna(0).values
    P["s1_best_p"] = main.groupby("s1").p.max().reindex(P.s1.values).fillna(0).values
    P["r_main_p"] = main.groupby("r").p.max().reindex(P.r.values).fillna(0).values
    # similarity to the entity's best accepted record (sibling)
    best = acc.sort_values("p", ascending=False).drop_duplicates("s1").set_index("s1").r
    sib = best.reindex(P.s1.values).fillna(-1).values.astype(np.int64)
    ok = sib >= 0
    SB = R.iloc[np.where(ok, sib, 0)].reset_index(drop=True)
    for col, sc, nm in (("ad_tok", fuzz.token_set_ratio, "sib_a"), ("nm_ascii", fuzz.token_set_ratio, "sib_n"),
                        ("nm_skel", fuzz.token_set_ratio, "sib_sk")):
        v = cpdist(SB[col].fillna("").tolist(), B[col].fillna("").tolist(), scorer=sc, workers=2)
        P[nm] = np.where(ok, v, -1)
    return P


FEATS = ["cos", "rank", "gap", "n_q", "rrank", "rgap", "n_r", "n_tset", "n_ratio", "n_sk", "n_part", "a_tset",
         "r_nonascii", "r_nm_len", "num_j", "num_extra_r", "s1_n_acc", "s1_best_p", "r_main_p", "sib_a", "sib_n",
         "sib_sk"]

import time; t0=time.time(); log=lambda m: print(f'[{time.time()-t0:4.0f}s] {m}',flush=True)
S, R = load("train")
val = pd.read_parquet(os.path.join(a.pkg, "val_pairs.parquet"), columns=["s1","r","p_lgb","y"]).rename(columns={"p_lgb":"p"})
vids = pd.read_parquet(os.path.join(a.pkg, "val_meta.parquet")).val_s1.values
P = pd.read_parquet(os.path.join(a.addr, "dev_addr_pairs.parquet")); P = P[np.isin(P.s1.values, vids)].reset_index(drop=True)
gt = pd.read_parquet(os.path.join(a.work, "gt_train.parquet")); P["y"] = np.isin(key(P.s1, P.r), key(gt.s1, gt.r)).astype(np.int8)
X = featurize(P, S, R, val[["s1","r","p"]]); log(f'val addr pairs {len(X)} pos {X.y.sum()}')
old_path = os.path.join(a.addr, "addr_model.txt")
old = lgb.Booster(model_file=old_path).predict(X[FEATS]) if os.path.exists(old_path) else None
fold = (pd.util.hash_array(X.s1.values.astype(np.int64)) % 4).astype(int); new = np.zeros(len(X))
prm = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=40, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=2)
for k in range(4):
    new[fold == k] = lgb.train(prm, lgb.Dataset(X.loc[fold != k, FEATS], X.y[fold != k]), 400).predict(X.loc[fold == k, FEATS])
truth = pd.read_parquet(os.path.join(a.pkg, "val_truth.parquet"))
def f05(pred):
    u = pd.Index(vids); tp = pred.merge(truth, on=["s1","r"]).groupby("s1").size().reindex(u).fillna(0).values
    npd = pred.groupby("s1").size().reindex(u).fillna(0).values; nt = truth.groupby("s1").size().reindex(u).fillna(0).values
    f = np.where((npd == 0) & (nt == 0), 1.0, 0.0); m = tp > 0; P_, R_ = tp[m]/npd[m], tp[m]/nt[m]; f[m] = 1.25*P_*R_/(.25*P_+R_); return float(f.mean())
d = val[val.p >= val.groupby("r").p.transform("max")]; base = d[d.p >= 0.85][["s1","r"]]; used = set(base.r)
fb = f05(base); log(f'base {fb:.5f}')
for name, pp in [(n_, q_) for n_, q_ in (("old", old), ("new", new)) if q_ is not None]:
    V = X[["s1","r","y"]].copy(); V["p"] = pp; V = V.sort_values("p", ascending=False).drop_duplicates("r"); V = V[~V.r.isin(used)]
    for t in (0.5, 0.6, 0.7, 0.8, 0.9):
        add = V[V.p >= t]; log(f'  {name} t={t}: +{len(add)} prec {add.y.mean():.3f} dF {f05(pd.concat([base, add[["s1","r"]]]))-fb:+.5f}')
mdl = lgb.train(prm, lgb.Dataset(X[FEATS], X.y), 400); mdl.save_model(os.path.join(a.addr, "addr_model_v2.txt"))
del S, R, X
S, R = load("test")
T = pd.read_parquet(os.path.join(a.pkg, "test_pairs.parquet"), columns=["s1","r","p_lgb"]).rename(columns={"p_lgb":"p"})
Pt = pd.read_parquet(os.path.join(a.addr, "test_addr_pairs.parquet"))
g = Pt.groupby("r").cos; Pt["_rrank"] = g.rank(ascending=False, method="first"); Pt["_rgap"] = g.transform("max") - Pt.cos; Pt["_n_r"] = g.transform("size")
parts = []
for st in range(0, len(Pt), 600_000):
    ch = Pt.iloc[st:st+600_000]; Xc = featurize(ch[["s1","r","cos"]], S, R, T)
    Xc["rrank"], Xc["rgap"], Xc["n_r"] = ch._rrank.values, ch._rgap.values, ch._n_r.values
    Xc["p"] = mdl.predict(Xc[FEATS]); parts.append(Xc[["s1","r","p"]]); log(f'test chunk {st}')
Xt = pd.concat(parts, ignore_index=True); Xt[Xt.p >= 0.3].reset_index(drop=True).to_parquet(os.path.join(a.addr, "test_addr_scored_v2.parquet"))
log(f'saved; p>=0.5 {int((Xt.p>=0.5).sum())} p>=0.8 {int((Xt.p>=0.8).sum())}')
