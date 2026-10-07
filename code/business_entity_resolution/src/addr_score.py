"""Stage 3c: score address-pass pairs (addr_pass.py) with a small LightGBM and evaluate on VALID entities.

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

if not a.test_only:
    # ---------------- dev: main candidate set with final probabilities
    import sys
    S, R = load("train")
    val = pd.read_parquet(os.path.join(a.pkg, "val_pairs.parquet"), columns=["s1", "r", "p_lgb", "y"]).rename(columns={"p_lgb": "p"})
    vids = pd.read_parquet(os.path.join(a.pkg, "val_meta.parquet")).val_s1.values
    dev1 = pd.read_parquet(os.path.join(a.sim, "dev_r1.parquet"), columns=["s1", "r", "p1"]).rename(columns={"p1": "p"})
    Pn = pd.read_parquet(os.path.join(a.sim, "dev_r2_Pn.parquet"), columns=["s1", "r"])
    main_trn = dev1[~np.isin(dev1.s1.values, vids)]
    excl = np.concatenate([key(main_trn.s1, main_trn.r), key(Pn.s1, Pn.r)])
    P = pd.read_parquet(os.path.join(a.addr, "dev_addr_pairs.parquet"))
    inval = np.isin(P.s1.values, vids)
    P = P[inval | ~np.isin(key(P.s1, P.r), excl)].reset_index(drop=True)
    gt = pd.read_parquet(os.path.join(a.work, "gt_train.parquet"))
    P["y"] = np.isin(key(P.s1, P.r), key(gt.s1, gt.r)).astype(np.int8)
    main = pd.concat([val[["s1", "r", "p"]], main_trn[["s1", "r", "p"]]], ignore_index=True)
    X = featurize(P, S, R, main)
    inval = np.isin(X.s1.values, vids)
    print("dev new pairs", len(X), "pos rate trn", X.y[~inval].mean(), "val", X.y[inval].mean(), flush=True)
    dtr = lgb.Dataset(X.loc[~inval, FEATS], X.y[~inval])
    mdl = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=50, verbose=-1,
                         feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, num_threads=2), dtr, 600)
    mdl.save_model(os.path.join(a.addr, "addr_model.txt"))
    pv = mdl.predict(X.loc[inval, FEATS])
    V = X[inval][["s1", "r", "y"]].copy(); V["p"] = pv
    truth = pd.read_parquet(os.path.join(a.pkg, "val_truth.parquet"))


    def f05(pred):
        u = pd.Index(vids)
        tp = pred.merge(truth, on=["s1", "r"]).groupby("s1").size().reindex(u).fillna(0).values
        npd = pred.groupby("s1").size().reindex(u).fillna(0).values
        nt = truth.groupby("s1").size().reindex(u).fillna(0).values
        f = np.where((npd == 0) & (nt == 0), 1.0, 0.0)
        m = tp > 0
        P_, R_ = tp[m] / npd[m], tp[m] / nt[m]
        f[m] = 1.25 * P_ * R_ / (0.25 * P_ + R_)
        return float(f.mean())


    def decide(d, thr):
        d = d[d.p >= d.groupby("r").p.transform("max")]
        return d[d.p >= thr][["s1", "r"]]


    base = decide(val[["s1", "r", "p"]], a.thr)
    print("VAL base F0.5 (lgb only, thr %.2f): %.5f" % (a.thr, f05(base)))
    for t2 in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        comb = pd.concat([val[["s1", "r", "p"]], V[["s1", "r", "p"]].assign(p=np.where(V.p >= t2, V.p.clip(a.thr, None), 0))])
        add = V[V.p >= t2]
        print(f"  add addr pairs p>={t2}: +{len(add)} pairs (prec {add.y.mean():.3f})  F0.5 {f05(decide(comb, a.thr)):.5f}", flush=True)
    imp = pd.Series(mdl.feature_importance("gain"), FEATS).sort_values(ascending=False)
    print(imp.head(10))


else:
    mdl = lgb.Booster(model_file=os.path.join(a.addr, 'addr_model.txt'))

# ---------------- test: score the address-pass pairs; keep p >= FILTER as extra candidates
FILTER = 0.5
S, R = load("test")
T = pd.read_parquet(os.path.join(a.pkg, "test_pairs.parquet"), columns=["s1", "r", "p_lgb"]).rename(columns={"p_lgb": "p"})
Pt = pd.read_parquet(os.path.join(a.addr, "test_addr_pairs.parquet"))
parts = []
g = Pt.groupby("r").cos
Pt["_rrank"] = g.rank(ascending=False, method="first"); Pt["_rgap"] = g.transform("max") - Pt.cos; Pt["_n_r"] = g.transform("size")
for st in range(0, len(Pt), 600_000):
    ch = Pt.iloc[st:st + 600_000]
    ch = ch[ch.s1.isin(ch.s1.values)]  # chunks are s1-sorted (retrieval order), s1 groups stay whole
    Xc = featurize(ch[["s1", "r", "cos"]], S, R, T)
    Xc["rrank"], Xc["rgap"], Xc["n_r"] = ch._rrank.values, ch._rgap.values, ch._n_r.values
    Xc["p"] = mdl.predict(Xc[FEATS])
    parts.append(Xc[["s1", "r", "p"]]); print("chunk", st, flush=True)
Xt = pd.concat(parts, ignore_index=True)
out = Xt[Xt.p >= FILTER][["s1", "r", "p"]].reset_index(drop=True)
out.to_parquet(os.path.join(a.addr, "test_addr_scored.parquet"))
ids = pd.read_parquet(os.path.join(a.pkg, "test_ids.parquet")).entity_id.values
rids = pd.read_parquet(os.path.join(a.pkg, "test_r_ids.parquet")).entity_id.values
pd.DataFrame({"source1_entity_id": ids[out.s1.values], "entity_id": rids[out.r.values], "p": out.p.round(4)}).to_csv(
    os.path.join(a.addr, "addr_extra_pairs.tsv"), sep="\t", index=False)
print("test addr pairs", len(Pt), "kept (p>=%.2f)" % FILTER, len(out), "of which p>=0.8:", int((out.p >= 0.8).sum()))
