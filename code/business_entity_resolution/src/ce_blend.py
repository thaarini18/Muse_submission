"""Stage 5c: blend LightGBM with one or more cross-encoders and write the submission files.

    python src/ce_blend.py --pkg ce_pkg --scores final --tags ce xlmr sib --thr 0.85 --out output

For the uncertain pairs (0.02 < p_lgb < 0.98) a logistic regression on
    logit(p_lgb), logit(p_ce_k) for every model k, logit(p_lgb) * logit(p_ce_k)
is fitted on the held-out VALID entities; other pairs keep p_lgb. Decision: each Source2/3 record goes to
its highest-probability Source1 entity, kept if p >= --thr. The held-out optimum is ~0.75; the submitted
0.85 is stricter because the test set has a higher share of unmatched look-alike records than training.
Optional --addr adds the scored address-pass pairs (addr_pass.py / addr_score.py).
"""
import argparse, json, os
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression

ap = argparse.ArgumentParser()
ap.add_argument("--pkg", required=True); ap.add_argument("--scores", required=True)
ap.add_argument("--tags", nargs="+", default=["ce"]); ap.add_argument("--thr", type=float, default=0.85)
ap.add_argument("--addr", default=None, help="addr_score.py output (s1, r, p) for extra address-pass pairs (test)")
ap.add_argument("--addr_thr", type=float, default=0.8)
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
D = a.pkg
lg = lambda p: np.log(np.clip(p, 1e-5, 1 - 1e-5) / (1 - np.clip(p, 1e-5, 1 - 1e-5)))


def feats(pl, ces):
    cols = [lg(pl)] + [lg(c) for c in ces] + [lg(pl) * lg(c) for c in ces]
    return np.column_stack(cols)


val = pd.read_parquet(os.path.join(D, "val_pairs.parquet"), columns=["s1", "r", "y", "p_lgb"])
test = pd.read_parquet(os.path.join(D, "test_pairs.parquet"), columns=["s1", "r", "p_lgb"])
mv = ((val.p_lgb > 0.02) & (val.p_lgb < 0.98)).values
mt = ((test.p_lgb > 0.02) & (test.p_lgb < 0.98)).values
vce = [np.load(os.path.join(a.scores, f"{t}_val.npy")) for t in a.tags]
tce = [np.load(os.path.join(a.scores, f"{t}_test.npy")) for t in a.tags]
lr = LogisticRegression(C=1.0, max_iter=2000).fit(feats(val.p_lgb.values[mv], vce), val.y.values[mv])


def blended(df, m, ces):
    p = df.p_lgb.values.astype(np.float64).copy()
    p[m] = lr.predict_proba(feats(df.p_lgb.values[m], ces))[:, 1]
    return p


def decide(df, p, thr):
    d = pd.DataFrame({"s1": df.s1.values, "r": df.r.values, "p": p})
    d = d[d.p >= d.groupby("r").p.transform("max")]
    return d[d.p >= thr][["s1", "r"]]


truth = pd.read_parquet(os.path.join(D, "val_truth.parquet"))
vids = pd.read_parquet(os.path.join(D, "val_meta.parquet")).val_s1.values


def f05(pred):
    u = pd.Index(vids)
    tp = pred.merge(truth, on=["s1", "r"]).groupby("s1").size().reindex(u).fillna(0).values
    npd = pred.groupby("s1").size().reindex(u).fillna(0).values
    nt = truth.groupby("s1").size().reindex(u).fillna(0).values
    f = np.where((npd == 0) & (nt == 0), 1.0, 0.0)
    m = tp > 0
    P, R = tp[m] / npd[m], tp[m] / nt[m]
    f[m] = 1.25 * P * R / (0.25 * P + R)
    return float(f.mean())


pv = blended(val, mv, vce)
table = [(t, f05(decide(val, pv, t))) for t in (0.6, 0.7, 0.75, 0.8, 0.85, 0.9)]
print(pd.DataFrame(table, columns=["thr", "F05_blend"]).to_string(index=False))

pt = blended(test, mt, tce)
P = test[["s1", "r"]].copy(); P["p"] = pt
match = decide(P, P.p.values, a.thr)
if a.addr:   # address pass: kept pairs join the candidate set; confident ones fill records nobody claimed
    X = pd.read_parquet(a.addr, columns=["s1", "r", "p"]).sort_values("p", ascending=False).drop_duplicates("r")
    X = X[~np.isin(X.r.values * (1 << 25) + X.s1.values, P.r.values.astype(np.int64) * (1 << 25) + P.s1.values)]
    add = X[(X.p >= a.addr_thr) & ~np.isin(X.r.values, match.r.values)]
    match = pd.concat([match, add[["s1", "r"]]], ignore_index=True)
    P = pd.concat([P, X], ignore_index=True)
ids = pd.read_parquet(os.path.join(D, "test_ids.parquet")).entity_id.values
rids = pd.read_parquet(os.path.join(D, "test_r_ids.parquet")).entity_id.values


def write(pairs, path, col):
    pairs = pairs.sort_values(["s1", "r"])
    s1v = pairs.s1.values
    rs = rids[pairs.r.values]
    starts = np.flatnonzero(np.r_[True, s1v[1:] != s1v[:-1]]) if len(s1v) else np.zeros(0, int)
    joined = [",".join(x) for x in np.split(rs, starts[1:])] if len(rs) else []
    c = np.full(len(ids), "", dtype=object)
    c[s1v[starts]] = joined
    pd.DataFrame({"source1_entity_id": ids, col: c}).to_csv(path, sep="\t", index=False)
    print("wrote", path)


write(match, os.path.join(a.out, "matching_results.tsv"), "matched_entity_ids")
write(P[["s1", "r"]], os.path.join(a.out, "candidate_pairs.tsv"), "candidate_entity_ids")
json.dump({"tags": a.tags, "thr": a.thr, "coef": lr.coef_.tolist(), "val_table": table},
          open(os.path.join(a.out, "blend_report.json"), "w"), indent=1)
