"""Stage 3b: address-only retrieval pass.

Recovers true matches that name-driven blocking misses (native-script / heavily noised names at the same
address). Address tokens (normalised, stop tokens removed) are IDF-weighted and L2-normalised; each Source1
entity retrieves its top-K records of the same country by cosine. Pairs already in the main candidate set
are dropped; the rest are scored by a small LightGBM model (addr_model.txt) trained on training entities.

    python src/addr_pass.py dev   --work work --sim sim --pkg ce_pkg --out addr   # retrieval on train split
    python src/addr_pass.py test  --work work --sim sim --pkg ce_pkg --out addr   # retrieval on test split
"""
import argparse, os, time
import numpy as np, pandas as pd, scipy.sparse as sp, pyarrow.parquet as pq
from sklearn.feature_extraction.text import HashingVectorizer
from sparse_dot_topn import sp_matmul_topn
from normalize import ADDR_STOP

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["dev", "test"])
ap.add_argument("--work", required=True); ap.add_argument("--sim", required=True)
ap.add_argument("--pkg", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--k", type=int, default=5); ap.add_argument("--min_cos", type=float, default=0.5)
ap.add_argument("--n_trn", type=int, default=300_000)
ap.add_argument("--df_cap", type=int, default=3000, help="tokens in more records than this are not used to retrieve")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
t0 = time.time()
log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
NF = 1 << 22
hv = HashingVectorizer(n_features=NF, token_pattern=r"\S+", lowercase=False, binary=True, norm=None,
                       alternate_sign=False, stop_words=sorted(t for t in ADDR_STOP if t), dtype=np.float32)


def read(split, src, cols):
    return pq.read_table(os.path.join(a.work, f"{split}_{src}.parquet"), columns=cols).to_pandas()


split = "train" if a.mode == "dev" else "test"
S = read(split, "s1", ["ad_tok", "country"])
R = pd.concat([read(split, "s2", ["ad_tok", "country"]), read(split, "s3", ["ad_tok", "country"])],
              ignore_index=True)
if a.mode == "dev":
    vids = pd.read_parquet(os.path.join(a.pkg, "val_meta.parquet")).val_s1.values
    hidden = np.load(os.path.join(a.sim, "hidden.npy"))
    pool = np.flatnonzero(~hidden & ~np.isin(np.arange(len(S)), vids))
    trn = np.random.default_rng(5).choice(pool, a.n_trn, replace=False)
    qidx = np.sort(np.concatenate([vids, trn]))
else:
    qidx = np.arange(len(S))
log(f"queries {len(qidx)}  records {len(R)}")

res = []
for c in pd.unique(S.country.values[qidx]):
    qi = qidx[S.country.values[qidx] == c]
    ri = np.flatnonzero((R.country.values == c) & (R.ad_tok.fillna("").str.len().values > 0))
    if len(qi) == 0 or len(ri) == 0:
        continue
    B = hv.transform(R.ad_tok.values[ri].astype(str)).tocsr()
    df = np.bincount(B.indices, minlength=NF).astype(np.float32)
    idf = np.log((len(ri) + 1) / (df + 1)).astype(np.float32); idf[df == 0] = 0
    def weigh(M):
        M.data = idf[M.indices]
        nrm = np.sqrt(np.asarray(M.multiply(M).sum(1)).ravel()); nrm[nrm == 0] = 1
        return sp.diags(1 / nrm).dot(M).astype(np.float32).tocsr()
    B = weigh(B)
    A = weigh(hv.transform(S.ad_tok.fillna("").values[qi].astype(str)).tocsr())
    Ap = A.copy(); Ap.data[df[Ap.indices] > a.df_cap] = 0; Ap.eliminate_zeros()   # retrieve on rare tokens only
    BT = B.T.tocsr()
    for st in range(0, len(qi), 50_000):
        C = sp_matmul_topn(Ap[st:st + 50_000], BT, top_n=a.k * 3, threshold=0.05, n_threads=2).tocoo()
        rows, cols = st + C.row, C.col
        cos = np.asarray(A[rows].multiply(B[cols]).sum(1)).ravel().astype(np.float32)   # exact full cosine
        d = pd.DataFrame({"s1": qi[rows], "r": ri[cols], "cos": cos})
        d = d[d.cos >= a.min_cos].sort_values(["s1", "cos"], ascending=[True, False])
        res.append(d.groupby("s1").head(a.k))
    log(f"{c}: {len(qi)} queries vs {len(ri)} records")
    del A, Ap, B, BT
P = pd.concat(res, ignore_index=True)
# drop pairs already in the main candidate set
main = pd.read_parquet(os.path.join(a.pkg, "val_pairs.parquet" if a.mode == "dev" else "test_pairs.parquet"),
                       columns=["s1", "r"])
key = lambda s, r: s.astype(np.int64) * (1 << 25) + r.astype(np.int64)
P = P[~np.isin(key(P.s1.values, P.r.values), key(main.s1.values, main.r.values))].reset_index(drop=True)
P.to_parquet(os.path.join(a.out, f"{a.mode}_addr_pairs.parquet"))
log(f"new pairs {len(P)} ({len(P)/len(qidx):.2f}/query)")
if a.mode == "dev":
    gt = pd.read_parquet(os.path.join(a.work, "gt_train.parquet"))
    y = np.isin(key(P.s1.values, P.r.values), key(gt.s1.values, gt.r.values))
    tv = gt[np.isin(gt.s1.values, vids)]
    inval = np.isin(P.s1.values, vids)
    miss = len(tv) - np.isin(key(tv.s1.values, tv.r.values), key(main.s1.values, main.r.values)).sum()
    log(f"positives {y.sum()} (rate {y.mean():.3f}); VAL recovers {y[inval].sum()} of {miss} missed truth pairs")
