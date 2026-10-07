"""Stage 5a': package for the SIBLING-AWARE cross-encoder.

For every pair (s1, r) we add the text of the best OTHER candidate of the same Source1 entity
("sibling" = highest LightGBM / stage-1 probability, excluding r). The cross-encoder then sees
    [S1 name | address]  [SEP]  [candidate name | address] ## [sibling name | address]
which lets it accept fragments that resemble an already-matched fragment (native-script names,
truncated addresses) and reject look-alikes that match neither.
Outputs (parquet, zstd): ce_train_sib.parquet, val_sib_text.parquet (rows = val[mv] order),
test_sib_text.parquet (rows = test[mt] order, mt = 0.02 < p_lgb < 0.98).
"""
import argparse, os, time
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
from common import load_split

ap = argparse.ArgumentParser()
ap.add_argument("--work", required=True); ap.add_argument("--sim", required=True)
ap.add_argument("--pkg", required=True, help="existing ce_prep output (val_pairs/test_pairs/ce_train)")
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
t0 = time.time()
log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
wr = lambda df, n: pq.write_table(pa.Table.from_pandas(df, preserve_index=False), os.path.join(a.out, n), compression="zstd")


def sibling_of(s1, r, p):
    """For each row: r-index of the highest-p OTHER row with the same s1 (-1 if none)."""
    d = pd.DataFrame({"s1": s1, "r": r, "p": p}).sort_values(["s1", "p"], ascending=[True, False])
    rk = d.groupby("s1").cumcount().values
    top1 = pd.Series(d.r.values[rk == 0], index=d.s1.values[rk == 0])
    top2 = pd.Series(d.r.values[rk == 1], index=d.s1.values[rk == 1])
    t1 = top1.reindex(s1).values.astype(np.int64)
    t2 = top2.reindex(s1).fillna(-1).values.astype(np.int64)
    return np.where(t1 == r, t2, t1)


def fetch(tbl, idx):
    ok = idx >= 0
    out_n = np.full(len(idx), "", dtype=object); out_a = np.full(len(idx), "", dtype=object)
    if ok.any():
        t = tbl.take(pa.array(idx[ok])).to_pandas()
        out_n[ok] = t.business_name.values; out_a[ok] = t.business_address.values
    return out_n, out_a


# ---------------- validation / test: siblings from the full candidate lists (p_lgb)
for split, fn_pairs, fn_out, lo_hi in (("train", "val_pairs.parquet", "val_sib_text.parquet", True),
                                        ("test", "test_pairs.parquet", "test_sib_text.parquet", True)):
    if os.path.exists(os.path.join(a.out, fn_out)):
        continue
    P = pd.read_parquet(os.path.join(a.pkg, fn_pairs), columns=["s1", "r", "p_lgb"])
    sib = sibling_of(P.s1.values, P.r.values, P.p_lgb.values)
    m = ((P.p_lgb > 0.02) & (P.p_lgb < 0.98)).values
    _, rt = load_split(a.work, split, columns=["business_name", "business_address"])
    n, ad = fetch(rt, sib[m])
    wr(pd.DataFrame({"sib_name": n, "sib_addr": ad}), fn_out)
    log(f"{split}: {m.sum()} uncertain rows, {(sib[m] >= 0).mean():.3f} have a sibling")

# ---------------- training pairs: same selection as ce_prep, siblings from stage-1 OOF p1
if not os.path.exists(os.path.join(a.out, "ce_train_sib.parquet")):
    from run_pipeline import _perm, N_VALID, _key
    S, W = a.sim, a.work
    dev1 = pd.read_parquet(f"{S}/dev_r1.parquet")
    sib_all = sibling_of(dev1.s1.values, dev1.r.values, dev1.p1.values)
    sib_map = dict(zip(_key(dev1.s1, dev1.r).tolist(), sib_all.tolist()))
    top_map = dev1.sort_values("p1", ascending=False).drop_duplicates("s1").set_index("s1").r
    trn = dev1[~dev1.valid.values]
    Pn = pd.read_parquet(f"{S}/dev_r2_Pn.parquet"); gt = pd.read_parquet(f"{W}/gt_train.parquet")
    vids = _perm(len(np.load(f"{S}/hidden.npy")))[:N_VALID]
    Pn = Pn[~np.isin(Pn.s1.values, vids)].copy()
    Pn["y"] = np.isin(_key(Pn.s1, Pn.r), _key(gt.s1, gt.r)).astype(np.int8)
    hard = trn[(trn.p1 > 0.02) & (trn.p1 < 0.98)]
    easy = trn.drop(hard.index).sample(300_000, random_state=1)
    r2 = pd.concat([Pn[Pn.y == 1], Pn[Pn.y == 0].sample(min(150_000, int((Pn.y == 0).sum())), random_state=2)])
    T = pd.concat([hard[["s1", "r", "y"]], easy[["s1", "r", "y"]], r2[["s1", "r", "y"]]], ignore_index=True)
    if len(T) > 1_200_000:
        T = T.sample(1_200_000, random_state=3)
    T = T.sample(frac=1, random_state=4).reset_index(drop=True)
    k = _key(T.s1, T.r).tolist()
    sib = np.array([sib_map.get(x, -2) for x in k], np.int64)
    miss = sib == -2                                   # round-2 rows: sibling = entity's best round-1 row
    sib[miss] = top_map.reindex(T.s1.values[miss]).fillna(-1).values.astype(np.int64)
    sib[sib == T.r.values] = -1
    s1t, rt = load_split(W, "train", columns=["business_name", "business_address"])
    A = s1t.take(pa.array(T.s1.values)).to_pandas(); B = rt.take(pa.array(T.r.values)).to_pandas()
    n, ad = fetch(rt, sib)
    out = pd.DataFrame({"s1_name": A.business_name.values, "s1_addr": A.business_address.values,
                        "r_name": B.business_name.values, "r_addr": B.business_address.values,
                        "sib_name": n, "sib_addr": ad, "y": T.y.values})
    wr(out, "ce_train_sib.parquet")
    log(f"ce_train_sib {len(out)} pos {out.y.mean():.3f}; with sibling {(sib >= 0).mean():.3f}")
os.system(f"ls -la {a.out}")
