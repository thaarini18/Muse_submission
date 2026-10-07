"""Stage 5a: package for the GPU cross-encoder stage (ce_train.py).

Only pairs whose LightGBM probability is uncertain are rescored by the cross-encoder; everything else keeps
the LightGBM decision. Files (parquet, zstd) in --out:
  ce_train.parquet    s1_name, s1_addr, r_name, r_addr, y            (training pairs, from TRN entities)
  val_pairs.parquet   s1, r, y, p_lgb, s1_name, s1_addr, r_name, r_addr (VALID entities, all final candidates)
  val_meta.parquet    val_s1 (200k ids) ; val_truth.parquet (s1, r)
  test_pairs.parquet  s1, r, p_lgb, s1_name, s1_addr, r_name, r_addr   (all final candidates)
  test_ids.parquet    s1 entity_id list (row order = s1 index) ; test_r_ids.parquet (r entity_id)
"""
import os, gc, time
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq, lightgbm as lgb
from common import load_split
from model import context_features
from round2 import assemble_final, NEW_FILTER_THR, NEW_FEATS
from run_pipeline import _perm, N_VALID, _key

import argparse
_ap = argparse.ArgumentParser(); _ap.add_argument("--work", required=True); _ap.add_argument("--sim", required=True); _ap.add_argument("--out", required=True)
_a = _ap.parse_args()
W, S, OUT = _a.work, _a.sim, _a.out
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
wr = lambda df, n: pq.write_table(pa.Table.from_pandas(df, preserve_index=False), os.path.join(OUT, n), compression="zstd")


def texts(split, s1_idx, r_idx):
    s1, r = load_split(W, split, columns=["entity_id", "business_name", "business_address"])
    a = s1.take(pa.array(s1_idx)).to_pandas(); b = r.take(pa.array(r_idx)).to_pandas()
    return a, b


# ---------------- validation: recompute final LightGBM probabilities for VALID rows
if not os.path.exists(os.path.join(OUT, "val_pairs.parquet")):
    n1 = len(np.load(f"{S}/hidden.npy")); perm = _perm(n1); vids = perm[:N_VALID]
    dev1 = pd.read_parquet(f"{S}/dev_r1.parquet"); r1 = pd.read_parquet(f"{S}/r1_all.parquet")
    sib1 = np.load(f"{S}/dev_r2_sib1.npy"); Pn = pd.read_parquet(f"{S}/dev_r2_Pn.parquet"); Xn = np.load(f"{S}/dev_r2_Xn.npy")
    pn = lgb.Booster(model_file=f"{S}/newfilter.txt").predict(Xn, num_threads=2).astype(np.float32)
    k2 = pn >= NEW_FILTER_THR
    U = pd.concat([r1[["s1", "r"]], Pn[k2][["s1", "r"]]], ignore_index=True)
    up = np.concatenate([r1.p1.values, pn[k2]]).astype(np.float32)
    unfe = np.concatenate([r1.nfe.values, Xn[k2, NEW_FEATS.index("num_first_eq")]]).astype(np.float32)
    uleg = np.concatenate([r1.leg.values, Xn[k2, NEW_FEATS.index("legal_eq")]]).astype(np.float32)
    C = context_features(U, None, up, nfe=unfe, leg=uleg)
    C = np.vstack([C[:len(dev1)], C[len(r1):]]); del U, up, r1
    v1 = dev1.valid.values; vn = np.isin(Pn.s1.values[k2], vids)
    X1 = np.load(f"{S}/dev_r1_X.npy", mmap_mode="r")
    Xv = assemble_final(np.asarray(X1[v1]), sib1[v1], Xn[k2][vn], np.vstack([C[:len(dev1)][v1], C[len(dev1):][vn]]))
    p = lgb.Booster(model_file=f"{S}/final.txt").predict(Xv, num_threads=2)
    gt = pd.read_parquet(f"{W}/gt_train.parquet")
    Pv = pd.concat([dev1[v1][["s1", "r"]], Pn[k2][vn][["s1", "r"]]], ignore_index=True)
    Pv["y"] = np.isin(_key(Pv.s1, Pv.r), _key(gt.s1, gt.r)).astype(np.int8); Pv["p_lgb"] = p.astype(np.float32)
    a, b = texts("train", Pv.s1.values, Pv.r.values)
    Pv["s1_name"], Pv["s1_addr"] = a.business_name.values, a.business_address.values
    Pv["r_name"], Pv["r_addr"] = b.business_name.values, b.business_address.values
    wr(Pv, "val_pairs.parquet")
    wr(pd.DataFrame({"val_s1": vids.astype(np.int64)}), "val_meta.parquet")
    wr(gt[np.isin(gt.s1.values, vids)].reset_index(drop=True), "val_truth.parquet")
    log(f"val pairs {len(Pv)}; uncertain(0.02-0.98) {((p > .02) & (p < .98)).mean():.3f}")
    del Xv, X1, Xn, C; gc.collect()

# ---------------- cross-encoder training pairs (TRN entities): hard + random easy
if not os.path.exists(os.path.join(OUT, "ce_train.parquet")):
    dev1 = pd.read_parquet(f"{S}/dev_r1.parquet"); dev1 = dev1[~dev1.valid.values]
    Pn = pd.read_parquet(f"{S}/dev_r2_Pn.parquet"); gt = pd.read_parquet(f"{W}/gt_train.parquet")
    n1 = len(np.load(f"{S}/hidden.npy")); vids = _perm(n1)[:N_VALID]
    Pn = Pn[~np.isin(Pn.s1.values, vids)]
    Pn["y"] = np.isin(_key(Pn.s1, Pn.r), _key(gt.s1, gt.r)).astype(np.int8)
    rng = np.random.default_rng(11)
    hard = dev1[(dev1.p1 > 0.02) & (dev1.p1 < 0.98)]
    easy = dev1.drop(hard.index).sample(300_000, random_state=1)
    r2 = pd.concat([Pn[Pn.y == 1], Pn[Pn.y == 0].sample(min(150_000, (Pn.y == 0).sum()), random_state=2)])
    T = pd.concat([hard[["s1", "r", "y"]], easy[["s1", "r", "y"]], r2[["s1", "r", "y"]]], ignore_index=True)
    if len(T) > 1_200_000:
        T = T.sample(1_200_000, random_state=3)
    T = T.sample(frac=1, random_state=4).reset_index(drop=True)
    a, b = texts("train", T.s1.values, T.r.values)
    out = pd.DataFrame({"s1_name": a.business_name.values, "s1_addr": a.business_address.values,
                        "r_name": b.business_name.values, "r_addr": b.business_address.values, "y": T.y.values})
    wr(out, "ce_train.parquet")
    log(f"ce_train {len(out)} pos {out.y.mean():.3f} (hard {len(hard)})")

# ---------------- test
if not os.path.exists(os.path.join(OUT, "test_pairs.parquet")):
    U = pd.read_parquet(f"{S}/test_pairs.parquet"); U["p_lgb"] = np.load(f"{S}/test_p.npy").astype(np.float32)
    mm = ((U.p_lgb > 0.02) & (U.p_lgb < 0.98)).values
    a, b = texts("test", U.s1.values[mm], U.r.values[mm])
    T = pd.DataFrame({"s1_name": a.business_name.values, "s1_addr": a.business_address.values,
                      "r_name": b.business_name.values, "r_addr": b.business_address.values})
    wr(T, "test_uncertain_text.parquet")
    wr(U[["s1", "r", "p_lgb"]], "test_pairs.parquet")
    s1, r = load_split(W, "test", columns=["entity_id"])
    wr(pd.DataFrame({"entity_id": s1["entity_id"].to_pylist()}), "test_ids.parquet")
    wr(pd.DataFrame({"entity_id": r["entity_id"].to_pylist()}), "test_r_ids.parquet")
    log(f"test pairs {len(U)}; uncertain {((U.p_lgb > .02) & (U.p_lgb < .98)).mean():.3f}")
os.system(f"ls -la {OUT}")
