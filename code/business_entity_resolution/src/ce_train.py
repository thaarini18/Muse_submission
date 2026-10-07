"""Stage 5b (GPU): cross-encoder rescoring of uncertain pairs, blend with LightGBM, write submission files.

Model: distilbert-base-multilingual-cased (Apache-2.0, 134M params) by default; any MIT/Apache model
<= 8B parameters can be passed with --model (e.g. xlm-roberta-base, MIT).

    python src/ce_train.py --pkg <ce_prep output dir> --out <output dir> [--model ...] [--epochs 1]

Only pairs with LO < p_lgb < HI are rescored; the rest keep the LightGBM decision. The blend (logistic
regression on logit(p_lgb), logit(p_ce), product) and the threshold are fitted on held-out VALID entities.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

ap = argparse.ArgumentParser()
ap.add_argument("--pkg", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="distilbert-base-multilingual-cased")
ap.add_argument("--maxlen", type=int, default=96)
ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--epochs", type=float, default=1.0)
ap.add_argument("--lr", type=float, default=3e-5)
ap.add_argument("--warmup", type=float, default=0.05, help="linear warm-up fraction of the training steps")
ap.add_argument("--lo", type=float, default=0.02)
ap.add_argument("--hi", type=float, default=0.98)
ap.add_argument("--workers", type=int, default=2)
ap.add_argument("--sib", default=None, help="ce_prep_sib output dir: sibling-aware variant (candidate ## best sibling)")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--time_budget", type=float, default=0, help="stop fine-tuning after this many minutes (0 = no limit)")
ap.add_argument("--tag", default="ce", help="name of this model's score files (<tag>_val.npy / <tag>_test.npy)")
a = ap.parse_args()

# ---- reproducibility: fixed seeds + deterministic kernels (set before any CUDA work)
import random
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONHASHSEED", str(a.seed))
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
D, OUT = a.pkg, a.out
os.makedirs(OUT, exist_ok=True)
CKPT = os.path.join(OUT, f"{a.tag}_model")
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "")
tok = AutoTokenizer.from_pretrained(a.model)


def side(name, addr, sn=None, sa=None):
    s = name.fillna("").astype(str).values + " | " + addr.fillna("").astype(str).values
    if sn is not None:        # sibling-aware: append the entity's best other candidate
        s = s + " ## " + sn.fillna("").astype(str).values + " | " + sa.fillna("").astype(str).values
    return s.tolist()


class PairData(torch.utils.data.Dataset):
    def __init__(self, x, y, lab=None):
        self.a, self.b, self.y = x, y, lab

    def __len__(self):
        return len(self.a)

    def __getitem__(self, i):
        return i


def collate_fn(ds):
    def f(idx):
        enc = tok([ds.a[i] for i in idx], [ds.b[i] for i in idx], truncation=True, max_length=a.maxlen,
                  padding=True, return_tensors="pt")
        if ds.y is not None:
            enc["labels"] = torch.tensor([ds.y[i] for i in idx], dtype=torch.float32)
        return enc
    return f


@torch.no_grad()
def predict(model, x, y, bs=512):
    model.eval()
    ds = PairData(x, y)
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, collate_fn=collate_fn(ds), num_workers=a.workers)
    out = []
    for i, enc in enumerate(dl):
        enc = {k: v.to(dev, non_blocking=True) for k, v in enc.items()}
        with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
            out.append(torch.sigmoid(model(**enc).logits.float().squeeze(-1)).cpu().numpy())
        if i % 200 == 0:
            print(f"  predict {i * bs}/{len(ds)}", flush=True)
    return np.concatenate(out) if out else np.zeros(0, np.float32)


# ---------------------------------------------------------------- 1. fine-tune
if os.path.exists(os.path.join(CKPT, "config.json")):
    model = AutoModelForSequenceClassification.from_pretrained(CKPT).to(dev)
    print("loaded saved model")
else:
    tr = pd.read_parquet(os.path.join(a.sib, "ce_train_sib.parquet") if a.sib else os.path.join(D, "ce_train.parquet"))
    x, lab = side(tr.s1_name, tr.s1_addr), tr.y.values.astype(np.float32)
    y = side(tr.r_name, tr.r_addr, *( (tr.sib_name, tr.sib_addr) if a.sib else () ))
    del tr
    model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=1).to(dev)
    ds = PairData(x, y, lab)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, shuffle=True, collate_fn=collate_fn(ds),
                                     generator=torch.Generator().manual_seed(a.seed),
                                     num_workers=a.workers, drop_last=True)
    total = int(len(dl) * a.epochs)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(a.warmup * total), total)
    scaler = torch.amp.GradScaler(dev, enabled=dev == "cuda")
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    t0, run, step = time.time(), 0.0, 0
    while step < total:
        for enc in dl:
            enc = {k: v.to(dev, non_blocking=True) for k, v in enc.items()}
            lb = enc.pop("labels")
            with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
                loss = lossf(model(**enc).logits.float().squeeze(-1), lb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sch.step()
            run = 0.98 * run + 0.02 * loss.item()
            if step % 500 == 0:
                print(f"step {step}/{total} loss {run:.4f} {time.time() - t0:.0f}s", flush=True)
            step += 1
            if step >= total or (a.time_budget and time.time() - t0 > 60 * a.time_budget):
                total = step
                break
    model.save_pretrained(CKPT)
    tok.save_pretrained(CKPT)
print("model ready")

# ---------------------------------------------------------------- 2. score uncertain pairs (cached)
val = pd.read_parquet(os.path.join(D, "val_pairs.parquet"))
mv = ((val.p_lgb > a.lo) & (val.p_lgb < a.hi)).values
fv = os.path.join(OUT, f"{a.tag}_val.npy")
if not os.path.exists(fv):
    sv = pd.read_parquet(os.path.join(a.sib, "val_sib_text.parquet")) if a.sib else None   # rows = val[0.02<p<0.98]
    base_v = ((val.p_lgb > 0.02) & (val.p_lgb < 0.98)).values
    extra = () if sv is None else (sv.sib_name[mv[base_v]].reset_index(drop=True), sv.sib_addr[mv[base_v]].reset_index(drop=True))
    np.save(fv, predict(model, side(val.s1_name[mv], val.s1_addr[mv]),
                        side(val.r_name[mv].reset_index(drop=True), val.r_addr[mv].reset_index(drop=True), *extra)))
val_ce = np.load(fv)
test = pd.read_parquet(os.path.join(D, "test_pairs.parquet"))
base = ((test.p_lgb > 0.02) & (test.p_lgb < 0.98)).values          # rows present in the text file
mt = ((test.p_lgb > a.lo) & (test.p_lgb < a.hi)).values & base
ft = os.path.join(OUT, f"{a.tag}_test.npy")
if not os.path.exists(ft):
    tt = pd.read_parquet(os.path.join(D, "test_uncertain_text.parquet"))   # rows = test[base], same order
    assert len(tt) == base.sum()
    sel = mt[base]
    extra = ()
    if a.sib:
        st_ = pd.read_parquet(os.path.join(a.sib, "test_sib_text.parquet"))                # rows = test[base]
        extra = (st_.sib_name[sel].reset_index(drop=True), st_.sib_addr[sel].reset_index(drop=True))
    np.save(ft, predict(model, side(tt.s1_name[sel], tt.s1_addr[sel]),
                        side(tt.r_name[sel].reset_index(drop=True), tt.r_addr[sel].reset_index(drop=True), *extra)))
    del tt
test_ce = np.load(ft)
print("scored: val", mv.sum(), "test", mt.sum())

# ---------------------------------------------------------------- 3. blend + threshold on VALID entities
def lg(p):
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def feats(pl, pc):
    return np.column_stack([lg(pl), lg(pc), lg(pl) * lg(pc)])


lr = LogisticRegression(C=1.0, max_iter=1000).fit(feats(val.p_lgb.values[mv], val_ce), val.y.values[mv])


def blended(df, m, ce):
    p = df.p_lgb.values.astype(np.float64).copy()
    p[m] = lr.predict_proba(feats(df.p_lgb.values[m], ce))[:, 1]
    return p


truth = pd.read_parquet(os.path.join(D, "val_truth.parquet"))
vids = pd.read_parquet(os.path.join(D, "val_meta.parquet")).val_s1.values


def decide(df, p, thr):
    d = pd.DataFrame({"s1": df.s1.values, "r": df.r.values, "p": p})
    d = d[d.p >= d.groupby("r").p.transform("max")]      # each record -> its best Source1 entity only
    return d[d.p >= thr][["s1", "r"]]


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


pv = blended(val, mv, val_ce)
rows = [(t, f05(decide(val, val.p_lgb.values, t)), f05(decide(val, pv, t)))
        for t in (0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)]
res = pd.DataFrame(rows, columns=["thr", "F05_lgb_only", "F05_blend"])
print(res.to_string(index=False))
best = res.loc[res.F05_blend.idxmax()]
rep = {"model": a.model, "coef": lr.coef_.tolist(), "table": rows, "best_thr": float(best.thr),
       "best_blend": float(best.F05_blend), "best_lgb_only": float(res.F05_lgb_only.max())}
json.dump(rep, open(os.path.join(OUT, f"{a.tag}_report.json"), "w"), indent=1)

# ---------------------------------------------------------------- 4. write submission files
ids = pd.read_parquet(os.path.join(D, "test_ids.parquet")).entity_id.values
rids = pd.read_parquet(os.path.join(D, "test_r_ids.parquet")).entity_id.values
pt = blended(test, mt, test_ce)


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


thr = float(best.thr)
write(decide(test, pt, thr), os.path.join(OUT, "matching_results.tsv"), "matched_entity_ids")
write(decide(test, pt, min(thr + 0.1, 0.95)), os.path.join(OUT, "matching_results_stricter.tsv"),
      "matched_entity_ids")
write(test[["s1", "r"]], os.path.join(OUT, "candidate_pairs.tsv"), "candidate_entity_ids")
print(json.dumps(rep, indent=1))
