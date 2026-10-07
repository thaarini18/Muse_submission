"""Stage 6: final decision + submission files.

    python src/final_stack.py --pkg ce_pkg --scores scores --addr addr --out output

- Main pairs: stacker probability (ce_stack.py). Each record goes to its highest-probability Source1 entity
  (exclusivity); a pair is kept if p >= 0.75.
- Empty-entity rescue: an entity left with no match keeps its single best candidate if p >= 0.5.
- Address pass (addr_pass.py + addr_score_v2.py): pairs with p >= 0.5 join the candidate set; those whose
  record was not claimed above are added to the matches.
Writes <out>/matching_results.tsv and <out>/candidate_pairs.tsv (every matched ID is inside the candidates).
"""
import argparse, os
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--pkg", required=True); ap.add_argument("--scores", required=True)
ap.add_argument("--addr", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--thr", type=float, default=0.75); ap.add_argument("--rescue", type=float, default=0.5)
ap.add_argument("--addr_thr", type=float, default=0.5)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
D = a.pkg
test = pd.read_parquet(os.path.join(D, "test_pairs.parquet"), columns=["s1", "r"])
test["p"] = np.load(os.path.join(a.scores, "stack_test_p.npy"))


def decide(P, thr):
    d = P[P.p >= P.groupby("r").p.transform("max")]
    return d[d.p >= thr][["s1", "r"]]


match = decide(test, a.thr)
d = test[test.p >= test.groupby("r").p.transform("max")]
best = d.sort_values("p", ascending=False).drop_duplicates("s1")
resc = best[(~best.s1.isin(set(match.s1))) & (best.p >= a.rescue)][["s1", "r"]]
match = pd.concat([match, resc], ignore_index=True)

X = pd.read_parquet(os.path.join(a.addr, "test_addr_scored_v2.parquet"), columns=["s1", "r", "p"])
X = X.sort_values("p", ascending=False).drop_duplicates("r")
k = lambda s, r: np.asarray(r, np.int64) * (1 << 25) + np.asarray(s, np.int64)
X = X[~np.isin(k(X.s1, X.r), k(test.s1, test.r))]
add = X[(X.p >= a.addr_thr) & ~np.isin(X.r.values, match.r.values)]
match = pd.concat([match, add[["s1", "r"]]], ignore_index=True)

ids = pd.read_parquet(os.path.join(D, "test_ids.parquet")).entity_id.values
rids = pd.read_parquet(os.path.join(D, "test_r_ids.parquet")).entity_id.values


def write(pairs, path, col):
    pairs = pairs.sort_values(["s1", "r"]); s1v = pairs.s1.values; rs = rids[pairs.r.values]
    st = np.flatnonzero(np.r_[True, s1v[1:] != s1v[:-1]])
    c = np.full(len(ids), "", dtype=object); c[s1v[st]] = [",".join(x) for x in np.split(rs, st[1:])]
    pd.DataFrame({"source1_entity_id": ids, col: c}).to_csv(path, sep="\t", index=False)
    print("wrote", path)


write(match, os.path.join(a.out, "matching_results.tsv"), "matched_entity_ids")
C = pd.concat([test[["s1", "r"]], X[X.p >= 0.5][["s1", "r"]]], ignore_index=True)
write(C, os.path.join(a.out, "candidate_pairs.tsv"), "candidate_entity_ids")
assert np.isin(k(match.s1, match.r), k(C.s1, C.r)).all()
print(f"matches {len(match)} (rescued {len(resc)}, address {len(add)}); candidates {len(C)}")
