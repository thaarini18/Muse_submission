"""Stage 3: pairwise features for (Source1 record, candidate Source2/3 record)."""
import math
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pcmp
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

from normalize import skeleton, ADDR_STOP

LEGAL_CANON = {
    "pvt": "pvt", "private": "pvt", "prvt": "pvt", "praivet": "pvt",
    "ltd": "ltd", "limited": "ltd", "limitd": "ltd",
    "llc": "llc", "inc": "inc", "incorporated": "inc", "corp": "corp", "corporation": "corp",
    "llp": "llp", "lp": "lp", "pc": "pc", "pllc": "pllc", "public": "public", "plc": "plc",
    "co": "co", "company": "co", "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl", "sa": "sa",
    "sci": "sci", "snc": "snc", "opc": "opc",
}
FEATURE_NAMES = None

# ------------------------------------------------------------------ idf tables

def skeleton_idf(str_col, addr=False):
    """dict skeleton-token -> idf over a pyarrow column of space-joined raw tokens (df aggregated
    after mapping every distinct raw token to its phonetic skeleton)."""
    lists = pcmp.split_pattern(str_col, " ")
    flat = pcmp.list_flatten(lists)
    vc = pcmp.value_counts(flat)
    toks = vc.field("values").to_pylist()
    cnt = vc.field("counts").to_numpy()
    df = {}
    for t, c in zip(toks, cnt.tolist()):
        if not t:
            continue
        if addr and (t.isdigit() or t in ADDR_STOP or len(t) < 3):
            continue
        k = skeleton(t)
        df[k] = df.get(k, 0) + c
    n = len(str_col)
    return {k: math.log1p(n / v) for k, v in df.items()}


# ------------------------------------------------------------------ per record parsing
_G = {}


def _init(name_idf, addr_idf):
    _G["nidf"] = name_idf
    _G["aidf"] = addr_idf


def parse(nm_ascii, nm_core, ad_tok, raw_name):
    core = nm_core.split()
    sk = [skeleton(t) for t in core]
    at = ad_tok.split()
    nums = [t.lstrip("0") or "0" for t in at if t.isdigit()]
    words = [skeleton(t) for t in at if not t.isdigit() and t not in ADDR_STOP and len(t) >= 3]
    legal = {LEGAL_CANON[t] for t in nm_ascii.replace(".", "").replace(",", " ").split() if t in LEGAL_CANON}
    return {
        "core": nm_core, "core_set": set(core), "sk": " ".join(sk), "sk_set": set(sk), "sk_list": sk,
        "concat": "".join(core), "ad": ad_tok, "nums": nums, "num_set": set(nums), "w_set": set(words),
        "legal": legal, "nonascii": 0 if raw_name.isascii() else 1,
        "domainish": 1 if any(x in nm_ascii for x in (".com", "www", "#", "@", ".in", ".net", ".org")) else 0,
        "n_core": len(core), "n_at": len(at),
    }


def _jac(a, b):
    if not a and not b:
        return -1.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def _wsum(s, idf, default):
    return sum(idf.get(t, default) for t in s)


def pair_features(A, B):
    nidf, aidf = _G["nidf"], _G["aidf"]
    f = []
    ca, cb = A["core"], B["core"]
    f += [fuzz.ratio(ca, cb), fuzz.token_set_ratio(ca, cb), fuzz.token_sort_ratio(ca, cb),
          fuzz.partial_ratio(ca, cb)]
    sa, sb = A["sk"], B["sk"]
    f += [fuzz.ratio(sa, sb), fuzz.token_set_ratio(sa, sb), fuzz.partial_ratio(sa, sb)]
    f += [JaroWinkler.similarity(A["concat"], B["concat"]),
          Levenshtein.normalized_similarity(A["concat"], B["concat"])]
    # common prefix of concatenations
    x, y = A["concat"], B["concat"]
    p = 0
    for c1, c2 in zip(x, y):
        if c1 != c2:
            break
        p += 1
    f.append(p)
    f.append(p / max(1, min(len(x), len(y))))
    # substring containment of concatenated names (domain-like names)
    f.append(1 if (len(y) >= 4 and y in x) or (len(x) >= 4 and x in y) else 0)
    ta, tb = A["sk_set"], B["sk_set"]
    inter = ta & tb
    f += [_jac(A["core_set"], B["core_set"]), _jac(ta, tb)]
    wa, wb, wi = _wsum(ta, nidf, 12.0), _wsum(tb, nidf, 12.0), _wsum(inter, nidf, 12.0)
    f += [wi, wi / wa if wa else 0, wi / wb if wb else 0, wb - wi, wa - wi]
    f.append(1 if A["sk_list"] and B["sk_list"] and A["sk_list"][0] == B["sk_list"][0] else 0)
    f += [A["n_core"], B["n_core"], A["nonascii"], B["nonascii"], B["domainish"]]
    la, lb = A["legal"], B["legal"]
    f += [len(la), len(lb), 1 if la == lb else 0, len(la & lb), len(la ^ lb),
          1 if (("public" in la) != ("public" in lb)) else 0]
    # address
    f.append(1 if not B["ad"] else 0)
    aa, ab = A["ad"], B["ad"]
    if aa and ab:
        f += [fuzz.ratio(aa, ab), fuzz.token_set_ratio(aa, ab), fuzz.partial_ratio(aa, ab)]
    else:
        f += [-1, -1, -1]
    na, nb = A["num_set"], B["num_set"]
    ni = na & nb
    f += [_jac(na, nb), len(ni), len(nb - na), len(na - nb)]
    f.append(1 if A["nums"] and B["nums"] and A["nums"][0] == B["nums"][0] else -1 if not B["nums"] else 0)
    f.append(1 if B["nums"] and B["nums"][0] in na else -1 if not B["nums"] else 0)
    # fuzzy number match (e.g. 1425 vs 5425, 4/79 vs 4/792)
    best = -1.0
    if nb and na:
        best = max(Levenshtein.normalized_similarity(p, q) for p in nb for q in na)
    f.append(best)
    wa_, wb_ = A["w_set"], B["w_set"]
    wi_ = wa_ & wb_
    xa, xb, xi = _wsum(wa_, aidf, 12.0), _wsum(wb_, aidf, 12.0), _wsum(wi_, aidf, 12.0)
    f += [_jac(wa_, wb_), xi, xi / xa if xa else 0, xi / xb if xb else -1, xb - xi, A["n_at"], B["n_at"]]
    return f


FEATURE_NAMES = (
    ["n_ratio", "n_tset", "n_tsort", "n_partial", "sk_ratio", "sk_tset", "sk_partial", "cc_jw", "cc_lev",
     "cc_prefix", "cc_prefix_rel", "cc_contain", "n_jac", "sk_jac", "n_widf_i", "n_widf_ra", "n_widf_rb",
     "n_widf_extra_b", "n_widf_extra_a", "n_first_eq", "a_ncore", "b_ncore", "a_nonascii", "b_nonascii",
     "b_domain", "a_nlegal", "b_nlegal", "legal_eq", "legal_i", "legal_x", "public_x",
     "b_addr_empty", "ad_ratio", "ad_tset", "ad_partial", "num_jac", "num_i", "num_extra_b", "num_extra_a",
     "num_first_eq", "num_b_first_in_a", "num_best_fuzzy", "aw_jac", "aw_idf_i", "aw_ra", "aw_rb",
     "aw_extra_b", "a_nat", "b_nat"])


def _work(args):
    recs_a, recs_b, pairs = args
    PA = {i: parse(*v) for i, v in recs_a.items()}
    PB = {i: parse(*v) for i, v in recs_b.items()}
    out = np.empty((len(pairs), len(FEATURE_NAMES)), np.float32)
    for n, (i, j) in enumerate(pairs):
        out[n] = pair_features(PA[i], PB[j])
    return out


def _records(table, idx):
    cols = [table[c].take(pa.array(idx)).to_pylist() for c in ("nm_ascii", "nm_core", "ad_tok", "business_name")]
    return dict(zip(idx.tolist(), zip(*cols)))


def make_pool(name_idf, addr_idf, workers=2):
    return Pool(workers, initializer=_init, initargs=(name_idf, addr_idf), maxtasksperchild=50)


def compute_features(s1, r, pairs: pd.DataFrame, name_idf, addr_idf, workers=2, chunk=20000, out=None,
                     pool=None):
    """pairs: DataFrame with s1, r (global row indices). Fills / returns a float32 matrix aligned with
    pairs (columns FEATURE_NAMES). `out` may be a larger preallocated matrix (first columns are used)."""
    order = np.argsort(pairs.s1.values, kind="stable")
    ps = pairs.s1.values[order]
    pr = pairs.r.values[order]
    nf = len(FEATURE_NAMES)
    if out is None:
        out = np.empty((len(pairs), nf), np.float32)

    def jobs():
        for st in range(0, len(order), chunk):
            a = ps[st:st + chunk]
            b = pr[st:st + chunk]
            yield (_records(s1, np.unique(a)), _records(r, np.unique(b)), list(zip(a.tolist(), b.tolist())))

    st = 0
    p = pool if pool is not None else make_pool(name_idf, addr_idf, workers)
    for res in p.imap(_work, jobs()):
        out[order[st:st + len(res)], :nf] = res
        st += len(res)
    if pool is None:
        p.close(); p.join()
    return out


# ------------------------------------------------------------------ sibling (group) features
SIB_NAMES = ["sib_n", "sib_name_tset", "sib_name_ratio", "sib_name_eq", "sib_sk_tset", "sib_addr_tset",
             "sib_addr_eq", "sib_num_share", "sib_addr_empty_b"]


def _sib_work(args):
    recs, jobs = args          # recs: idx -> (nm_core, ad_tok); jobs: list of (r, [sibling r...])
    out = np.empty((len(jobs), len(SIB_NAMES)), np.float32)
    for n, (r, sibs) in enumerate(jobs):
        nb, ab = recs[r]
        skb = " ".join(skeleton(t) for t in nb.split())
        numb = {t for t in ab.split() if t.isdigit()}
        if not sibs:
            out[n] = [0, -1, -1, -1, -1, -1, -1, -1, 1 if not ab else 0]
            continue
        bt = br = eq = bs = ba = aeq = ns = 0.0
        ba = -1.0
        for s in sibs:
            ns_, as_ = recs[s]
            bt = max(bt, fuzz.token_set_ratio(nb, ns_))
            br = max(br, fuzz.ratio(nb, ns_))
            eq = max(eq, 1.0 if nb == ns_ else 0.0)
            bs = max(bs, fuzz.token_set_ratio(skb, " ".join(skeleton(t) for t in ns_.split())))
            if ab and as_:
                ba = max(ba, fuzz.token_set_ratio(ab, as_))
                aeq = max(aeq, 1.0 if ab == as_ else 0.0)
            if numb and numb & {t for t in as_.split() if t.isdigit()}:
                ns = 1.0
        out[n] = [len(sibs), bt, br, eq, bs, ba, aeq, ns, 1 if not ab else 0]
    return out


def sibling_features(r_table, pairs: pd.DataFrame, seeds: pd.DataFrame, workers=2, chunk=50000):
    """For each candidate (s1, r): similarity of r to the confident round-1 matches ("seeds") of the same
    Source1 entity, excluding r itself. Captures group consistency: a true fragment usually looks like
    its siblings; a same-address sibling business usually does not share the name."""
    sib = seeds.groupby("s1")["r"].apply(lambda x: x.values.tolist()).to_dict()
    ps, pr = pairs.s1.values, pairs.r.values
    out = np.empty((len(pairs), len(SIB_NAMES)), np.float32)

    def jobs():
        for st in range(0, len(ps), chunk):
            js = [(int(r), [s for s in sib.get(int(a), []) if s != r]) for a, r in zip(ps[st:st + chunk], pr[st:st + chunk])]
            need = np.unique(np.array([r for r, _ in js] + [s for _, l in js for s in l], np.int64))
            cols = [r_table[c].take(pa.array(need)).to_pylist() for c in ("nm_core", "ad_tok")]
            yield dict(zip(need.tolist(), zip(*cols))), js

    st = 0
    with Pool(workers, maxtasksperchild=50) as p:
        for res in p.imap(_sib_work, jobs()):
            out[st:st + len(res)] = res
            st += len(res)
    return out
