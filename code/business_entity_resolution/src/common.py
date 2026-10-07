"""Shared I/O helpers and the F0.5 metric."""
import os
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pcmp
import pyarrow.csv as pc
import pyarrow.parquet as pq


def load_split(cache_dir, split, columns=None):
    s1 = pq.read_table(os.path.join(cache_dir, f"{split}_s1.parquet"), columns=columns)
    s2 = pq.read_table(os.path.join(cache_dir, f"{split}_s2.parquet"), columns=columns)
    s3 = pq.read_table(os.path.join(cache_dir, f"{split}_s3.parquet"), columns=columns)
    r = pa.concat_tables([s2, s3]).combine_chunks()
    return s1.combine_chunks(), r


def load_gt_pairs(gt_path, s1_ids, r_ids):
    """Ground-truth as integer (s1_idx, r_idx) pairs aligned with the given id arrays."""
    gt = pc.read_csv(gt_path, parse_options=pc.ParseOptions(delimiter="\t", quote_char=False),
                     convert_options=pc.ConvertOptions(strings_can_be_null=False))
    lists = pcmp.split_pattern(gt["matched_entity_ids"], ",")
    flat = pcmp.list_flatten(lists)
    par = pcmp.list_parent_indices(lists)
    keep = pcmp.not_equal(flat, "")
    flat = pcmp.filter(flat, keep)
    par = pcmp.filter(par, keep).to_numpy()
    s1_of_gt = pcmp.index_in(gt["source1_entity_id"], value_set=s1_ids).to_numpy()
    r_idx = pcmp.index_in(flat, value_set=r_ids).fill_null(-1).to_numpy()
    s1_idx = s1_of_gt[par]
    ok = r_idx >= 0
    return pd.DataFrame({"s1": s1_idx[ok].astype(np.int64), "r": r_idx[ok].astype(np.int64)})


def f05_macro(pred: pd.DataFrame, truth: pd.DataFrame, s1_universe: np.ndarray, beta=0.5):
    """Macro F-beta over the given Source1 indices. pred/truth: DataFrames with columns s1, r."""
    s1_universe = np.asarray(s1_universe)
    u = pd.Index(s1_universe)
    pred = pred[pred.s1.isin(u)]
    truth = truth[truth.s1.isin(u)]
    tp = pred.merge(truth, on=["s1", "r"]).groupby("s1").size()
    npred = pred.groupby("s1").size()
    ntrue = truth.groupby("s1").size()
    df = pd.DataFrame(index=u)
    df["tp"] = tp.reindex(u).fillna(0).values
    df["np"] = npred.reindex(u).fillna(0).values
    df["nt"] = ntrue.reindex(u).fillna(0).values
    b2 = beta * beta
    f = np.zeros(len(df))
    both_empty = (df.np.values == 0) & (df.nt.values == 0)
    f[both_empty] = 1.0
    m = (df.tp.values > 0)
    p = df.tp.values[m] / df.np.values[m]
    r = df.tp.values[m] / df.nt.values[m]
    f[m] = (1 + b2) * p * r / (b2 * p + r)
    return float(f.mean())
