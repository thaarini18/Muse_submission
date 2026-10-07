"""Turning pair probabilities into final match lists.

1. exclusivity: a Source2/3 record describes one real business, so it is kept only for the Source1
   entity that gives it the highest probability;
2. per-entity selection: keep candidates with p >= thr.
"""
import numpy as np
import pandas as pd


def exclusive(P: pd.DataFrame, p: np.ndarray):
    """Keep, for every r, only the row with the highest probability."""
    d = P[["s1", "r"]].copy()
    d["p"] = p
    best = d.groupby("r")["p"].transform("max")
    return d[d.p >= best]


def decide(P: pd.DataFrame, p: np.ndarray, thr=0.5, use_exclusive=True):
    d = exclusive(P, p) if use_exclusive else P[["s1", "r"]].assign(p=p)
    return d[d.p >= thr][["s1", "r"]]
