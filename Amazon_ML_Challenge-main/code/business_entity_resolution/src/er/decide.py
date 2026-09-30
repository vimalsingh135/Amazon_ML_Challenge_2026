"""Decision layer: pair probabilities -> per-S1 match sets.

N1  target exclusivity: every S2/S3 record belongs to at most one S1 (verified on train),
    so a target is only kept for the S1 that wants it most (optionally with a margin).
N2  expected-F0.5 set selection: for each S1, candidates are sorted by probability and the
    prefix size k (k = 0 means "no match") that maximises expected F0.5 is chosen.
    E[F(k)] ~ (1+b2) * sum_{i<=k} p_i / (b2 * (sum_i p_i + m0) + k)   (ratio of expectations)
    E[F(0)] = P(no match) = prod_i (1 - p_i) * (1 - q_miss)
    m0 / q_miss absorb true matches that blocking never retrieved; all knobs are tuned on
    validation against the exact macro-F0.5 metric.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

B2 = 0.25


@dataclass
class DecisionParams:
    margin: float = 0.0       # exclusivity: required lead over the runner-up S1 for a target
    floor: float = 0.0        # never select a candidate below this probability
    m0: float = 0.0           # expected matches outside the candidate set
    empty_bias: float = 1.0   # multiplier on P(no match) (<1 discourages empty predictions)
    exclusive: bool = True


def select(df: pl.DataFrame, prm: DecisionParams, p: str = "p") -> pl.DataFrame:
    """df: s1_idx, src, t_idx, p. Returns selected (s1_idx, src, t_idx) pairs."""
    d = df.select("s1_idx", "src", "t_idx", pl.col(p).alias("p"))
    if prm.exclusive:
        d = d.with_columns(
            pl.col("p").rank("ordinal", descending=True).over("src", "t_idx").alias("_tr"),
            pl.col("p").top_k(2).min().over("src", "t_idx").alias("_p2"),
            pl.len().over("src", "t_idx").alias("_tn"))
        lead = pl.when(pl.col("_tn") > 1).then(pl.col("p") - pl.col("_p2")).otherwise(pl.col("p"))
        # candidates losing the target keep their probability mass for E[F] but cannot be chosen
        d = d.with_columns(((pl.col("_tr") == 1) & (lead >= prm.margin)).alias("_ok"))
    else:
        d = d.with_columns(pl.lit(True).alias("_ok"))
    d = d.with_columns(pl.when(pl.col("_ok")).then(pl.col("p")).otherwise(0.0).alias("_q"))
    d = d.sort(["s1_idx", "_q"], descending=[False, True]).with_columns(
        pl.col("_q").cum_sum().over("s1_idx").alias("_cs"),
        pl.int_range(1, pl.len() + 1).over("s1_idx").alias("_k"),
        pl.col("p").sum().over("s1_idx").alias("_tot"),
        (1 - pl.col("p")).log().sum().over("s1_idx").alias("_lp0"))
    d = d.with_columns(((1 + B2) * pl.col("_cs") / (B2 * (pl.col("_tot") + prm.m0) + pl.col("_k")))
                       .alias("_ef"))
    best = d.group_by("s1_idx").agg(
        pl.col("_ef").max().alias("_efmax"),
        pl.col("_k").get(pl.col("_ef").arg_max()).alias("_kbest"),
        (pl.col("_lp0").first().exp() * prm.empty_bias).alias("_ef0"))
    d = d.join(best, on="s1_idx").filter(
        (pl.col("_efmax") > pl.col("_ef0")) & (pl.col("_k") <= pl.col("_kbest"))
        & pl.col("_ok") & (pl.col("p") >= prm.floor))
    return d.select("s1_idx", "src", "t_idx")


def threshold_select(df: pl.DataFrame, thr: float, p: str = "p") -> pl.DataFrame:
    """Baseline: global threshold + exclusivity (argmax S1 per target)."""
    d = df.filter(pl.col(p) >= thr).with_columns(
        pl.col(p).rank("ordinal", descending=True).over("src", "t_idx").alias("_tr"))
    return d.filter(pl.col("_tr") == 1).select("s1_idx", "src", "t_idx")


# ---------------------------------------------------------------------------
# Exact Bayes-optimal expected-F selection (independent candidates)
# ---------------------------------------------------------------------------
def _pb_prefix(P: np.ndarray) -> list[np.ndarray]:
    """Poisson-binomial distributions of #true among the first k columns, k = 0..n.
    P: (G, n) probabilities. Returns list of (G, k+1) arrays."""
    G, n = P.shape
    out = [np.ones((G, 1))]
    for k in range(n):
        prev, p = out[-1], P[:, k:k + 1]
        nxt = np.zeros((G, k + 2))
        nxt[:, :-1] += prev * (1 - p)
        nxt[:, 1:] += prev * p
        out.append(nxt)
    return out


def expected_f_exact(P: np.ndarray, beta2: float = B2, m0: float = 0.0) -> np.ndarray:
    """E[F_beta] of predicting the top-k (columns already sorted descending), k = 0..n.

    Candidates are independent Bernoulli(p). ``m0`` is the expected number of true matches the
    candidate set missed (added to the recall denominator). k = 0 scores 1 only if there are
    no true matches: P(all false) * exp(-m0).  Returns (G, n+1)."""
    G, n = P.shape
    pre = _pb_prefix(P)
    suf = _pb_prefix(P[:, ::-1])          # suf[j] = distribution over the last j columns
    ef = np.zeros((G, n + 1))
    ef[:, 0] = pre[n][:, 0] * np.exp(-m0)
    for k in range(1, n + 1):
        A, B = pre[k], suf[n - k]          # A: #true in top-k, B: #true in the rest
        a = np.arange(k + 1)[:, None]
        b = np.arange(n - k + 1)[None, :]
        f = (1 + beta2) * a / (k + beta2 * (a + b + m0))          # (k+1, n-k+1)
        ef[:, k] = np.einsum("ga,ab,gb->g", A, f, B)
    return ef


def select_exact(df: pl.DataFrame, prm: DecisionParams, p: str = "p", max_n: int = 40) -> pl.DataFrame:
    """Like ``select`` but maximises the exact expected F0.5 per S1 (vectorised by set size)."""
    d = df.select("s1_idx", "src", "t_idx", pl.col(p).alias("p"))
    if prm.exclusive:
        d = d.with_columns(
            pl.col("p").rank("ordinal", descending=True).over("src", "t_idx").alias("_tr"),
            pl.col("p").top_k(2).min().over("src", "t_idx").alias("_p2"),
            pl.len().over("src", "t_idx").alias("_tn"))
        lead = pl.when(pl.col("_tn") > 1).then(pl.col("p") - pl.col("_p2")).otherwise(pl.col("p"))
        d = d.with_columns(((pl.col("_tr") == 1) & (lead >= prm.margin)).alias("_ok"))
    else:
        d = d.with_columns(pl.lit(True).alias("_ok"))
    # candidates that cannot be chosen keep their mass in the "rest" (they may still be true)
    d = d.with_columns(pl.when(pl.col("_ok")).then(pl.col("p")).otherwise(0.0).alias("_q"))
    d = d.sort(["s1_idx", "_ok", "p"], descending=[False, True, True]).with_columns(
        pl.int_range(1, pl.len() + 1).over("s1_idx").alias("_k"),
        pl.col("_ok").sum().over("s1_idx").alias("_nok"),
        pl.len().over("s1_idx").alias("_n"))
    d = d.filter(pl.col("_k") <= max_n).with_columns(pl.len().over("s1_idx").alias("_n"))
    picks = []
    for (n,), g in d.group_by("_n"):
        g = g.sort(["s1_idx", "_k"])
        ids = g["s1_idx"].to_numpy().reshape(-1, n)[:, 0]
        P = np.clip(g["p"].to_numpy().reshape(-1, n), 0.0, 1.0)
        nok = g["_nok"].to_numpy().reshape(-1, n)[:, 0]
        ef = expected_f_exact(P, m0=prm.m0)
        ef[:, 0] *= prm.empty_bias
        ks = np.arange(n + 1)[None, :]
        ef[ks > nok[:, None]] = -1.0                      # only choosable candidates can be picked
        kbest = ef.argmax(1)
        picks.append(pl.DataFrame({"s1_idx": ids, "_kbest": kbest}))
    best = pl.concat(picks) if picks else pl.DataFrame({"s1_idx": [], "_kbest": []})
    d = d.join(best, on="s1_idx").filter((pl.col("_k") <= pl.col("_kbest")) & pl.col("_ok")
                                         & (pl.col("p") >= prm.floor))
    return d.select("s1_idx", "src", "t_idx")
