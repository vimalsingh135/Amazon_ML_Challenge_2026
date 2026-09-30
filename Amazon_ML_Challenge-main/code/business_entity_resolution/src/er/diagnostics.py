"""Layer-level diagnostics: blocking recall/volume, per-family contribution, miss inspection.

python -m er.diagnostics blocking --n 40000
"""
from __future__ import annotations

import argparse
import time

import polars as pl

from .blocking import block
from .pipeline import load
from .prep import load_gt

SHOW = ["entity_id", "business_name", "business_address"]


def blocking_report(n: int = 40000, seed: int = 1, show_miss: int = 6, **block_kw) -> pl.DataFrame:
    gt = load_gt()
    s1_all = load("train", "1").sample(n, seed=seed)
    rows = []
    for country in s1_all["country"].unique().sort().to_list():
        a = s1_all.filter(pl.col("country") == country)
        ids1 = a.select(pl.col("idx").alias("s1_idx"), pl.col("entity_id").alias("s1"))
        cands, tgts = [], []
        for src in ("2", "3"):
            t = load("train", src, country)
            t0 = time.time()
            r = block(a, t, **block_kw)
            dt = time.time() - t0
            cands.append(r.join(ids1, on="s1_idx").join(
                t.select(pl.col("idx").alias("t_idx"), pl.col("entity_id").alias("tid")), on="t_idx"))
            tgts.append(t.select(SHOW))
            print(f"{country} S{src}: {dt:.0f}s, {r.height / a.height:.1f} cands/S1")
        c = pl.concat(cands)
        m = load_gt_subset(gt, a).join(c, on=["s1", "tid"], how="left")
        hit = pl.col("bs").is_not_null()
        rec = {"country": country, "n_s1": a.height, "cands_per_s1": c.height / a.height,
               "recall": m.select(hit.mean()).item(),
               "recall_top40": m.select((pl.col("brank") <= 40).fill_null(False).mean()).item(),
               "recall_top20": m.select((pl.col("brank") <= 20).fill_null(False).mean()).item(),
               "recall_top10": m.select((pl.col("brank") <= 10).fill_null(False).mean()).item(),
               "only_name_budget": m.select((hit & (pl.col("brank") > 40) & (pl.col("brank_n") <= 15)).mean()).item(),
               "only_addr_budget": m.select((hit & (pl.col("brank") > 40) & (pl.col("brank_a") <= 15)).mean()).item()}
        rows.append(rec)
        if show_miss:
            tg = pl.concat(tgts).rename({"entity_id": "tid", "business_name": "n2", "business_address": "a2"})
            miss = (m.filter(~hit).head(show_miss).join(tg, on="tid")
                     .join(a.select(pl.col("entity_id").alias("s1"), pl.col("business_name").alias("n1"),
                                    pl.col("business_address").alias("a1")), on="s1"))
            for q in miss.iter_rows(named=True):
                print(f"  MISS {q['n1']} | {q['a1']}  <>  {q['n2']} | {q['a2']}")
    out = pl.DataFrame(rows)
    print(out)
    return out


def miss_anatomy(country: str = "India", src: str = "2", n: int = 5000, seed: int = 3,
                 cap_single: int = 300, cap_pair: int = 1500) -> pl.DataFrame:
    """Why does blocking miss true pairs? Classify every missed pair:
    no_shared_key | only_capped_keys (all shared keys too frequent) | cut_by_budget (scored, ranked out).
    Also reports the rank the pair would have had without budgets and its per-family shared keys."""
    from .blocking import FAM_ID, SINGLE, block, make_keys
    inv = {v: k for k, v in FAM_ID.items()}
    gt = load_gt()
    a = load("train", "1", country).sample(n, seed=seed)
    t = load("train", src, country)
    ids_t = t.select(pl.col("idx").alias("t_idx"), pl.col("entity_id").alias("tid"))
    ids_a = a.select(pl.col("idx").alias("s1_idx"), pl.col("entity_id").alias("s1"))
    truth = gt.join(ids_a, on="s1").join(ids_t, on="tid")
    got = block(a, t, cap_single=cap_single, cap_pair=cap_pair).select("s1_idx", "t_idx", pl.lit(1).alias("hit"))
    miss = truth.join(got, on=["s1_idx", "t_idx"], how="left").filter(pl.col("hit").is_null())
    tk = make_keys(t)
    dfk = tk.group_by("key", "fam").len("df").with_columns(
        (pl.col("df") <= pl.when(pl.col("fam").is_in([FAM_ID[f] for f in SINGLE])).then(cap_single)
         .otherwise(cap_pair)).alias("ok"))
    ka = make_keys(a.join(miss.select(pl.col("s1_idx").alias("idx")).unique(), on="idx"))
    kt = tk.join(miss.select(pl.col("t_idx").alias("idx")).unique(), on="idx")
    sh = (miss.select("s1_idx", "t_idx")
          .join(ka.rename({"idx": "s1_idx"}), on="s1_idx")
          .join(kt.rename({"idx": "t_idx"}), on=["t_idx", "key", "fam"])
          .join(dfk, on=["key", "fam"]))
    per = sh.group_by("s1_idx", "t_idx").agg(
        pl.col("ok").sum().alias("n_ok"), pl.len().alias("n_shared"),
        pl.col("fam").filter(pl.col("ok")).alias("fams_ok"))
    m = miss.join(per, on=["s1_idx", "t_idx"], how="left").with_columns(
        pl.when(pl.col("n_shared").is_null()).then(pl.lit("no_shared_key"))
          .when(pl.col("n_ok") == 0).then(pl.lit("only_capped_keys"))
          .otherwise(pl.lit("cut_by_budget")).alias("why"))
    print(f"{country} S{src}: true={truth.height} missed={miss.height} "
          f"recall={1 - miss.height / max(truth.height, 1):.4f}")
    print(m.group_by("why").len().sort("len", descending=True))
    fam_counts = (m.filter(pl.col("why") == "cut_by_budget").explode("fams_ok")
                    .group_by("fams_ok").len().with_columns(pl.col("fams_ok").replace_strict(inv)))
    print("families present in budget-cut misses:", dict(fam_counts.iter_rows()))
    show = (m.join(a.select(pl.col("idx").alias("s1_idx"), pl.col("business_name").alias("n1"),
                             pl.col("business_address").alias("a1")), on="s1_idx")
              .join(t.select(pl.col("idx").alias("t_idx"), pl.col("business_name").alias("n2"),
                             pl.col("business_address").alias("a2")), on="t_idx"))
    for why in ("no_shared_key", "only_capped_keys"):
        for q in show.filter(pl.col("why") == why).head(4).iter_rows(named=True):
            print(f"  [{why}] {q['n1']} | {q['a1']}  <>  {q['n2']} | {q['a2']}")
    return m


def load_gt_subset(gt: pl.DataFrame, s1: pl.DataFrame) -> pl.DataFrame:
    return gt.join(s1.select(pl.col("entity_id").alias("s1")), on="s1")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["blocking"])
    ap.add_argument("--n", type=int, default=40000)
    args = ap.parse_args()
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(200)
    blocking_report(args.n)
