"""LB progression analysis: what changed between leaderboard-scored submissions, and what the LB rewarded.

For consecutive submissions A -> B (ordered by LB), per country: pairs added (B \\ A) and removed (A \\ B), each
profiled by name similarity, address similarity and house-number status (token_set_ratio on the normalised test
fields). A rising LB means B's changes were net-correct, so the profile of added pairs describes what the LB
rewards and the profile of removed pairs what it punished. An optional candidate file is profiled against the
best file the same way, to see whether it moves in the rewarded direction.
Usage: python lb_progression.py [candidate_file]"""
import io
import json
import os
import sys
import zipfile

import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein

R = "E:/projects/Amazon ML challenge"
FILES = [("LB0.965_ours_val0.9762", R + "/matching_results_0.9762.tsv"),
         ("LB0.982_v6ce", R + "/matching_results_0.982_on_unstop"),
         ("LB0.986_final2", R + "/matching_results_0.986_on_unstop")]
if len(sys.argv) > 1:
    FILES.append(("CANDIDATE", sys.argv[1]))
W = R + "/.worktrees/v6val/work_v6"
OUT = R + "/.worktrees/v6val/out/lb_progression.json"


def load(path):
    if os.path.isdir(path):
        path = os.path.join(path, "matching_results.tsv")
    if zipfile.is_zipfile(path):
        z = zipfile.ZipFile(path)
        data = z.read(next(n for n in z.namelist() if n.endswith("matching_results.tsv")))
    else:
        data = open(path, "rb").read()
    d = pl.read_csv(io.BytesIO(data), separator="\t", quote_char=None, infer_schema=False)
    d = d.rename({d.columns[0]: "s1", d.columns[1]: "m"})
    return (d.with_columns(pl.col("m").fill_null("").str.split(",")).explode("m")
             .filter(pl.col("m") != "").select("s1", pl.col("m").alias("t")))


C = ["entity_id", "name_norm", "addr_norm", "hno"]
s1 = pl.read_parquet(f"{W}/test_s1.parquet", columns=C + ["country"]).rename(
    {"entity_id": "s1", "name_norm": "n1", "addr_norm": "a1", "hno": "h1"})
tg = pl.concat([pl.read_parquet(f"{W}/test_s{s}.parquet", columns=C) for s in (2, 3)]).rename(
    {"entity_id": "t", "name_norm": "n2", "addr_norm": "a2", "hno": "h2"})


def profile(pairs):
    """pairs: s1, t -> per-country profile of similarity buckets (shares) + counts."""
    if pairs.height == 0:
        return {}
    x = (pairs.join(s1, on="s1", how="left").join(tg, on="t", how="left")
              .with_columns(pl.col("n1", "n2", "a1", "a2", "h1", "h2").fill_null("").str.strip_chars()))
    x = x.with_columns(
        pl.Series("sn", process.cpdist(x["n1"].to_list(), x["n2"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)),
        pl.Series("sa", process.cpdist(x["a1"].to_list(), x["a2"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)),
        pl.Series("ed", process.cpdist(x["h1"].to_list(), x["h2"].to_list(), scorer=Levenshtein.distance, workers=-1)))
    x = x.with_columns(
        pl.when(pl.col("sn") >= 90).then(pl.lit("name>=90")).when(pl.col("sn") >= 70).then(pl.lit("name70-90")).otherwise(pl.lit("name<70")).alias("nm"),
        pl.when((pl.col("a1") == "") | (pl.col("a2") == "")).then(pl.lit("addr_null")).when(pl.col("sa") >= 80).then(pl.lit("addr>=80"))
          .when(pl.col("sa") >= 50).then(pl.lit("addr50-80")).otherwise(pl.lit("addr<50")).alias("ad"),
        pl.when((pl.col("h1") == "") | (pl.col("h2") == "")).then(pl.lit("hno_missing")).when(pl.col("ed") == 0).then(pl.lit("hno_same"))
          .when(pl.col("ed") == 1).then(pl.lit("hno_1edit")).otherwise(pl.lit("hno_diff")).alias("hn"))
    res = {}
    for (c,), g in x.group_by("country"):
        n = g.height
        res[c] = {"n": n, **{f"{col}:{k}": round(v / n, 4) for col in ("nm", "ad", "hn")
                            for k, v in g.group_by(col).len().rows()}}
    return res


def structure(pairs):
    cnt = s1.select("s1", "country").join(pairs.group_by("s1").len("k"), on="s1", how="left").with_columns(pl.col("k").fill_null(0))
    return {c: {"matches_per_s1": round(float(g["k"].mean()), 4), "empty": round(float((g["k"] == 0).mean()), 4)}
            for (c,), g in cnt.group_by("country")}


P = {name: load(path) for name, path in FILES}
rep = {"structure": {n: structure(p) for n, p in P.items()}, "steps": {}}
names = [n for n, _ in FILES]
steps = list(zip(names[:3], names[1:3]))
if "CANDIDATE" in P:
    steps.append(("LB0.986_final2", "CANDIDATE"))
for a, b in steps:
    add = P[b].join(P[a], on=["s1", "t"], how="anti")
    rem = P[a].join(P[b], on=["s1", "t"], how="anti")
    rep["steps"][f"{a} -> {b}"] = {"added": profile(add), "removed": profile(rem)}
    print(f"{a} -> {b}: added {add.height}, removed {rem.height}", flush=True)
json.dump(rep, open(OUT, "w"), indent=1)

# compact printout: per step x country, the dominant patterns of added vs removed pairs
for step, v in rep["steps"].items():
    print("\n==", step)
    for kind in ("added", "removed"):
        for c, prof in sorted(v[kind].items()):
            top = sorted(((k, s) for k, s in prof.items() if k != "n"), key=lambda z: -z[1])
            print(f"  {kind:7s} {c:7s} n={prof['n']:7d} | " + ", ".join(f"{k}={s:.2f}" for k, s in top if s >= 0.05))
print("\nstructure:", json.dumps(rep["structure"], indent=0))
