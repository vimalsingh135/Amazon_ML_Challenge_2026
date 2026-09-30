"""Kaggle CPU kernel: dense stacker + LEXICAL features (US/India lever).
Diagnosis (our v6cd val): dense retrieval found 7,995 true val pairs but the dense stacker (cos, rank, gap1, ce, lce, src)
accepted only 4,337; the missed ones get p ~0.02-0.08. Here the dense stacker also sees name / address token_set
similarity, house-number status (same / 1-edit / one missing / both missing / different), name length ratio and a
missing-address flag, computed from the dense export texts ("name | address"). Trained on FIT fold-1 dense pairs only
(Zayaan's er.dense.merge, unchanged except DENSE_FEATS), tuned, gated vs our v6cd replica (er.compare + held-out halves).
If accepted: test predictions (US/India) zipped."""
import glob
import json
import os
import re
import shutil
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rapidfuzz"], check=False)
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from rapidfuzz import fuzz, process  # noqa: E402
from rapidfuzz.distance import Levenshtein  # noqa: E402

g = lambda p: sorted(glob.glob(f"/kaggle/input/**/{p}", recursive=True))
E = os.path.dirname(g("eval_manifest.json")[0])
TX = os.path.dirname(g("train_roles.parquet")[-1]) if False else os.path.dirname([p for p in g("train_s2.parquet") if "eval" not in p][0])
DENSE = os.path.dirname(g("dense_val.parquet")[0])
DATA = os.path.dirname(os.path.dirname(g("train_ground_truth.tsv")[0]))
W, OUT = "/kaggle/working/work", "/kaggle/working"
for d in ("runs/v6ce", "runs/v6cd"):
    os.makedirs(f"{W}/{d}", exist_ok=True)
for f in os.listdir(f"{E}/work"):
    shutil.copyfile(f"{E}/work/{f}", f"{W}/{f}")
for run in ("v6ce", "v6cd"):
    for f in os.listdir(f"{E}/runs_{run}"):
        shutil.copyfile(f"{E}/runs_{run}/{f}", f"{W}/runs/{run}/{f}")
env = dict(os.environ, ER_WORK=W, ER_DATA=DATA, ER_OUT=f"{OUT}/sub", POLARS_MAX_THREADS="4", ER_WORKERS="4", PYTHONPATH=f"{E}/code")
os.environ.update(ER_WORK=W, ER_DATA=DATA)
sys.path.insert(0, f"{E}/code")
K = ["s1_idx", "src", "t_idx"]
HNO = re.compile(r"\b(\d+[a-z]?)\b")


def texts(split, s):
    d = pl.read_parquet(f"{TX}/{split}_s{s}.parquet", columns=["idx", "text"])
    parts = d["text"].str.split_exact(" | ", 1)
    return d.select(pl.col("idx").cast(pl.UInt32)).with_columns(
        parts.struct.field("field_0").fill_null("").str.to_lowercase().alias("n"),
        parts.struct.field("field_1").fill_null("").str.to_lowercase().alias("a"))


def lex(d, split):
    s1 = texts(split, 1).rename({"idx": "s1_idx", "n": "n1", "a": "a1"})
    tg = pl.concat([texts(split, s).rename({"idx": "t_idx", "n": "n2", "a": "a2"}).with_columns(pl.lit(s, pl.UInt8).alias("src")) for s in (2, 3)])
    x = d.select(K).join(s1, on="s1_idx", how="left").join(tg, on=["src", "t_idx"], how="left").with_columns(pl.col("n1", "n2", "a1", "a2").fill_null(""))
    x = x.with_columns(pl.col("a1").str.extract(r"\b(\d+[a-z]?)\b").fill_null("").alias("h1"),
                       pl.col("a2").str.extract(r"\b(\d+[a-z]?)\b").fill_null("").alias("h2"))
    sn = process.cpdist(x["n1"].to_list(), x["n2"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)
    sa = process.cpdist(x["a1"].to_list(), x["a2"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)
    ed = process.cpdist(x["h1"].to_list(), x["h2"].to_list(), scorer=Levenshtein.distance, workers=-1)
    x = x.with_columns(pl.Series("lx_name", sn), pl.Series("lx_addr", sa), pl.Series("_ed", ed)).with_columns(
        pl.when((pl.col("h1") == "") & (pl.col("h2") == "")).then(0).when((pl.col("h1") == "") | (pl.col("h2") == "")).then(1)
          .when(pl.col("_ed") == 0).then(2).when(pl.col("_ed") == 1).then(3).otherwise(4).cast(pl.Float32).alias("lx_hno"),
        (pl.min_horizontal(pl.col("n1").str.len_chars(), pl.col("n2").str.len_chars())
         / pl.max_horizontal(pl.col("n1").str.len_chars(), pl.col("n2").str.len_chars(), pl.lit(1))).cast(pl.Float32).alias("lx_lenr"),
        (pl.col("a1").eq("") | pl.col("a2").eq("")).cast(pl.Float32).alias("lx_anull"))
    return d.join(x.select(K + LEX), on=K, how="left")


LEX = ["lx_name", "lx_addr", "lx_hno", "lx_lenr", "lx_anull"]
cast = lambda d: d.with_columns(pl.col("s1_idx").cast(pl.UInt32), pl.col("src").cast(pl.UInt8), pl.col("t_idx").cast(pl.UInt32))
LD = "/kaggle/working/dense_lex"
os.makedirs(LD, exist_ok=True)
for part in ("fit1", "val", "test"):
    d = lex(cast(pl.read_parquet(f"{DENSE}/dense_{part}.parquet")), "test" if part == "test" else "train")
    d.write_parquet(f"{LD}/dense_{part}.parquet")
    print("lex", part, d.height, flush=True)

from er import dense  # noqa: E402

dense.DENSE_FEATS = dense.DENSE_FEATS + LEX
print("merge", dense.merge("v6ce", LD, "v6cd_lex"), flush=True)


def er(*a):
    r = subprocess.run([sys.executable, "-m", *a], env=env, capture_output=True, text=True)
    print(">>", " ".join(a), "\n", r.stdout[-1500:], r.stderr[-1500:], flush=True)
    if r.returncode:
        raise RuntimeError(f"{a} failed")
    return r.stdout


er("er.run", "tune", "--run", "v6cd_lex", "--scores", "ce")
shutil.copyfile(f"{W}/runs/v6cd_lex/decision_ce.json", f"{W}/runs/v6cd_lex/decision.json")
res = {"tuned": json.load(open(f"{W}/runs/v6cd_lex/decision_ce.json"))}
res["gate_vs_v6cd"] = json.loads(er("er.compare", "--a", f"{W}/runs/v6cd", "--b", f"{W}/runs/v6cd_lex", "--work-a", W, "--work-b", W))

# held-out halves (decision tuned on one half of val S1, scored on the other)
from er.decide import DecisionParams, select  # noqa: E402
from er.pipeline import id_maps, train_roles  # noqa: E402
from er.prep import load_gt  # noqa: E402

roles = train_roles().filter(pl.col("role") == "val").select(pl.col("idx").alias("s1_idx"), "country")
roles = roles.with_columns(((pl.col("s1_idx") * 2654435761) % 1000 < 500).cast(pl.Int8).alias("half"))
s1m, tg = id_maps("train")
truth = load_gt().join(s1m.select("s1_idx", "s1"), on="s1").join(tg, on="tid").select(K).join(roles.select("s1_idx"), on="s1_idx")
nt = truth.group_by("s1_idx").len("nt")
GRID = [DecisionParams(margin=mg, m0=m0, empty_bias=eb, coh=1.0) for mg in (0.0, 0.05, 0.1) for m0 in (0.0, 0.25, 0.5, 0.75) for eb in (1.5, 2.0, 2.5, 3.0)]


def per_s1(sel, s1s):
    tp = sel.join(truth, on=K).group_by("s1_idx").len("tp")
    x = (s1s.join(nt, on="s1_idx", how="left").join(sel.group_by("s1_idx").len("np"), on="s1_idx", how="left").join(tp, on="s1_idx", how="left").fill_null(0))
    f = pl.when((pl.col("nt") == 0) & (pl.col("np") == 0)).then(1.0).otherwise(1.25 * pl.col("tp") / (0.25 * pl.col("nt") + pl.col("np")).clip(lower_bound=1e-9))
    return x.with_columns(f.alias("f")).select("s1_idx", "country", "f")


def heldout(run):
    v = pl.read_parquet(f"{W}/runs/{run}/val_scores_ce.parquet").select(K + ["p"]).join(roles.select("s1_idx", "half"), on="s1_idx")
    parts = []
    for h in (0, 1):
        tr, te = v.filter(pl.col("half") != h).select(K + ["p"]), v.filter(pl.col("half") == h).select(K + ["p"])
        s_tr, s_te = roles.filter(pl.col("half") != h), roles.filter(pl.col("half") == h)
        best = max(GRID, key=lambda gp: float(per_s1(select(tr, gp), s_tr)["f"].mean()))
        parts.append(per_s1(select(te, best), s_te))
    return pl.concat(parts).sort("s1_idx")


fa, fb = heldout("v6cd"), heldout("v6cd_lex")
dd = fa.join(fb.select("s1_idx", pl.col("f").alias("fb")), on="s1_idx").with_columns((pl.col("fb") - pl.col("f")).alias("d"))
x = dd["d"].to_numpy()
rng = np.random.default_rng(0)
bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(1000)]
h = {"a": float(dd["f"].mean()), "b": float(dd["fb"].mean()), "delta": float(x.mean()),
     "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "by_country": {c: float(q["d"].mean()) for (c,), q in dd.group_by("country")}}
h["accept"] = h["ci95"][0] > 0 and all(v >= 0 for v in h["by_country"].values())
res["heldout_vs_v6cd"] = h
json.dump(res, open(f"{OUT}/lex_results.json", "w"), indent=1)
print("HELDOUT", json.dumps(h), flush=True)
if h["accept"]:
    env["ER_SUB_NAME"] = "v6cd_lex"
    er("er.run", "predict", "--run", "v6cd_lex", "--scores", "ce")
    import zipfile
    for f in ("matching_results", "candidate_pairs"):
        with zipfile.ZipFile(f"{OUT}/v6cd_lex_{f}.zip", "w", zipfile.ZIP_DEFLATED) as zz:
            zz.write(f"{OUT}/sub/v6cd_lex/{f}.tsv", f"{f}.tsv")
shutil.rmtree(f"{OUT}/sub", ignore_errors=True)
shutil.rmtree(W, ignore_errors=True)
shutil.rmtree(LD, ignore_errors=True)
print("DONE", flush=True)
