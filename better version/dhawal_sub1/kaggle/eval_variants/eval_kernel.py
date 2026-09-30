"""Kaggle CPU kernel: evaluate dense-retrieval variants carved from a superset dense run (K=20/K=50, keep 99%) with
Zayaan's code (er.dense merge -> er.run tune -> er.compare), gated vs v6ce and vs our v6cd replica, plus a held-out
half check (decision tuned on one half of val S1, scored on the other). Best accepted variant also gets test predictions.
Env (in-script): VARIANTS list. Inputs: almc-eval-data dataset + the dense run's kernel output (dense_{fit1,val,test})."""
import glob
import json
import os
import shutil
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rapidfuzz"], check=False)
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

g = lambda p: sorted(glob.glob(f"/kaggle/input/**/{p}", recursive=True))
E = os.path.dirname(g("eval_manifest.json")[0])
DENSE = os.path.dirname(g("dense_val.parquet")[0])
DATA = os.path.dirname(os.path.dirname(g("train_ground_truth.tsv")[0]))
VARIANTS = json.load(open(f"{E}/eval_manifest.json"))["variants"]
W, OUT = "/kaggle/working/work", "/kaggle/working"
for d in ("runs/v6ce", "runs/v6cd"):
    os.makedirs(f"{W}/{d}", exist_ok=True)
for f in os.listdir(f"{E}/work"):
    if f.endswith(".parquet"):
        shutil.copyfile(f"{E}/work/{f}", f"{W}/{f}")
for run in ("v6ce", "v6cd"):
    for f in os.listdir(f"{E}/runs_{run}"):
        shutil.copyfile(f"{E}/runs_{run}/{f}", f"{W}/runs/{run}/{f}")
env = dict(os.environ, ER_WORK=W, ER_DATA=DATA, ER_OUT=f"{OUT}/sub", POLARS_MAX_THREADS="4", ER_WORKERS="4",
           PYTHONPATH=f"{E}/code")
sys.path.insert(0, f"{E}/code")
os.environ.update(ER_WORK=W, ER_DATA=DATA)
res = {}


def er(*a):
    r = subprocess.run([sys.executable, "-m", *a], env=env, capture_output=True, text=True)
    print(">>", " ".join(a), "\n", r.stdout[-1500:], r.stderr[-1500:], flush=True)
    if r.returncode:
        raise RuntimeError(f"{a} failed")
    return r.stdout


def carve(k, keep):
    dst = f"/kaggle/working/var_k{k}_{int(keep * 100)}"
    os.makedirs(dst, exist_ok=True)
    v = pl.read_parquet(f"{DENSE}/dense_val.parquet").filter(pl.col("rank") <= k)
    tau = float(v.filter(pl.col("y") == 1)["cos"].quantile(1 - keep))
    st = {"tau": tau}
    for part in ("fit1", "val", "test"):
        d = pl.read_parquet(f"{DENSE}/dense_{part}.parquet").filter((pl.col("rank") <= k) & (pl.col("cos") >= tau))
        d.write_parquet(f"{dst}/dense_{part}.parquet")
        st[part] = d.height
    return dst, st


# held-out half gate (tune decision on one half of val S1, score the other), same metric as er.compare
from er.decide import DecisionParams, select  # noqa: E402
from er.pipeline import id_maps, train_roles  # noqa: E402
from er.prep import load_gt  # noqa: E402

K = ["s1_idx", "src", "t_idx"]
roles = train_roles().filter(pl.col("role") == "val").select(pl.col("idx").alias("s1_idx"), "country")
roles = roles.with_columns(((pl.col("s1_idx") * 2654435761) % 1000 < 500).cast(pl.Int8).alias("half"))
s1m, tg = id_maps("train")
truth = load_gt().join(s1m.select("s1_idx", "s1"), on="s1").join(tg, on="tid").select(K).join(roles.select("s1_idx"), on="s1_idx")
nt = truth.group_by("s1_idx").len("nt")
GRID = [DecisionParams(margin=mg, m0=m0, empty_bias=eb, coh=1.0) for mg in (0.0, 0.05, 0.1) for m0 in (0.0, 0.25, 0.5, 0.75)
        for eb in (1.5, 2.0, 2.5, 3.0)]


def per_s1(sel, s1s):
    tp = sel.join(truth, on=K).group_by("s1_idx").len("tp")
    x = (s1s.join(nt, on="s1_idx", how="left").join(sel.group_by("s1_idx").len("np"), on="s1_idx", how="left")
            .join(tp, on="s1_idx", how="left").fill_null(0))
    f = pl.when((pl.col("nt") == 0) & (pl.col("np") == 0)).then(1.0).otherwise(
        1.25 * pl.col("tp") / (0.25 * pl.col("nt") + pl.col("np")).clip(lower_bound=1e-9))
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


def honest(a, b):
    fa, fb = heldout(a), heldout(b)
    dd = fa.join(fb.select("s1_idx", pl.col("f").alias("fb")), on="s1_idx").with_columns((pl.col("fb") - pl.col("f")).alias("d"))
    x = dd["d"].to_numpy()
    rng = np.random.default_rng(0)
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(1000)]
    out = {"a": float(dd["f"].mean()), "b": float(dd["fb"].mean()), "delta": float(x.mean()),
           "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
           "by_country": {c: float(q["d"].mean()) for (c,), q in dd.group_by("country")}}
    out["accept"] = out["ci95"][0] > 0 and all(v >= 0 for v in out["by_country"].values())
    return out


best_run, best_delta = None, 0.0
for k, keep in VARIANTS:
    run = f"var_k{k}_{int(keep * 100)}"
    try:
        dst, st = carve(k, keep)
        info = er("er.dense", "merge", "--run", "v6ce", "--dense", dst, "--out-run", run)
        er("er.run", "tune", "--run", run, "--scores", "ce")
        shutil.copyfile(f"{W}/runs/{run}/decision_ce.json", f"{W}/runs/{run}/decision.json")
        dec = json.load(open(f"{W}/runs/{run}/decision_ce.json"))
        r = {"carve": st, "merge": info.strip()[-400:], "val_tuned": {k_: dec[k_] for k_ in ("best", "by_country") if k_ in dec}}
        r["gate_vs_v6cd"] = json.loads(er("er.compare", "--a", f"{W}/runs/v6cd", "--b", f"{W}/runs/{run}", "--work-a", W, "--work-b", W))
        r["heldout_vs_v6cd"] = honest("v6cd", run)
        res[run] = r
        print(run, json.dumps(r["heldout_vs_v6cd"]), flush=True)
        if r["heldout_vs_v6cd"]["accept"] and r["heldout_vs_v6cd"]["delta"] > best_delta:
            best_run, best_delta = run, r["heldout_vs_v6cd"]["delta"]
    except Exception as e:  # keep going with the other variants
        res[run] = {"error": repr(e)}
    json.dump(res, open(f"{OUT}/eval_results.json", "w"), indent=1)
    for d in glob.glob("/kaggle/working/var_k*"):
        shutil.rmtree(d, ignore_errors=True)
res["best"] = {"run": best_run, "heldout_delta": best_delta}
json.dump(res, open(f"{OUT}/eval_results.json", "w"), indent=1)
if best_run:   # test predictions of the winner (matching/candidates), zipped
    env["ER_SUB_NAME"] = best_run
    er("er.run", "predict", "--run", best_run, "--scores", "ce")
    import zipfile
    for f in ("matching_results", "candidate_pairs"):
        with zipfile.ZipFile(f"{OUT}/{best_run}_{f}.zip", "w", zipfile.ZIP_DEFLATED) as zz:
            zz.write(f"{OUT}/sub/{best_run}/{f}.tsv", f"{f}.tsv")
    shutil.rmtree(f"{OUT}/sub", ignore_errors=True)
for run in list(os.listdir(f"{W}/runs")):
    if run.startswith("var_"):
        shutil.rmtree(f"{W}/runs/{run}", ignore_errors=True)
shutil.rmtree(W, ignore_errors=True)
print("DONE", json.dumps(res["best"]), flush=True)
