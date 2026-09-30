"""Evaluate dense-retrieval variants with Zayaan's final2 code (er.dense merge -> er.run tune -> er.compare).

Inputs:  rep/dense_out   (account 1, K=10 keep 95%: faithful v6cd replica)   -> run v6cd
         rep3/dense_out  (account 3, K=20 keep 99%: superset)                 -> runs v6cd_k{K}_{keep}
A variant keeps pairs with rank <= K and cos >= tau, tau = (1-keep) quantile of cos over val TRUE pairs with rank <= K
(the same rule dense.py uses). Every run is gated against v6ce and against the replica v6cd.
Usage: python dense_variants.py [replica|variants|all]      Results: out/dense_variants.json (appended per run)"""
import json
import os
import shutil
import subprocess
import sys

import polars as pl

R = "E:/projects/Amazon ML challenge"
WT = R + "/.worktrees"
WORK = WT + "/v6val/work_v6"
SRC = WT + "/final2/code/business_entity_resolution/src"
OUTJ = WT + "/v6val/out/dense_variants.json"
LOGF = WT + "/v6val/out/dense_variants.log"
env = dict(os.environ, ER_WORK=WORK, ER_DATA=R + "/student_resource/dataset", ER_OUT=WT + "/rep/output",
           POLARS_MAX_THREADS="6", ER_WORKERS="6", PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONPATH=SRC)
VARIANTS = [(10, 0.95), (10, 0.99), (15, 0.97), (20, 0.95), (20, 0.97), (20, 0.99)]
what = sys.argv[1] if len(sys.argv) > 1 else "all"


def er(*a):
    with open(LOGF, "a", encoding="utf-8") as lf:
        lf.write(">> " + " ".join(a) + "\n")
        lf.flush()
        r = subprocess.run([sys.executable, "-m", *a], cwd=SRC, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
        lf.write(r.stdout[-4000:] + r.stderr[-4000:] + "\n")
    if r.returncode:
        raise SystemExit(f"FAILED {a}: rc {r.returncode}")
    return r.stdout


def results():
    return json.load(open(OUTJ)) if os.path.exists(OUTJ) else {}


def save(key, val):
    d = results()
    d[key] = val
    json.dump(d, open(OUTJ, "w"), indent=1)


def build(run, dense_dir, mode=None, ce2_dir=None):
    """merge (optionally with extra stacker features: lex / ce2 / lex+ce2) -> tune -> decision; gates vs v6ce + replica."""
    rd = f"{WORK}/runs/{run}"
    if not os.path.exists(f"{rd}/decision_ce.json"):
        if os.path.exists(rd):
            shutil.rmtree(rd)
        if mode:
            args = [sys.executable, WT + "/v6val/tmp/merge_extra.py", dense_dir, f"{dense_dir}_{mode.replace('+', '_')}", run, mode]
            r = subprocess.run(args + ([ce2_dir] if ce2_dir else []),
                               env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
            if r.returncode:
                raise SystemExit(f"FAILED merge_extra: {r.stderr[-2000:]}")
            info = r.stdout
        else:
            info = er("er.dense", "merge", "--run", "v6ce", "--dense", dense_dir, "--out-run", run)
        er("er.run", "tune", "--run", run, "--scores", "ce")
        shutil.copyfile(f"{rd}/decision_ce.json", f"{rd}/decision.json")
        save(f"{run}/merge", info.strip()[-600:])
    dec = json.load(open(f"{rd}/decision_ce.json"))
    save(f"{run}/decision", {k: dec[k] for k in ("best", "by_country") if k in dec})
    save(f"{run}/gate_vs_v6ce", json.loads(er("er.compare", "--a", f"{WORK}/runs/v6ce", "--b", rd, "--work-a", WORK, "--work-b", WORK)))
    if run != "v6cd" and os.path.exists(f"{WORK}/runs/v6cd/decision_ce.json"):
        save(f"{run}/gate_vs_v6cd", json.loads(er("er.compare", "--a", f"{WORK}/runs/v6cd", "--b", rd, "--work-a", WORK, "--work-b", WORK)))
    print(run, "f05", dec["best"].get("f05"), dec.get("by_country"), flush=True)


def carve(k, keep):
    src, dst = f"{WT}/rep3/dense_out", f"{WT}/rep3/var_k{k}_{int(keep * 100)}"
    if os.path.exists(f"{dst}/dense_test.parquet"):
        return dst
    os.makedirs(dst, exist_ok=True)
    v = pl.read_parquet(f"{src}/dense_val.parquet").filter(pl.col("rank") <= k)
    tau = float(v.filter(pl.col("y") == 1)["cos"].quantile(1 - keep))
    stat = {"k": k, "keep": keep, "tau": tau}
    for part in ("fit1", "val", "test"):
        d = pl.read_parquet(f"{src}/dense_{part}.parquet").filter((pl.col("rank") <= k) & (pl.col("cos") >= tau))
        d.write_parquet(f"{dst}/dense_{part}.parquet")
        stat[part] = d.height
    save(f"carve_k{k}_{int(keep * 100)}", stat)
    return dst


if what in ("replica", "all") and os.path.exists(f"{WT}/rep/dense_out/dense_test.parquet"):
    build("v6cd", f"{WT}/rep/dense_out")
if what in ("variants", "all") and os.path.exists(f"{WT}/rep3/dense_out/dense_test.parquet"):
    for k, keep in VARIANTS:
        build(f"v6cd_k{k}_{int(keep * 100)}", carve(k, keep))
if what.startswith("extra:"):   # extra:<k>:<keep>:<lex|ce2|lex+ce2>[:<ce2_dir>]  -> a variant + extra stacker features
    parts = what.split(":", 4)
    k, keep, mode = int(parts[1]), float(parts[2]), parts[3]
    ce2_dir = parts[4] if len(parts) > 4 else None
    src = f"{WT}/rep/dense_out" if (k, keep) == (10, 0.95) and not os.path.exists(f"{WT}/rep3/dense_out/dense_test.parquet") else carve(k, keep)
    build(f"v6cd_k{k}_{int(keep * 100)}_{mode.replace('+', '_')}", src, mode, ce2_dir)
print("DONE", flush=True)
