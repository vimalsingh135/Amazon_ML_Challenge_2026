"""Generate notebooks/kaggle_v6ce_ens.ipynb - ONE self-contained Kaggle cell that:

  0. clones Zayaan's public `v6ce` branch (code + the committed artifacts/v6ce/ no-GPU rebuild kit),
     preps the raw data and rebuilds his LB-0.982 submission (output/v6ce);
  1. exports the CE training pairs (runs/v6/fit_oof, FIT fold 0) and the uncertain bands;
  C. CANARY: re-stacks HIS cross-encoder scores through this environment -> must reproduce 0.98730;
  2. trains OUR cross-encoder ensemble ("personalities", ce_train_ens.py) and scores the bands;
  3. stacks OUR ensemble with his exact stacker recipe, tunes, and GATES it vs his (paired bootstrap on
     the same validation S1), then writes our test submission if the v6 test base scores are present.

Edit this script (not the .ipynb) and re-run it. Our CE scripts and two helpers are embedded as base64
so the notebook needs nothing but the public v6ce branch + the raw dataset.
"""
import base64
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
KG = os.path.join(HERE, "..", "scripts", "kaggle")


def b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


TRAIN = open(os.path.join(KG, "train", "ce_train_ens.py"), encoding="utf-8").read()
SCORE = open(os.path.join(KG, "score", "ce_score_ens.py"), encoding="utf-8").read()

# Uncertain-band export (er.cross.export_band) that tolerates a missing runs/<run>/test_scores.parquet
# (>100 MB, so not in git): fit1 + val always, test only if present.
BANDS = r'''import os, sys
import polars as pl
from er import config as C
from er.cross import _fit1_calibrated, _attach
run, out, lo, hi = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
os.makedirs(out, exist_ok=True)
rd = C.WORK / "runs" / run
band = (pl.col("p") >= lo) & (pl.col("p") <= hi)
keys = ["s1_idx", "src", "t_idx", "p"]
parts = [("fit1", lambda: _fit1_calibrated(rd), "train"),
         ("val", lambda: pl.read_parquet(rd / "val_scores.parquet"), "train")]
if (rd / "test_scores.parquet").exists():
    parts.append(("test", lambda: pl.read_parquet(rd / "test_scores.parquet"), "test"))
else:
    print("NOTE: runs/%s/test_scores.parquet absent -> test band skipped (val gate unaffected)" % run, flush=True)
for name, load, split in parts:
    p = "%s/ce_band_%s.parquet" % (out, name)
    if os.path.exists(p):
        print("band exists", name, flush=True)
        continue
    d = _attach(load().select(keys).filter(band), split)
    d.write_parquet(p)
    print("band", name, d.height, flush=True)
'''

# Faithful replica of er.cross.stack (same features, same LightGBM params, same seed) that only
# processes the splits whose inputs exist. Reads only the `ce` column (the ensemble mean), so the gate
# isolates the effect of swapping his single-CE score for our ensemble score.
STACK = r'''import os, sys
import lightgbm as lgb
import polars as pl
from er import config as C
from er.cross import KEYS, STACK_FEATS, _fit1_calibrated, _stack_feats
from er.pipeline import labels
run, ce_dir, out_run = sys.argv[1], sys.argv[2], sys.argv[3]
rd, od = C.WORK / "runs" / run, C.WORK / "runs" / out_run
od.mkdir(parents=True, exist_ok=True)
for f in ("summary.json", "calibrators.pkl", "stage2_f0.lgb", "stage2_f1.lgb"):
    if not (od / f).exists():
        os.link(rd / f, od / f)
cast = lambda d: d.with_columns(pl.col("s1_idx").cast(pl.UInt32), pl.col("src").cast(pl.UInt8),
                                pl.col("t_idx").cast(pl.UInt32))
ce = {}
for n in ("fit1", "val", "test"):
    p = "%s/ce_scores_%s.parquet" % (ce_dir, n)
    if os.path.exists(p):
        ce[n] = cast(pl.read_parquet(p).select(KEYS + ["ce"]))
lab = cast(labels()).with_columns(pl.lit(1, pl.Int8).alias("yy"))
tr = _stack_feats(cast(_fit1_calibrated(rd)).join(ce["fit1"], on=KEYS)).join(lab, on=KEYS, how="left")
y = tr["yy"].fill_null(0).to_numpy()
prm = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.9,
           bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=C.SEED, num_threads=C.WORKERS)
m = lgb.train(prm, lgb.Dataset(tr.select(STACK_FEATS).to_numpy(), y), 400)
m.save_model(str(od / "stacker.lgb"))
print("stacker trained on fit1 band:", tr.height, "rows,", int(y.sum()), "pos", flush=True)
for name, fname in (("val", "val_scores.parquet"), ("test", "test_scores.parquet")):
    if name not in ce or not (rd / fname).exists():
        print("stack: skip", name, "(ce scores or base scores missing)", flush=True)
        continue
    base = cast(pl.read_parquet(rd / fname).select(KEYS + ["p"]))
    bd = _stack_feats(base.join(ce[name], on=KEYS))
    bd = bd.with_columns(pl.Series("p2", m.predict(bd.select(STACK_FEATS).to_numpy()).astype("float32")))
    out = (base.join(bd.select(KEYS + ["p2"]), on=KEYS, how="left")
               .with_columns(pl.coalesce("p2", "p").alias("p")).drop("p2"))
    out.write_parquet(od / ("%s_scores_ce.parquet" % name))
    if name == "val":
        out.write_parquet(od / "val_scores.parquet")
    print("stacked", name, bd.height, "band rows", flush=True)
'''

CELL = r'''# =====================================================================================================
# AMAZON ML CHALLENGE 2026 - reproduce Zayaan's v6ce (LB 0.982 / val 0.98730) from his committed
# artifacts, then train OUR cross-encoder ENSEMBLE ("personalities") on top and GATE it vs his single CE.
#
#   Kaggle settings:  Accelerator = GPU T4 x2   |   Internet = ON
#   Add Input:        the RAW dataset  (satwiksps/amazon-ml-challenge-2026 : train/test *_source*.tsv + gt)
#                     + the CODE dataset (dhawal2209/almc-v6ce-src : v6ce code + artifacts/v6ce, no git)
#   Optional input:   runs/v6/test_scores.parquet (ask Zayaan; >100 MB so not in git) -> also writes OUR
#                     test submission. Without it the validation gate still runs.
#   Then:             Save Version -> "Save & Run All"  (background, survives disconnects) -> sleep.
#
# Results land in /kaggle/working: SUMMARY.txt, gate_*.json, output/v6ce (his 0.982), output/v6ce_ens
# (ours, if the test base is present), *_submission.zip, run_v6ce_ens.log.  Re-running skips finished steps.
# =====================================================================================================
CE_PRESET = "full"               # "full" = 5 CE members (~3-5 h GPU) | "fast" = 2 members | "smoke" = quick check
BAND_LO, BAND_HI = 0.005, 0.995  # his uncertain band (the CE only re-scores these pairs)

import os, sys, time, json, shutil, glob, base64, subprocess
T0 = time.time()
# (the V6_* env overrides exist only so the identical cell can be validated off-Kaggle; ignore them)
OUTW = os.environ.get("V6_OUT", "/kaggle/working")
def _scratch():
    """Large scratch dir kept OUT of /kaggle/working (so the committed output stays small)."""
    for d in ("/kaggle/temp/v6ce", "/tmp/v6ce"):
        try:
            os.makedirs(d, exist_ok=True)
            open(d + "/.w", "w").close()
            return d
        except OSError:
            pass
    return "/kaggle/working/v6ce_tmp"


ROOT = os.environ.get("V6_ROOT") or _scratch()
INPUT = os.environ.get("V6_INPUT", "/kaggle/input")
REPO = os.environ.get("V6_REPO", ROOT + "/repo")
W, DATA, S = ROOT + "/work_v6", ROOT + "/dataset", ROOT + "/scripts"
if os.path.exists(INPUT + "/train/train_source1.tsv"):   # input already in the dataset layout: use in place
    DATA = INPUT
LOG = OUTW + "/run_v6ce_ens.log"
for d in (ROOT, OUTW, S, DATA + "/train", DATA + "/test"):
    os.makedirs(d, exist_ok=True)
KEY = ("done", "SNAP", "SKIP", "stage", "predict", "tune", "band", "stack", "Error", "error", "Traceback",
       "loaded", "members", "=====", " ep ", "FAILED", "NOTE", "export", "gate", "delta")


def log(*a):
    s = "[%6.1fm] %s" % ((time.time() - T0) / 60, " ".join(str(x) for x in a))
    print(s, flush=True)
    with open(LOG, "a") as fh:
        fh.write(s + "\n")


def sh(cmd, env=None, cwd=None, check=True):
    """Run a command, streaming its output into the log (and key lines to the notebook)."""
    log("$", cmd if isinstance(cmd, str) else " ".join(str(c) for c in cmd))
    p = subprocess.Popen(cmd, shell=isinstance(cmd, str), env=env, cwd=cwd, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    buf = []
    with open(LOG, "a") as fh:
        for line in p.stdout:
            fh.write(line)
            buf = (buf + [line])[-300:]
            if any(k in line for k in KEY):
                print("    " + line.rstrip()[:220], flush=True)
    rc = p.wait()
    if rc != 0:
        log("FAILED rc=%d -- last lines:" % rc)
        print("".join(buf[-40:]), flush=True)
        if check:
            raise RuntimeError("command failed: %s" % cmd)
    return rc, "".join(buf)


def done(name):
    return os.path.exists("%s/.done_%s" % (ROOT, name))


def mark(name):
    open("%s/.done_%s" % (ROOT, name), "w").write(str(time.time()))


def find(name):
    """First file called `name` under the inputs; followlinks=True walks Kaggle's symlinked mounts
    (pathlib.rglob and plain `find` do not)."""
    for dp, _, fns in os.walk(INPUT, followlinks=True):
        if name in fns:
            return os.path.join(dp, name)
    return ""


def link(srcf, dst):
    """Symlink, falling back to a copy where symlinks are not allowed."""
    if srcf and not os.path.exists(dst):
        try:
            os.symlink(srcf, dst)
        except OSError:
            shutil.copy2(srcf, dst)


def main():
    # ---------- hardware ----------
    import torch
    gpu, ngpu, ncpu = torch.cuda.is_available(), torch.cuda.device_count(), os.cpu_count() or 4
    log("HARDWARE | GPU:", gpu, ngpu, [torch.cuda.get_device_name(i) for i in range(ngpu)], "| CPU cores:", ncpu)
    if not gpu:
        log("NO GPU -> reproduces his 0.982 + the canary on CPU; our CE ensemble needs a GPU (skipped).")

    # ---------- 1. code + artifacts: from the Kaggle dataset (no git on Kaggle) ----------
    # The v6ce code + artifacts/v6ce are uploaded from the local checkout as dataset
    # dhawal2209/almc-v6ce-src (folder `repo/`, possibly still zipped). Copy it to writable scratch.
    if not os.path.exists(REPO + "/artifacts/v6ce/runs/v6/fit_oof.parquet"):
        assert REPO == ROOT + "/repo", "V6_REPO=%s has no artifacts/v6ce" % REPO   # never touch a given repo
        f = find("fit_oof.parquet")
        if not f:
            z = find("repo.zip")
            assert z, "code dataset missing -> Add Input: dhawal2209/almc-v6ce-src"
            import zipfile
            zipfile.ZipFile(z).extractall(ROOT + "/unzipped")
            f = next((os.path.join(dp, "fit_oof.parquet") for dp, _, fns in os.walk(ROOT + "/unzipped")
                      if "fit_oof.parquet" in fns), "")
        assert f.replace("\\", "/").endswith("artifacts/v6ce/runs/v6/fit_oof.parquet"), "unexpected layout: %s" % f
        src_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(f)))))
        shutil.rmtree(REPO, ignore_errors=True)
        shutil.copytree(src_root, REPO)
        log("code+artifacts copied from dataset:", src_root)
    src, art = REPO + "/code/business_entity_resolution/src", REPO + "/artifacts/v6ce"
    log("code:", src, "| artifacts present:", os.path.isdir(art))

    # ---------- 2. raw data (walks Kaggle's symlinked dataset mounts) ----------
    if DATA != INPUT:
        for split in ("train", "test"):
            for s in (1, 2, 3):
                link(find("%s_source%d.tsv" % (split, s)), "%s/%s/%s_source%d.tsv" % (DATA, split, split, s))
        link(find("train_ground_truth.tsv"), DATA + "/train/train_ground_truth.tsv")
    need = (["train/train_source%d.tsv" % s for s in (1, 2, 3)] + ["test/test_source%d.tsv" % s for s in (1, 2, 3)]
            + ["train/train_ground_truth.tsv"])
    missing = [f for f in need if not os.path.exists(DATA + "/" + f)]
    assert not missing, "raw data missing %s -> Add Input: satwiksps/amazon-ml-challenge-2026" % missing
    log("raw data OK")

    # ---------- 3. artifacts -> work dir (artifacts/v6ce/README.md copy map) ----------
    if not done("artifacts"):
        os.makedirs(W + "/runs", exist_ok=True)
        shutil.copytree(art + "/aux_models", W + "/aux", dirs_exist_ok=True)
        for f in ("stage1_f0.lgb", "stage1_f1.lgb", "train_roles.parquet"):
            shutil.copy2(art + "/" + f, W + "/" + f)
        shutil.copytree(art + "/runs/v6", W + "/runs/v6", dirs_exist_ok=True)
        shutil.copytree(art + "/runs/v6ce", W + "/runs/v6ce", dirs_exist_ok=True)
        shutil.copytree(art + "/ce_scores_v6", ROOT + "/ce_scores_v6", dirs_exist_ok=True)
        for f in ("stage2_f0.lgb", "stage2_f1.lgb"):   # predict loads them; his stack() normally hard-links them
            if not os.path.exists(W + "/runs/v6ce/" + f):
                shutil.copy2(W + "/runs/v6/" + f, W + "/runs/v6ce/" + f)
        mark("artifacts")
    ts = find("test_scores.parquet")   # optional v6 test base (not in git): accept only the v6 schema
    if ts and not os.path.exists(W + "/runs/v6/test_scores.parquet"):
        import pyarrow.parquet as pq
        cols = set(pq.read_schema(ts).names)
        if {"s1_idx", "src", "t_idx", "p"} <= cols:
            shutil.copy2(ts, W + "/runs/v6/test_scores.parquet")
            log("using attached v6 test base scores:", ts)
        else:
            log("ignoring", ts, "- not a v6 test_scores table (columns %s)" % sorted(cols))
    has_test = os.path.exists(W + "/runs/v6/test_scores.parquet")
    log("v6 test base scores present:", has_test, "(needed only for OUR test submission)")

    # ---------- deps + embedded scripts ----------
    if not done("deps") and not os.environ.get("V6_NO_PIP"):
        sh("pip -q install polars==1.38.1 rapidfuzz anyascii sentencepiece protobuf", check=False)
        mark("deps")
    for fname, blob in (("ce_train_ens.py", "@@TRAIN@@"), ("ce_score_ens.py", "@@SCORE@@"),
                        ("bands.py", "@@BANDS@@"), ("stack_safe.py", "@@STACK@@")):
        open(S + "/" + fname, "w").write(base64.b64decode(blob).decode("utf-8"))

    env = dict(os.environ, PYTHONPATH=src, ER_WORK=W, ER_DATA=DATA, ER_OUT=OUTW + "/output",
               ER_WORKERS=str(ncpu), POLARS_MAX_THREADS=str(ncpu), ER_N_FIT="700000",
               ER_ADMIN_PARTS=W + "/aux/admin_parts.json", ER_TRANSLIT=W + "/aux/translit.json",
               ER_TRAIN_ROLES_ONLY="0", ER_NO_TX="0", PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    py = sys.executable

    def er(*args, extra=None, check=True):
        return sh([py, "-m"] + list(args), env=dict(env, **(extra or {})), cwd=src, check=check)

    def gate(a, b, tag):
        rc, out = er("er.compare", "--a", W + "/runs/" + a, "--b", W + "/runs/" + b, "--work-a", W, "--work-b", W,
                     check=False)
        js = out[out.find("{"):] if "{" in out else out
        open("%s/gate_%s.json" % (OUTW, tag), "w").write(js)
        log("GATE %s vs %s ->" % (b, a), js[:600].replace("\n", " "))

    def tune(run):
        er("er.run", "tune", "--run", run, "--scores", "ce")
        shutil.copy2("%s/runs/%s/decision_ce.json" % (W, run), "%s/runs/%s/decision.json" % (W, run))

    # ================= PHASE 0: prep + rebuild his 0.982 =================
    if not done("prep"):
        log("PHASE 0a: prep raw TSVs -> normalised tables (~20-40 min on 4 vCPU)")
        er("er.run", "prep", "--split", "all")
        mark("prep")
    if not done("p0_predict"):
        log("PHASE 0b: rebuild HIS v6ce submission from the committed stacked scores")
        rc, _ = er("er.run", "predict", "--run", "v6ce", "--scores", "ce", extra={"ER_SUB_NAME": "v6ce"}, check=False)
        if rc == 0:
            mark("p0_predict")
        log("his v6ce submission:", "OK -> output/v6ce" if rc == 0 else "FAILED (continuing)")

    # ================= PHASE 1: CE training pairs + uncertain bands =================
    ce_data, ce_band = ROOT + "/ce_data", ROOT + "/ce_band"
    if not done("export_train"):
        log("PHASE 1a: export CE training pairs (runs/v6/fit_oof, FIT fold 0)")
        os.makedirs(ce_data, exist_ok=True)
        er("er.cross", "export-train", "--run", "v6", "--out", ce_data)
        mark("export_train")
    if not done("bands"):
        log("PHASE 1b: export uncertain bands (fit1 / val / test-if-available)")
        sh([py, S + "/bands.py", "v6", ce_band, str(BAND_LO), str(BAND_HI)], env=env, cwd=src)
        mark("bands")

    # ================= CANARY: his CE scores through our environment =================
    if not done("canary"):
        log("CANARY: re-stack HIS ce_scores_v6 -> v6ce_repro (must reproduce ~0.98730)")
        sh([py, S + "/stack_safe.py", "v6", ROOT + "/ce_scores_v6", "v6ce_repro"], env=env, cwd=src)
        tune("v6ce_repro")
        gate("v6ce", "v6ce_repro", "canary_committed_vs_repro")
        mark("canary")

    if not gpu:
        log("no GPU -> stopping after the canary")
        return

    # ================= PHASE 2: OUR CE ensemble (GPU) =================
    models, ce_ours = ROOT + "/models", ROOT + "/ce_scores_ens"
    el = (time.time() - T0) / 3600
    preset = CE_PRESET if el < 4 else ("fast" if el < 7 else "smoke")
    if preset != CE_PRESET:
        log("TIME GUARD: %.1f h elapsed -> preset %s (12 h session limit)" % (el, preset))
    ce_env = dict(env, CE_TRAIN=ce_data + "/ce_train.parquet", CE_MODELS=models, CE_PRESET=preset,
                  CE_BAND=ce_band, CE_OUT=ce_ours)
    if not done("ce_train"):
        log("PHASE 2a: train OUR CE ensemble (preset=%s)" % preset)
        sh([py, S + "/ce_train_ens.py"], env=ce_env, cwd=S)
        mark("ce_train")
    if not done("ce_score"):
        log("PHASE 2b: score the bands with every member (ce = mean)")
        sh([py, S + "/ce_score_ens.py"], env=ce_env, cwd=S)
        mark("ce_score")

    # ================= PHASE 3: stack ours (his exact recipe), gate, predict =================
    if not done("stack_ens"):
        log("PHASE 3a: stack OUR ensemble -> v6ce_ens (same stacker recipe as his)")
        sh([py, S + "/stack_safe.py", "v6", ce_ours, "v6ce_ens"], env=env, cwd=src)
        mark("stack_ens")
    if not done("gate"):
        log("PHASE 3b: tune + GATE ours vs his")
        tune("v6ce_ens")
        gate("v6ce_repro", "v6ce_ens", "OURS_vs_his_same_env")
        gate("v6ce", "v6ce_ens", "OURS_vs_his_committed")
        mark("gate")
    if os.path.exists(W + "/runs/v6ce_ens/test_scores_ce.parquet") and not done("p3_predict"):
        log("PHASE 3c: OUR test submission")
        rc, _ = er("er.run", "predict", "--run", "v6ce_ens", "--scores", "ce", extra={"ER_SUB_NAME": "v6ce_ens"},
                   check=False)
        if rc == 0:
            mark("p3_predict")
    elif not has_test:
        log("NOTE: no v6 test base scores -> our test submission not written. If the gate says we WIN, attach "
            "runs/v6/test_scores.parquet (from Zayaan) and re-run: only phase 1b/2b/3 redo, in minutes.")


def summary():
    lines = ["=" * 90, "RESULTS  (validation macro F0.5 on the same 150k val S1; France has no labels -> LB only)",
             "his committed v6ce (README): 0.98730", ""]
    for run, label in (("v6ce", "HIS v6ce (committed)"), ("v6ce_repro", "HIS CE re-stacked here (canary)"),
                       ("v6ce_ens", "OUR CE ensemble")):
        p = "%s/runs/%s/decision_ce.json" % (W, run)
        if os.path.exists(p):
            d = json.load(open(p))
            b = d.get("best", {})
            lines.append("%-34s F0.5=%s  P=%s  R=%s  | by country %s" % (
                label, b.get("f05"), b.get("prec"), b.get("rec"),
                {k: (v.get("f05") if isinstance(v, dict) else v) for k, v in (d.get("by_country") or {}).items()}))
    lines.append("")
    for g in sorted(glob.glob(OUTW + "/gate_*.json")):
        lines.append("--- %s ---" % os.path.basename(g))
        lines.append(open(g).read()[:1500])
    for sub in ("v6ce", "v6ce_ens"):
        d = OUTW + "/output/" + sub
        if os.path.isdir(d):
            shutil.make_archive(OUTW + "/" + sub + "_submission", "zip", d)
            lines.append("submission: %s_submission.zip  (%s)" % (sub, ", ".join(sorted(os.listdir(d)))))
    lines.append("total time: %.2f h" % ((time.time() - T0) / 3600))
    txt = "\n".join(str(x) for x in lines)
    open(OUTW + "/SUMMARY.txt", "w").write(txt)
    print(txt, flush=True)
    try:
        from IPython.display import FileLink, display
        for f in sorted(glob.glob(OUTW + "/*_submission.zip")) + [OUTW + "/SUMMARY.txt"]:
            display(FileLink(os.path.relpath(f, OUTW)))
    except Exception:
        pass


try:
    main()
except Exception as e:
    log("STOPPED:", repr(e), "-> see run_v6ce_ens.log; re-run the cell to resume from the last finished step")
finally:
    summary()
'''

cell = (CELL.replace("@@TRAIN@@", b64(TRAIN)).replace("@@SCORE@@", b64(SCORE))
            .replace("@@BANDS@@", b64(BANDS)).replace("@@STACK@@", b64(STACK)))
assert "@@" not in cell


def md(*lines):
    return {"cell_type": "markdown", "id": "intro", "metadata": {}, "source": [l + "\n" for l in lines]}


nb = {
    "cells": [
        md("# ALMC - v6ce reproduction + our CE-ensemble on top (single cell)",
           "",
           "**Settings:** Accelerator = GPU T4 x2, Internet = ON. **Add Input:** the raw dataset",
           "(`satwiksps/amazon-ml-challenge-2026`) and the code dataset (`dhawal2209/almc-v6ce-src`, uploaded from",
           "the local checkout - no git on Kaggle). Optional: `runs/v6/test_scores.parquet` from Zayaan (enables our",
           "test submission). Then **Save Version -> Save & Run All** and sleep.",
           "",
           "Read `SUMMARY.txt` in the Output tab: his 0.98730, the canary (must match), our ensemble, and the gate."),
        {"cell_type": "code", "id": "run", "execution_count": None, "metadata": {}, "outputs": [],
         "source": [l + "\n" for l in cell.split("\n")[:-1]] + [cell.split("\n")[-1]]},
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}, "kaggle": {"accelerator": "nvidiaTeslaT4",
                                                                 "isInternetEnabled": True}},
    "nbformat": 4, "nbformat_minor": 5,
}
out = os.path.join(HERE, "kaggle_v6ce_ens.ipynb")
with open(out, "w", encoding="utf-8", newline="\n") as f:
    json.dump(nb, f, indent=1)
print("wrote", out, "| cell lines:", cell.count("\n") + 1, "| cell chars:", len(cell))
