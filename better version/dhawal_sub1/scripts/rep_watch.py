"""Watcher for the v6cd replication + variant lanes across three Kaggle accounts. Every 2 min:
  A. bi-encoder trains on account 2 (codelearner00/almc-bi-train). When COMPLETE: download, publish bi_model as a
     dataset on account 1 (dhawal2209/almc-bi-model) and account 3 (dhawal2209000/almc-bi-model).
  B. CE trains on account 1 (dhawal2209/almc-ce-train). When COMPLETE: download, publish as dataset on account 3
     (dhawal2209000/almc-ce-model).
  C. account 1: push almc-dense (faithful replica, K=10, keep 95%) -> download to rep/dense_out.
  D. account 3: push almc-dense-k20 (K=20, keep 99%; superset for local variants) -> download to rep3/dense_out.
Idempotent via marker files. Log: .worktrees/v6val/out/rep_watch.log"""
import json
import os
import subprocess
import sys
import time

W = "E:/projects/Amazon ML challenge/.worktrees"
REP, REP3 = W + "/rep", W + "/rep3"
LOG = W + "/v6val/out/rep_watch.log"
PY = sys.executable
TOK = {n: open(os.path.expanduser(f"~/.kaggle/acct{n}_token")).read().strip() for n in (2, 3, 4, 5)}
REP4 = W + "/rep4"


def log(*a):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a) + "\n")


def kg(*a, acct=1, cwd=REP):
    env = dict(os.environ, TEMP="D:/kaggle_tmp", TMP="D:/kaggle_tmp",    # upload zips on D:, not the nearly full C:
               PYTHONIOENCODING="utf-8", PYTHONUTF8="1")                  # cp1252 console breaks the CLI progress bar
    os.makedirs("D:/kaggle_tmp", exist_ok=True)
    if acct != 1:
        env["KAGGLE_API_TOKEN"] = TOK[acct]
    r = subprocess.run([PY, "-m", "kaggle", *a], cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return (r.stdout + r.stderr).strip()


def status(ref, acct=1):
    s = kg("kernels", "status", ref, acct=acct)
    for w in ("COMPLETE", "RUNNING", "QUEUED", "ERROR", "CANCEL"):
        if w in s:
            return w
    return "NONE"


mark = lambda n: os.path.exists(f"{REP}/.{n}")
setmark = lambda n: open(f"{REP}/.{n}", "w").close()
ready = lambda ref, acct=1: "ready" in kg("datasets", "status", ref, acct=acct)


def publish(folder_parent, folder, ref, acct):
    json.dump({"title": ref.split("/")[1], "id": ref, "licenses": [{"name": "CC0-1.0"}]},
              open(f"{folder_parent}/{folder}/dataset-metadata.json", "w"))
    return kg("datasets", "create", "-q", "-p", folder, "--dir-mode", "zip", acct=acct, cwd=folder_parent)[-200:]


log("watcher v3 start (3 accounts)")
while True:
    try:
        st = {"bi@2": status("codelearner00/almc-bi-train", 2), "ce@1": status("dhawal2209/almc-ce-train"),
              "dense@1": status("dhawal2209/almc-dense"), "k20@3": status("dhawal2209000/almc-dense-k20", 3),
              "ens@1": status("dhawal2209/almc-v6ce-ens"), "k50@4": status("dhawal22092004/almc-dense-k50", 4),
              "ce2@5": status("dhwalkhatri12873/almc-ce2-train", 5)}
        log("status", st)
        if st["bi@2"] == "ERROR" or st["ce@1"] == "ERROR":
            log("TRAINING KERNEL ERROR - needs attention")
        # A. bi model -> accounts 1 and 3
        if st["bi@2"] == "COMPLETE" and not mark("bi_downloaded"):
            os.makedirs(f"{REP}/bi_model_ds", exist_ok=True)
            log("download bi", kg("kernels", "output", "codelearner00/almc-bi-train", "-p", "bi_model_ds", acct=2)[-200:])
            setmark("bi_downloaded")
        if mark("bi_downloaded") and not mark("bi_model_uploaded"):
            log("publish bi@1", publish(REP, "bi_model_ds", "dhawal2209/almc-bi-model", 1))
            log("publish bi@3", publish(REP, "bi_model_ds", "dhawal2209000/almc-bi-model", 3))
            setmark("bi_model_uploaded")
        # B. CE model -> account 3
        if st["ce@1"] == "COMPLETE" and not mark("ce_downloaded"):
            os.makedirs(f"{REP}/ce_model_ds", exist_ok=True)
            log("download ce", kg("kernels", "output", "dhawal2209/almc-ce-train", "-p", "ce_model_ds")[-200:])
            setmark("ce_downloaded")
        if mark("ce_downloaded") and not mark("ce_model_uploaded"):
            log("publish ce@3", publish(REP, "ce_model_ds", "dhawal2209000/almc-ce-model", 3))
            setmark("ce_model_uploaded")
        # C. faithful replica on account 1
        if (mark("bi_model_uploaded") and not mark("pushed_dense") and st["ce@1"] == "COMPLETE"
                and ready("dhawal2209/almc-dense-data") and ready("dhawal2209/almc-bi-model")):
            out = kg("kernels", "push", "-p", "k_dense")
            log("push dense@1 ->", out[-160:])
            if "successfully pushed" in out:
                setmark("pushed_dense")
        if mark("pushed_dense") and st["dense@1"] == "COMPLETE" and not mark("downloaded_dense"):
            os.makedirs(f"{REP}/dense_out", exist_ok=True)
            log("download dense@1", kg("kernels", "output", "dhawal2209/almc-dense", "-p", "dense_out")[-300:])
            setmark("downloaded_dense")
            log("DENSE_OUT_READY (K10 replica)")
        # D. K=20 / keep-99% variant on account 3
        if (mark("bi_model_uploaded") and mark("ce_model_uploaded") and not mark("pushed_k20")
                and ready("dhawal2209000/almc-dense-data", 3) and ready("dhawal2209000/almc-bi-model", 3)
                and ready("dhawal2209000/almc-ce-model", 3)):
            out = kg("kernels", "push", "-p", "k_dense", acct=3, cwd=REP3)
            log("push k20@3 ->", out[-160:])
            if "successfully pushed" in out:
                setmark("pushed_k20")
        if mark("pushed_k20") and st["k20@3"] == "COMPLETE" and not mark("downloaded_k20"):
            os.makedirs(f"{REP3}/dense_out", exist_ok=True)
            log("download k20@3", kg("kernels", "output", "dhawal2209000/almc-dense-k20", "-p", "dense_out", acct=3, cwd=REP3)[-300:])
            setmark("downloaded_k20")
            log("DENSE_OUT_READY (K20 variant)")
        # E. account 4: K=50 superset
        if mark("bi_downloaded") and not mark("bi_model_uploaded4"):
            log("publish bi@4", publish(REP, "bi_model_ds", "dhawal22092004/almc-bi-model", 4))
            setmark("bi_model_uploaded4")
        if mark("ce_downloaded") and not mark("ce_model_uploaded4"):
            log("publish ce@4", publish(REP, "ce_model_ds", "dhawal22092004/almc-ce-model", 4))
            setmark("ce_model_uploaded4")
        if (mark("bi_model_uploaded4") and mark("ce_model_uploaded4") and not mark("pushed_k50")
                and ready("dhawal22092004/almc-dense-data", 4) and ready("dhawal22092004/almc-bi-model", 4)
                and ready("dhawal22092004/almc-ce-model", 4)):
            out = kg("kernels", "push", "-p", "k_dense", acct=4, cwd=REP4)
            log("push k50@4 ->", out[-160:])
            if "successfully pushed" in out:
                setmark("pushed_k50")
        if mark("pushed_k50") and st["k50@4"] == "COMPLETE" and not mark("downloaded_k50"):
            os.makedirs(f"{REP4}/dense_out", exist_ok=True)
            log("download k50@4", kg("kernels", "output", "dhawal22092004/almc-dense-k50", "-p", "dense_out", acct=4, cwd=REP4)[-300:])
            setmark("downloaded_k50")
            log("DENSE_OUT_READY (K50 variant)")
        if st["ce2@5"] == "COMPLETE" and not mark("ce2_downloaded"):
            os.makedirs(f"{W}/rep5/ce2_model_ds", exist_ok=True)
            log("download ce2", kg("kernels", "output", "dhwalkhatri12873/almc-ce2-train", "-p", "ce2_model_ds", acct=5, cwd=W + "/rep5")[-200:])
            setmark("ce2_downloaded")
            log("CE2_MODEL_READY")
        # F. account 5: CE2 scores the K=20 dense pairs (published from rep3/dense_out)
        if mark("downloaded_k20") and not mark("pairs_uploaded5"):
            log("publish pairs@5", publish(REP3, "dense_out", "dhwalkhatri12873/almc-dense-pairs", 5))
            setmark("pairs_uploaded5")
        st5 = status("dhwalkhatri12873/almc-ce2-score", 5) if mark("pushed_ce2score") else "NONE"
        if (mark("pairs_uploaded5") and not mark("pushed_ce2score") and ready("dhwalkhatri12873/almc-dense-pairs", 5)
                and ready("dhwalkhatri12873/almc-dense-data", 5)):
            out = kg("kernels", "push", "-p", "k_ce2score", acct=5, cwd=W + "/rep5")
            log("push ce2score@5 ->", out[-160:])
            if "successfully pushed" in out:
                setmark("pushed_ce2score")
        if mark("pushed_ce2score") and st5 == "COMPLETE" and not mark("downloaded_ce2score"):
            os.makedirs(f"{W}/rep5/ce2_out", exist_ok=True)
            log("download ce2score", kg("kernels", "output", "dhwalkhatri12873/almc-ce2-score", "-p", "ce2_out", acct=5, cwd=W + "/rep5")[-300:])
            setmark("downloaded_ce2score")
            log("CE2_SCORES_READY")
        if st5 == "ERROR":
            log("CE2SCORE ERROR")
        # G. extra experiments (afternoon): bi2 -> dense-bi2 on acct2; CE-large -> ce3 scores on acct5; France K20 on acct1
        x = {"bi2@2": status("codelearner00/almc-bi2-train", 2), "dbi2@2": status("codelearner00/almc-dense-bi2", 2),
             "celarge@5": status("dhwalkhatri12873/almc-celarge-train", 5), "ce3@5": status("dhwalkhatri12873/almc-ce3-score", 5),
             "fr20@1": status("dhawal2209/almc-fr-dense-k20"), "bi3@3": status("dhawal2209000/almc-bi3-base", 3),
             "bi4@4": status("dhawal22092004/almc-bi4-2ep", 4)}
        log("extra", x)
        if (x["bi2@2"] == "COMPLETE" and not mark("pushed_dbi2") and ready("codelearner00/almc-dense-data", 2)
                and ready("codelearner00/almc-ce-model", 2)):
            out = kg("kernels", "push", "-p", "k_dense_bi2", acct=2, cwd=W + "/rep2")
            log("push dense-bi2@2 ->", out[-160:])
            if "successfully pushed" in out:
                setmark("pushed_dbi2")
        if mark("pushed_dbi2") and x["dbi2@2"] == "COMPLETE" and not mark("downloaded_dbi2"):
            os.makedirs(f"{W}/rep2/dense_out", exist_ok=True)
            log("download dense-bi2", kg("kernels", "output", "codelearner00/almc-dense-bi2", "-p", "dense_out", acct=2, cwd=W + "/rep2")[-300:])
            setmark("downloaded_dbi2")
            log("DENSE_OUT_READY (bi2 hard negatives)")
        if (x["celarge@5"] == "COMPLETE" and mark("pairs_uploaded5") and not mark("pushed_ce3")
                and ready("dhwalkhatri12873/almc-dense-pairs", 5)):
            out = kg("kernels", "push", "-p", "k_ce3score", acct=5, cwd=W + "/rep5")
            log("push ce3score@5 ->", out[-160:])
            if "successfully pushed" in out:
                setmark("pushed_ce3")
        if mark("pushed_ce3") and x["ce3@5"] == "COMPLETE" and not mark("downloaded_ce3"):
            os.makedirs(f"{W}/rep5/ce3_out", exist_ok=True)
            log("download ce3", kg("kernels", "output", "dhwalkhatri12873/almc-ce3-score", "-p", "ce3_out", acct=5, cwd=W + "/rep5")[-300:])
            setmark("downloaded_ce3")
            log("CE3_SCORES_READY")
        if x["fr20@1"] == "COMPLETE" and not mark("downloaded_fr20"):
            os.makedirs(f"{REP}/fr_dense_out", exist_ok=True)
            log("download fr20", kg("kernels", "output", "dhawal2209/almc-fr-dense-k20", "-p", "fr_dense_out")[-300:])
            setmark("downloaded_fr20")
            log("FRANCE_K20_READY")
        for ref, acct, cwd, tag_ in (("dhawal2209/almc-fr-dense-k50", 1, REP, "fr50"), ("codelearner00/almc-fr-dense-bi2", 2, W + "/rep2", "frbi2")):
            s_ = status(ref, acct)
            x[tag_] = s_
            if s_ == "COMPLETE" and not mark(f"downloaded_{tag_}"):
                os.makedirs(f"{cwd}/{tag_}_out", exist_ok=True)
                log(f"download {tag_}", kg("kernels", "output", ref, "-p", f"{tag_}_out", acct=acct, cwd=cwd)[-300:])
                setmark(f"downloaded_{tag_}")
                log(f"FRANCE_{tag_.upper()}_READY")
        for k_, v_ in x.items():
            if v_ == "ERROR":
                log("EXTRA ERROR", k_)
        if st["dense@1"] == "ERROR" and mark("pushed_dense"):
            log("DENSE@1 ERROR")
        if st["k20@3"] == "ERROR" and mark("pushed_k20"):
            log("K20@3 ERROR")
        if all(mark(m_) for m_ in ("downloaded_dense", "downloaded_k20", "downloaded_k50", "downloaded_ce2score",
                                   "downloaded_dbi2", "downloaded_ce3", "downloaded_fr20")):
            log("ALL DENSE OUTPUTS READY"); break
    except Exception as e:  # keep watching through transient API/network errors
        log("exception", repr(e))
    time.sleep(120)
log("watcher end")
