"""Wait until a Kaggle dataset is ready, then push a kernel folder. Usage: push_when_ready.py <acct> <parent> <dataset_ref> <kernel_folder>"""
import subprocess
import sys
import time

acct, parent, ds, kfolder = sys.argv[1:5]
KG = [sys.executable, "E:/projects/Amazon ML challenge/.worktrees/v6val/tmp/kgn.py", acct, parent]
for i in range(90):
    r = subprocess.run(KG + ["datasets", "status", ds], capture_output=True, text=True, errors="replace")
    if "ready" in r.stdout:
        break
    time.sleep(20)
print("dataset:", r.stdout.strip() or r.stderr.strip()[-120:], flush=True)
r = subprocess.run(KG + ["kernels", "push", "-p", kfolder], capture_output=True, text=True, errors="replace")
print(r.stdout[-300:], r.stderr[-300:], flush=True)
