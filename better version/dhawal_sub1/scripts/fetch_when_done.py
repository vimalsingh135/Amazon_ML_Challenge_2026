"""Poll a Kaggle kernel until it finishes, then download its outputs. Usage: fetch_when_done.py <acct> <parent> <kernel_ref> <out_folder>"""
import subprocess
import sys
import time

acct, parent, ref, out = sys.argv[1:5]
KG = [sys.executable, "E:/projects/Amazon ML challenge/.worktrees/v6val/tmp/kgn.py", acct, parent]
st = ""
for i in range(240):
    st = subprocess.run(KG + ["kernels", "status", ref], capture_output=True, text=True, errors="replace").stdout
    if "COMPLETE" in st or "ERROR" in st or "CANCEL" in st:
        break
    time.sleep(30)
print("status:", st.strip(), flush=True)
r = subprocess.run(KG + ["kernels", "output", ref, "-p", out], capture_output=True, text=True, errors="replace")
print(r.stdout[-1500:], r.stderr[-500:], flush=True)
