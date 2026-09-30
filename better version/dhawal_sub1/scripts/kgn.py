"""Kaggle CLI as account N (1 = default ~/.kaggle/access_token, 2/3 = ~/.kaggle/acct{N}_token), cwd = parent dir.
Usage: python kgn.py <N> <parent_dir> <kaggle args...>"""
import os
import subprocess
import sys

n, parent, args = sys.argv[1], sys.argv[2], sys.argv[3:]
env = dict(os.environ, TEMP="D:/kaggle_tmp", TMP="D:/kaggle_tmp", PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
os.makedirs("D:/kaggle_tmp", exist_ok=True)
if n != "1":
    env["KAGGLE_API_TOKEN"] = open(os.path.expanduser(f"~/.kaggle/acct{n}_token")).read().strip()
r = subprocess.run([sys.executable, "-m", "kaggle", *args], cwd=parent, env=env)
sys.exit(r.returncode)
