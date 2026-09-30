"""Kaggle CLI wrapper for Windows: run a kaggle command with cwd = parent folder so -p is a bare folder name
(the CLI's upload cache breaks on paths containing slashes). Usage: python kg.py <parent_dir> <kaggle args...>
Use the literal token DIR for the -p argument's folder name, e.g.  kg.py E:/x/rep datasets create -p ce_data"""
import subprocess
import sys

parent, args = sys.argv[1], sys.argv[2:]
r = subprocess.run([sys.executable, "-m", "kaggle", *args], cwd=parent)
sys.exit(r.returncode)
