"""Zip dhawal_sub1/ (code, docs, small results) + Zayaan's er/ package into one reproducible code bundle."""
import os
import zipfile

WT = "E:/projects/Amazon ML challenge/.worktrees"
SRC = WT + "/final2/dhawal_sub1"
ER = WT + "/final2/code/business_entity_resolution/src/er"
OUT = WT + "/rep/output/dhawal_sub1_code.zip"
n = 0
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, fs in os.walk(SRC):
        for f in fs:
            p = os.path.join(root, f)
            z.write(p, os.path.join("dhawal_sub1", os.path.relpath(p, SRC)))
            n += 1
    for root, _, fs in os.walk(ER):
        if "__pycache__" in root:
            continue
        for f in fs:
            p = os.path.join(root, f)
            z.write(p, os.path.join("dhawal_sub1", "code", "er", os.path.relpath(p, ER)))
            n += 1
print(n, "files ->", OUT, round(os.path.getsize(OUT) / 2**20, 2), "MB")
