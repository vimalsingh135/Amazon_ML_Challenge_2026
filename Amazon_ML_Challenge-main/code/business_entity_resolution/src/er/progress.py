"""Live progress report for a running pipeline:  python -m er.progress"""
from __future__ import annotations

import os
import time

import polars as pl

from . import config as C


def _s1_counts() -> dict[tuple[str, str], int]:
    out = {}
    for split in ("train", "test"):
        p = C.WORK / f"{split}_s1.parquet"
        roles = C.WORK / "train_roles.parquet"
        if split == "train" and C.TRAIN_ROLES_ONLY and roles.exists():
            for c, n in pl.read_parquet(roles).group_by("country").len().iter_rows():
                out[(split, c)] = n
        elif p.exists():
            for c, n in pl.scan_parquet(p).group_by("country").len().collect().iter_rows():
                out[(split, c)] = n
    return out


def report() -> None:
    now = time.time()
    counts = _s1_counts()
    print(f"{'partition':<22}{'status':<10}{'S1 done':>18}{'chunks':>8}{'min':>7}{'S1/min':>9}{'ETA min':>9}")
    total_left = 0.0
    rates = []
    for (split, country), n in sorted(counts.items(), key=lambda x: (x[0][0] != "train", x[0][1])):
        for src in ("2", "3"):
            d = C.WORK / f"{split}_A_{country}_{src}"
            name = f"{split}_{country}_S{src}"
            if (d / "_SUCCESS").exists():
                fs = list(d.glob("part_*.parquet"))
                ts = [os.path.getmtime(f) for f in fs]
                print(f"{name:<22}{'done':<10}{n:>18,}{len(fs):>8}")
                continue
            tmp = d / "tmp"
            fs = sorted(tmp.glob("part_*.parquet")) if tmp.exists() else []
            if not fs:
                print(f"{name:<22}{'queued':<10}{'0 / ' + format(n, ','):>18}")
                total_left += n
                continue
            ok = []
            for f in fs:  # skip a chunk that is still being written
                try:
                    ok.append(pl.read_parquet(f, columns=["s1_idx"]))
                except Exception:
                    pass
            done = pl.concat(ok)["s1_idx"].n_unique() if ok else 0
            start = os.path.getctime(tmp)
            mins = (now - start) / 60
            rate = done / max(mins, 1e-6)
            rates.append(rate)
            eta = (n - done) / rate if rate else float("nan")
            total_left += n - done
            print(f"{name:<22}{'running':<10}{f'{done:,} / {n:,}':>18}{len(fs):>8}{mins:>7.1f}{rate:>9,.0f}{eta:>9.1f}")
    if rates:
        r = sum(rates) / len(rates)
        print(f"\nStage A remaining: ~{total_left:,.0f} S1-partition units at ~{r:,.0f}/min "
              f"-> ~{total_left / r / 60:.1f} h (then stage1, stage B, train2, predict)")
    for st in ("P", "B"):
        fs = list(C.WORK.glob(f"*_{st}_*"))
        if fs:
            print(f"stage {st}: {len(fs)} partitions present")
    logs = sorted(C.WORK.glob("run_*.log"), key=os.path.getmtime)
    if logs:
        log = logs[-1]
        tail = [l for l in log.read_text(errors="ignore").splitlines() if " stage " in l or "exit=" in l]
        print("\nlast stage events:", *tail[-4:], sep="\n  ")
    mem = C.WORK / "mem.log"
    if mem.exists():
        print("memory:", mem.read_text().splitlines()[-1])


if __name__ == "__main__":
    report()
