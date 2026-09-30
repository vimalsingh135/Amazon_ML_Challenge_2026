"""Paths and global settings. Override the data/work roots with ER_DATA / ER_WORK."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]  # repo root
DATA = Path(os.environ.get("ER_DATA", ROOT / "student_resource" / "dataset"))
WORK = Path(os.environ.get("ER_WORK", ROOT / "work"))
OUT = Path(os.environ.get("ER_OUT", ROOT / "output"))
WORK.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

WORKERS = int(os.environ.get("ER_WORKERS", max(1, (os.cpu_count() or 4) - 2)))
SEED = 42

# Fast-path switches (v0): block only fit/val S1 on train, and drop competition (tx_*) features,
# which would be biased when not every S1 is blocked. Full-strength runs set both to 0.
TRAIN_ROLES_ONLY = os.environ.get("ER_TRAIN_ROLES_ONLY", "1") == "1"
NO_TX = os.environ.get("ER_NO_TX", "1") == "1"
CAP_SINGLE = int(os.environ.get("ER_CAP_SINGLE", 300))
CAP_PAIR = int(os.environ.get("ER_CAP_PAIR", 600))
JOIN_BUDGET = int(os.environ.get("ER_JOIN_BUDGET", 20_000_000))   # max joined rows per blocking chunk
# blocking candidate budgets per S1 per source: total / name / address / pure-name / house-number
BUDGETS = dict(top_k=int(os.environ.get("ER_TOP_K", 50)), top_name=int(os.environ.get("ER_TOP_NAME", 20)),
               top_addr=int(os.environ.get("ER_TOP_ADDR", 25)), top_pure=int(os.environ.get("ER_TOP_PURE", 15)),
               top_num=int(os.environ.get("ER_TOP_NUM", 15)))
