"""End-to-end regression test on a miniature real-data sample (slow; opt-in: pytest -m slow)."""
import os
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

from er import config as C

pytestmark = pytest.mark.slow


@pytest.mark.skipif(not (C.DATA / "train" / "train_source1.tsv").exists(), reason="dataset not present")
def test_pipeline_end_to_end(tmp_path):
    env = {**os.environ, "ER_DATA": str(tmp_path / "data"), "ER_WORK": str(tmp_path / "work"),
           "ER_OUT": str(tmp_path / "out"), "ER_N_VAL": "1200", "ER_N_FIT": "2800", "PYTHONIOENCODING": "utf-8"}
    src = Path(__file__).parents[1] / "src"
    run = lambda *a: subprocess.run([sys.executable, "-m", *a], cwd=src, env=env, check=True,
                                    capture_output=True, text=True)
    run("er.minidata", str(tmp_path / "data"), "--n", "2000")
    run("er.run", "all", "--run", "e2e")
    m = pl.read_csv(tmp_path / "out" / "matching_results.tsv", separator="\t", infer_schema=False)
    assert m.height == 6000 and m["source1_entity_id"].n_unique() == 6000
    import json
    s = json.loads((tmp_path / "work" / "runs" / "e2e" / "summary.json").read_text())
    assert s["best"]["f05"] > 0.95      # regression floor on the mini validation split
