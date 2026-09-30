"""CLI entry point.

python -m er.run prep      [--split train|test|all]   normalise raw TSVs            -> work/{split}_s*.parquet
python -m er.run aux                                  fit noise/alias models      -> work/aux/
python -m er.run stage_a   [--split ...]              blocking + cheap/competition  -> work/{split}_A_*.parquet
python -m er.run stage1                               OOF pruning ranker + prune    -> work/{split}_P_*.parquet
python -m er.run stage_b   [--split ...]              rich pair features            -> work/{split}_B_*.parquet
python -m er.run train2    [--run v1]                 matcher + calibration + decision tuning (val)
python -m er.run predict   [--run v1]                 test inference -> output/*.tsv (validated)
python -m er.run all                                  everything, resumable (finished stages are skipped)
"""
from __future__ import annotations

import argparse
import logging
import time

STAGES = ["prep", "aux", "stage_a", "stage1", "stage_b", "train2", "tune", "predict"]
EXTRA = ["ensemble", "analysis", "loco"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=STAGES + EXTRA + ["all"])
    ap.add_argument("--split", default="all")
    ap.add_argument("--run", default="v1")
    ap.add_argument("--scores", default="", help="score set for tune/predict: '' (stage-2 LGB) or 'ens'")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    splits = ["train", "test"] if args.split == "all" else [args.split]
    todo = STAGES if args.stage == "all" else [args.stage]
    from . import pipeline as P
    for stage in todo:
        t = time.time()
        if stage == "prep":
            from .prep import prep_split
            for s in splits:
                prep_split(s)
        elif stage == "aux":
            from .auxfit import fit_aux
            fit_aux()
        elif stage == "stage_a":
            for s in splits:
                P.run_stage_a(s)
        elif stage == "stage1":
            P.train_stage1()
            for s in splits:
                P.run_stage1(s)
        elif stage == "stage_b":
            for s in splits:
                P.run_stage_b(s)
        elif stage == "train2":
            P.run_train2(args.run)
        elif stage == "tune":
            P.run_tune(args.run, args.scores)
        elif stage == "predict":
            P.run_predict(args.run, args.scores)
        elif stage == "ensemble":
            from .ensemble import run_ensemble
            run_ensemble(args.run)
        elif stage == "loco":
            from .transfer import run_loco
            run_loco(args.run)
        elif stage == "analysis":
            from .analysis import analyse
            analyse(args.run)
        logging.info("stage %s done in %.1fs", stage, time.time() - t)


if __name__ == "__main__":
    main()
