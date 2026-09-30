# business_entity_resolution: reproduction guide

End-to-end pipeline for the ML Challenge 2026 Business Entity Resolution task. From the raw TSVs it produces:
- `output/matching_results.tsv`
- `output/candidate_pairs.tsv`

## 1. Environment
- Python 3.12 (tested on Windows 11 and Amazon Linux 2023), CPU only.
- 16 GB RAM is enough: every stage is memory-bounded by design.
- A free GPU is **not** required.

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt        # Linux/macOS: .venv/bin/pip
```

All dependencies are pinned in `requirements.txt`. Every model used is MIT / Apache-2.0 licensed (LightGBM MIT, CatBoost Apache-2.0, scikit-learn BSD-3). No model exceeds a few MB. No pretrained or external data is used.

## 2. Data layout
By default the pipeline reads `../../student_resource/dataset/{train,test}/*.tsv` relative to this folder. You can override the locations:

| Variable | Meaning | Default |
|---|---|---|
| `ER_DATA` | dataset root (contains `train/`, `test/`) | `<repo>/student_resource/dataset` |
| `ER_WORK` | cache / checkpoints | `<repo>/work` |
| `ER_OUT` | submission output | `<repo>/output` |
| `ER_WORKERS` | CPU workers | #cores − 2 |

## 3. Run
```bash
cd src
python -m er.run all --run v1                 # full pipeline, resumable
python -m er.run ensemble --run v1            # optional: LGB-rank + CatBoost + stacker
python -m er.run tune --run v1 --scores ens   # re-optimise the decision layer on validation
python -m er.run predict --run v1 --scores ens
```

`all` runs these stages in order. Each stage checkpoints to `ER_WORK`, and finished stages are skipped on rerun.

| Stage | Output | What it does |
|---|---|---|
| `prep` | `work/{split}_s{1,2,3}.parquet` | parallel normalisation (transliteration, legal forms, d/b/a split, address parsing) |
| `aux` | `work/aux/` | learned noise models from train positives (aliases, filler tokens, name-replacement detector) |
| `stage_a` | `work/{split}_A_*` | IDF multi-key blocking over **all** S1, cheap + context + competition features |
| `stage1` | `work/{split}_P_*.parquet` | 2-fold OOF LightGBM ranker. Pruned candidates = **candidate_pairs.tsv** |
| `stage_b` | `work/{split}_B_*` | about 150 rich pair features incl. aliases, OCR repair, cluster coherence |
| `train2` | `work/runs/<run>/` | 2-fold OOF LightGBM matcher, per-source isotonic calibration |
| `tune` | `decision.json` | exact / ratio expected-F0.5 decision layer tuned on untouched validation S1 |
| `predict` | `output/*.tsv` | scoring, decision, submission writer; invariants asserted and the official validator run automatically |

Diagnostics: `python -m er.run analysis --run v1` (layer recall, calibration, error taxonomy) and `python -m er.run loco --run v1` (leave-one-country-out transfer, the proxy for unseen France).

## 4. Tests
```bash
python -m pytest -q            # unit tests (normalisation, blocking keys, features, decision, writer, ...)
python -m pytest -m slow -q    # end-to-end regression on a mini real-data sample (er.minidata)
```

## 5. Source map (`src/er/`)
| Module | Role |
|---|---|
| `text.py` | Brahmic-script transliteration, ASCII folding, phonetic skeletons |
| `normalize.py` | name / address normalisation and component extraction |
| `aliases.py` | d/b/a marker split, alias mining, word segmentation |
| `noise.py` | filler-token mining, name-replacement detector |
| `auxfit.py` | fits / applies the learned noise models, OCR digit repair |
| `blocking.py` | IDF multi-key blocking with cost-aware chunking |
| `features.py` | pair features, cluster-coherence features |
| `pipeline.py` | stage orchestration (A → 1 → B → 2 → tune → predict) |
| `ensemble.py` | lambdarank + CatBoost + stacker |
| `decide.py` | target exclusivity + expected-F0.5 set selection (ratio / exact) |
| `metric.py` | exact macro F0.5 (incl. singleton rules) |
| `output.py` | submission writer with hard invariants + official validator |
| `analysis.py`, `transfer.py`, `diagnostics.py` | validation analysis, LOCO, blocking diagnostics |
| `minidata.py` | mini dataset builder for end-to-end tests |
