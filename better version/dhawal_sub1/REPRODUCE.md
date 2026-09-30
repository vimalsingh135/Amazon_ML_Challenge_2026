# Reproduce submission #1 (`SUB1_ours_usin_fr0988`) in under 14 hours

Wall-clock ≈ **7 h**. The long steps run in parallel on Kaggle GPUs, so there is ~7 h of margin under 14 h. Nothing
here needs external data; every model is trained on the competition's training set.

## Inputs
| Input | Where |
|---|---|
| Competition data (`train/`, `test/` TSVs) | Kaggle dataset `satwiksps/amazon-ml-challenge-2026`, or `student_resource/dataset/` |
| Zayaan's pipeline code `er/` | `code/er/` in this zip, identical to `code/business_entity_resolution/src/er` on branch `final2` |
| v6ce base artifacts (stage-1/2 LightGBM, calibrators, v6ce stacked val/test scores, train roles, aux models) | branch `final2` → `artifacts/v6ce/` (restore with `artifacts/final2/restore.py`), no retraining |
| France rows of LB 0.988 | Balaji's `final2_fr` output (`output_claude/final2_fr/matching_results_final2_fr.tsv.zip` on branch `final2_fr`) |

## Steps and timings

| # | Step | Where | Time |
|---|---|---|---|
| 1 | Restore v6ce artifacts into `work_v6/` and run `python -m er.run prep` (test tables + ID maps) | CPU (Kaggle or local) | ~25 min |
| 2 | `scripts/rep_export.py`: CE train pairs, bi-encoder pairs (1.2M FIT positives) and dense-retrieval inputs, with Zayaan's `er.cross export-biencoder` / `er.dense export` | CPU | ~10 min |
| 3a | `kaggle/ce_train_xlmr` (xlm-roberta-base cross-encoder, 2 epochs) | Kaggle GPU T4 | ~1.5 h |
| 3b | `kaggle/bi_train_e5small` (multilingual-e5-small bi-encoder, MNRL, 1 epoch), **in parallel with 3a** | Kaggle GPU T4 | ~1.5 h |
| 4 | `kaggle/dense_k10_replica` (top-10 dense retrieval of new pairs, keep 95% of new val true pairs, CE-scored) | Kaggle GPU T4 | ~3.8 h |
| 5 | `er.dense merge --run v6ce --dense dense_out --out-run v6cd`, then `er.run tune --run v6cd --scores ce`, then `er.compare` vs v6ce (`scripts/dense_variants.py replica`) | CPU | ~25 min |
| 6 | `kaggle/sub1_assemble` (decision + write_submission for v6cd US/India, validator `--check-ids`) | Kaggle CPU | ~10 min |
| 7 | `scripts/sub1_clean.py`: US/India from v6cd, France rows from the LB-0.988 file, official validator, zip | CPU | ~3 min |
| | **Total (3a and 3b in parallel)** | | **≈ 7 h** |

Expected checkpoints: step 5 gives val F0.5 ≈ 0.9900 (ours 0.99007: US 0.98995, India 0.99025) and a gate vs v6ce of
≈ +0.0028. Step 7 changes vs the 0.988 file: about +6.9k / −5.8k US/India pairs, France identical. Validator PASS.
Neural training is seeded (`SEED = 0` in the kernels), but GPU non-determinism means the exact pairs can differ slightly
from run to run. The val score should reproduce to about ±0.0002.

## Commands (per step)
```
# 1
python artifacts/final2/restore.py                                   # from repo root (branch final2)
cd code/business_entity_resolution/src && ER_WORK=../../../work_v6 python -m er.run prep
# 2
python dhawal_sub1/scripts/rep_export.py                             # edit the R/W paths at the top
# 3-4: upload rep/{ce_data,bi_data,dense_data} as Kaggle datasets and push the kernel folders
python dhawal_sub1/scripts/kgn.py 1 <parent> datasets create -p ce_data
python dhawal_sub1/scripts/kgn.py 1 <parent> kernels push -p <kernel folder>
python dhawal_sub1/scripts/rep_watch.py                              # optional: chains 3 -> 4 and downloads dense_out
# 5
python dhawal_sub1/scripts/dense_variants.py replica
# 6-7
python dhawal_sub1/scripts/stage_sub1.py  &&  push kaggle/sub1_assemble  &&  python dhawal_sub1/scripts/sub1_clean.py
```
Kaggle account names in `kernel-metadata.json` files are ours. Change the `id` / `dataset_sources` owner to your
account. Tokens are never stored in the code: the Kaggle CLI reads `~/.kaggle/access_token`.

## Licences / limits
xlm-roberta-base (MIT), multilingual-e5-small (MIT), LightGBM (MIT). All models are well under 8B parameters. No
external data lookups.
