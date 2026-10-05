# `vimal_sub1/` — vimal's lane on top of final2 / final2_fr (2026-09-27)

Branch `dhawal-sub1`, created from `final2_fr` (LB 0.988). Everything here builds on Zayaan's `final2` code (`er.*`,
unchanged) and Balaji's France category-swap rule. No data or model files are committed. Kaggle tokens live in
`~/.kaggle/` only.

## 1. Submission #1 (18:00 IST): `SUB1_ours_usin_fr0988`

| Part | Source | Evidence |
|---|---|---|
| **US/India** | **our v6cd replica**: Zayaan's v6cd recipe (bi-encoder dense retrieval + CE scoring + dense stacker), with the bi-encoder and cross-encoder **retrained on our own Kaggle accounts** from his code | val F0.5 **0.99007** (US 0.98995, India 0.99025) vs 0.98983 for Zayaan's v6cd (the US/India inside LB 0.986/0.988); gate vs v6ce +0.00277 [0.00259, 0.00295] |
| **France** | **exactly the LB-0.988 file** (final2 + Balaji's category-swap rule) | LB |

Changes vs the 0.988 file: US/India only, +6,860 / −5,790 pairs. Official validator PASS. Expected LB about 0.988 + 0.0002,
so it may still show as 0.988 at 3 decimals. **LB result: 0.988** (same as final2_fr, as predicted: a +0.0002 gain
does not show at 3 decimals). Takeaway: our US/India is at least as good (no regression), and only gains of 0.001 or
more are visible on the LB.

### Next lever found from the v6cd loss anatomy (`scripts/fn_origin.py`)
After dense retrieval, retrieval misses fell to 1,701 (US 930). The loss is now in scoring: dense retrieval found
**7,995 true val pairs but the dense stacker accepted only 4,337**. The rejected ones get p ≈ 0.02 (US) / 0.08 (India),
because the stacker sees only cos / rank / gap1 / ce. Perfect decision on the existing candidates would give 0.99907.
`kaggle/lex_dense_stacker` adds name / address / house-number / length / missing-address features to the dense stacker
(trained on FIT fold 1), then tune → gate vs our v6cd → held-out halves.

Built by `kaggle/sub1_assemble/sub1_kernel.py` (Kaggle CPU: decision + write_submission + validator `--check-ids`),
then `scripts/sub1_clean.py` (France rows swapped back to the 0.988 file).

### Rejected before submitting: France K=20 dense additions
Our France dense run at K=20 with Zayaan's filter (same house number + street ≥ 70), scored by our dense stacker, left
only **174** pairs that were new vs 0.988 after Balaji's rule. Balaji's label-free `pifit` puts **84% of them as false**
(his removal rule scored 94% false, i.e. correct to remove). So they were not added. France recall through wider
dense retrieval is **not** the lever. France gains come from precision (Balaji's direction).

## 2. How the v6cd replica was built (5 Kaggle accounts, laptop only orchestrates)

| Step | Where | Code |
|---|---|---|
| Export CE / bi / dense inputs with Zayaan's `er.cross export-biencoder`, `er.dense export` | local (light) | `scripts/rep_export.py` |
| CE (xlm-roberta-base, his `ce_train.py`) | `dhawal2209` | `kaggle/ce_train_xlmr` |
| Bi-encoder (multilingual-e5-small, his `bi_train.py`, 1.2M FIT pairs, 89.5 min) | `codelearner00` | `kaggle/bi_train_e5small` |
| Dense K=10 (his `dense.py`, bi lookup pinned to `bi_model/`) | `dhawal2209` | `kaggle/dense_k10_replica` |
| merge → tune → gate (`er.dense merge`, `er.run tune`, `er.compare`) | local, now moved to Kaggle | `scripts/dense_variants.py`, `kaggle/eval_variants` |

The replica came out slightly above the original: cosine cut 0.5854 vs 0.5908, fit1 pairs 2.06M vs 1.82M, val
+0.00024 (India +0.0005). The watcher `scripts/rep_watch.py` chains every hand-off across the accounts (publishes models
as datasets, pushes the next kernel when inputs are ready, downloads outputs). `scripts/kgn.py <acct> <dir> <kaggle args>`
runs the CLI per account (Windows-safe paths, temp on D:).

## 3. Everything we tried today, with the measured result

All val numbers are macro F0.5 on the 150k val S1 (US/India). "Held-out" means the decision was tuned on one half of
val S1 and scored on the other.

| # | Strategy | Result | Verdict |
|---|---|---|---|
| 1 | Decision-layer tuning (grid, rules for sole-candidate targets / empties) | unselected true-rate ≈ p in every bin | already Bayes-optimal |
| 2 | India-specific decision params (`analysis/india_decision.py`) | −0.00006 [−0.00018, +0.00005] | reject |
| 3 | Cluster-coherence re-scorer: target↔target similarity to the S1's other confident candidates (`analysis/coherence_exp.py`) | +0.00003 [−0.00007, +0.00013] | reject (noise) |
| 4 | House-number anatomy: 73% (India) / 81% (US) of decision FNs have a number on one side only; add-rule (`analysis/sibling_loss.py`, `calib_hno.py`) | p calibrated in those slices; rule −0.00005 | reject |
| 5 | TF-IDF char-trigram recall probe (`analysis/recall_probe.py`) | normal misses in top-5: US 63% / India 57%; null-address and Indic ≈ 2–3% | covered by dense retrieval |
| 6 | **5-personality CE ensemble** (3 xlm-r SGDR snapshots + 2 e5-base, augmentation, label smoothing; `kaggle/ce_ensemble_5personality`) | 0.98689 vs 0.98730, −0.0004 [−0.00053, −0.00028]; precision 0.9986 → 0.9981 | reject: smoothing and averaging soften scores and cost precision |
| 7 | France label-free diagnostics (`analysis/france_diag.py`, `france_band.py`) | selections cleaner than US/India; medium-address pairs are chain branches | no fixable defect |
| 8 | France: drop remaining different-house-number pairs (`scripts/fr_hno_profile.py`) | only 1.4% of France pairs; such pairs are true ~99.8% on US/India | reject |
| 9 | **v6cd replica with our retrained bi + CE** | **0.99007** (+0.00024 over Zayaan's v6cd) | **accept → submission #1** |
| 10 | France K=20 dense additions | 174 pairs, pifit 84% false | reject |
| 11 | France "twin" removal (another co-located S1 matches the target's name) (`scripts/fr_twin_probe.py`) | 18 pairs, artefacts | no-op |
| 12 | France same-building name-band removal (`scripts/fr_band_probe.py`) | would drop France to 3.07 matches/S1; Balaji's removals are rule-specific, not a band | not submitted |

### LB progression analysis (`analysis/lb_progression.py`, `results/lb_progression.json`)
What changed between LB-scored files, and what the LB rewarded:

| Step | LB | Change | Pattern of the change |
|---|---|---|---|
| ours val 0.9762 → v6ce | 0.965 → 0.982 | +266k / −60k | US/India recall; France: added same-number pairs, removed different-number (sibling) pairs |
| v6ce → final2 | 0.982 → 0.986 | +59k / −367 | India +45k same-address **renamed** pairs (dense retrieval); France +4k same-number |
| final2 → final2_fr | 0.986 → 0.988 | −17.9k France only | **same building + same number but a one-word category swap** in the name = a different business |

Matches per S1 moved toward the true ~3.46 with every recall step. France's remaining gap is precision on co-located
businesses (Balaji's direction), not recall.

## 4. Still running (results in `results/` when done)
- Dense **K=20 / keep 99%** (`dhawal2209000`) and **K=50 / keep 99%** (`dhawal22092004`): supersets. `kaggle/eval_variants` carves
  K ∈ {10,15,20,30,50} × keep ∈ {95,97,99}%, then merge → tune → gate vs our replica + held-out halves, on Kaggle CPU.
- **Bi-encoder v2 with hard negatives** (837,826 triplets: highest-oof wrong candidate per FIT positive), plus its own dense K=20
  (`codelearner00`). Also e5-base and 2-epoch variants (for later days).
- **CE2** (e5-base, his recipe) and **CE-large** (xlm-roberta-large, MIT, 560M, 1 epoch, single GPU): extra dense-stacker
  features (`scripts/merge_extra.py`, which also adds lexical name / address / house-number features).
- France dense K=50 and K=20 with bi v2: France probes only.

## 5. Rules we followed
- No external data. Country is never a model feature (France keeps the global decision).
- Anything fit on labels is cross-fitted. Nothing ships without a held-out gain with CI > 0 and no country worse.
- France is judged only by LB or Balaji's `pifit`.
- Heavy compute runs on Kaggle. The laptop only orchestrates.

## 6. Reproduce
```
python dhawal_sub1/scripts/rep_export.py            # inputs (Zayaan's exporters)
# push kaggle/<folder> with scripts/kgn.py <acct> <parent> kernels push -p <folder>; rep_watch.py automates the chain
python dhawal_sub1/scripts/dense_variants.py replica  # merge -> tune -> gate (or kaggle/eval_variants on Kaggle)
# kaggle/sub1_assemble -> scripts/sub1_clean.py      # submission #1
```
Paths at the top of each script point at `E:/projects/Amazon ML challenge/.worktrees/...`. Adjust them for another machine.
