# AZ_ML_Challenge — Business Entity Resolution (Amazon ML Challenge 2026)

Matches each Source-1 business record to every Source-2 / Source-3 record describing the
same real-world business. The records are noisy, come from multiple scripts and have no
shared identifiers. The metric is **macro F0.5 per Source-1 entity**, singletons included.

> Living document: updated at every milestone with what was built, why, and the measured
> effect. The latest numbers are in [Results log](#results-log).

## 1. Data facts that drive the design (EDA on train)

| Fact | Consequence |
|---|---|
| Train S1/S2/S3 = 2.21M / 5.03M / 5.29M; test = 1.73M / 4.89M / 5.08M | Everything is vectorised, chunked, run per country and checkpointed to Parquet |
| **Every S2/S3 record matches at most one S1** (7.64M pairs = 7.64M unique targets) | Global target-side assignment constraint; "competition" features |
| 5.6% of S1 are singletons; most S1 have 2–6 matches | Per-entity set selection by expected F0.5, not a single global threshold |
| No cross-country matches | Blocking runs inside each `country` value, which also handles the unseen **France** automatically |
| Only 14.5% of matched names are equal after light normalisation | Heavy normalisation plus fuzzy and phonetic features |
| Names in 8 Indic scripts, domains/hashtags as names, legal-suffix shuffles, typos, accent injection, name swaps | Script folding + transliteration, junk stripping, legal-form canonicalisation, address-only retrieval path |
| No PIN codes in India, ZIP in ~10% of US records, ~3.4% null target addresses | Postal codes are not relied on; address comparison is built from components |
| No leakage in IDs or row order | Clean ML problem |

## 2. Pipeline

```
raw TSV ─► prep (normalise, parallel) ─► blocking (IDF multi-key, per country × source)
       ─► pair features + context/competition features ─► LightGBM (+CatBoost, cross-encoder)
       ─► calibration ─► target-side 1-to-1 constraint ─► per-entity expected-F0.5 selection
       ─► matching_results.tsv / candidate_pairs.tsv (validated)
```

### 2.1 Normalisation (`src/er/text.py`, `src/er/normalize.py`, `src/er/prep.py`)
- **Script folding:** all Brahmic scripts (Devanagari, Bengali, Gurmukhi, Gujarati, Odia, Tamil, Telugu, Kannada, Malayalam) share the same Unicode layout. They are mapped onto one transliteration table, which handles inherent-vowel and schwa deletion and keeps the final "a" after a conjunct (आदित्य → `aditya`). Latin accents are stripped with NFKD. `anyascii` is the fallback for anything else.
- **Names:** text after `|` is dropped (`| www.x.com`), domain syntax is removed, dotted initialisms are collapsed (`L.L.C.` → `llc`), `&` becomes `and`, repeated words are removed, and legal forms are canonicalised for US, India and France (`private/pvt`, `limited/ltd`, `llc`, `sarl`, `sas`, `eurl`, …). The legal forms are split out from the **core** tokens.
- **Phonetic skeleton** (`skeleton`): a consonant skeleton that ignores vowels, aspiration and b/v, so a transliteration matches its English spelling (`praivet` ≈ `private` → `prvt`).
- **Addresses:** split into comma parts. A whole part is recognised as a state (US and Indian names and abbreviations → `st_<name>`). Street-type abbreviations are canonicalised (St/Street/Saint → `st`, R./Rue → `rue`, …) and markers are dropped (H.No, Door No, #, PO Box, …). The parser extracts:
  - **nums** (leading zeros stripped);
  - **hno**, the first non-ordinal number;
  - **loc**, alphabetic tokens from parts without digits, taken from the *end* of the address, where city and district sit;
  - **street**, alphabetic tokens from parts with digits.
- Runs in 20 worker processes, and each worker builds its own Polars DataFrame chunk. Output: `work/{split}_s{1,2,3}.parquet`.

### 2.2 Candidate generation (`src/er/blocking.py`)
IDF-weighted multi-key blocking runs per country × target source. Each record emits hashed keys from 9 families:

| Family | Key | Catches |
|---|---|---|
| `n` | core name token | distinctive names |
| `s` | phonetic skeleton | typos, transliterations |
| `np` | pair of core tokens | common-word names |
| `nl` | core token × locality | generic names in one city |
| `nx` | 4-char prefix × locality | suffix typos, concatenations |
| `a1` | number × street token | name swaps, domains as names |
| `a2` | number × locality | short addresses |
| `a3` | street × locality | missing house numbers |
| `a4` | number pair | "1728 865"-style addresses |

Keys that occur more than 400 times in the target source are dropped. A pair's score is the sum of the IDF weights of the keys it shares, kept separately for the name and address families. Per S1, we keep the union of the **top-40 by total score**, the **top-15 by name score** and the **top-15 by address score**. That way, name-only matches (null target address) and address-only matches (name swaps) never have to compete with pairs supported by both.

### 2.3 Multi-stage cascade (memory-bounded, partition-wise)
Everything below runs per (country × target source) partition, so peak memory stays inside 15.5 GB.

| Stage | What it does | Model / algorithm |
|---|---|---|
| **A. Retrieval** | multi-key blocking (above) over **all** S1, as at test time | IDF-weighted inverted index |
| **A'. Cheap scoring** | blocking scores, key-family counts, 3 fast rapidfuzz scores, context and competition features computed over all S1 | vectorised Polars + rapidfuzz `cpdist` |
| **1. Stage-1 ranker** | scores every retrieved pair and prunes to the final candidate set → **`candidate_pairs.tsv`** | LightGBM (MIT), trained **2-fold out-of-fold** so downstream features never see in-fold scores |
| **B. Rich features** | about 150 name, address, context, competition and cluster-coherence features on the pruned set | rapidfuzz, IDF overlap, directional containment, phonetic skeletons |
| **2. Stage-2 matcher** | final pair probability | LightGBM + CatBoost (Apache-2.0) ensemble; optional stacked cross-encoder |
| **Calibration** | out-of-fold isotonic regression, tuned per source (S2/S3) | scikit-learn |
| **Decision** | target-side exclusivity, then per-entity Bayes-optimal F0.5 set selection | see N1, N2 below |

**Features** (stage B):
- **Name:** ratio, partial, token-sort, token-set, WRatio, Jaro-Winkler and Levenshtein on the full/core/skeleton name. Also first-token similarity, IDF-weighted Jaccard and directional containment, legal-form agree/conflict, and name frequency (genericness).
- **Address:** equality and log-difference of house numbers, digit edit distance, numeric-set overlap, locality/street/state agreement, full-address fuzzy scores, IDF-weighted address overlap, missing-component flags.
- **Context:** rank and gap within the S1's list.
- **Competition:** rank and gap of this S1 among all S1 competing for the same target.
- **Retrieval provenance:** key families hit, per-family ranks.
- **Country:** deliberately **not** a feature, so the model transfers to the unseen France.

### 2.4 Novel, data-specific strategies (research-backed)
- **N1 — Capacity-1 target competition.** We verified that each S2/S3 record belongs to at most one S1, so this is a one-to-many resolution with a hard constraint on the target side (the one-to-one-structure literature, e.g. *Improving ER with Global Constraints*, uses the same idea). This helps in two ways:
  - (a) "competition" features: how this S1 compares with the best other S1 wanting the same target;
  - (b) a final assignment step that gives each target only to its highest-probability S1, with a margin.
- **N2 — Bayes-optimal F0.5 set selection.** Macro F0.5 is decided per entity. Following the F-measure-maximiser theory (Dembczyński et al., *An Exact Algorithm for F-Measure Maximization*; Waegeman et al., JMLR 2014), each S1's prediction set is chosen to maximise **expected F0.5**. The inputs are calibrated per-candidate marginals and an explicit P(no match) from an entity-level "has-match" model. The candidate count is small (≤40), so the maximisation is exact, not a global threshold.
- **N3 — Triangulated cluster coherence (collective ER).** A true cluster has an S1, several S2 duplicates and several S3 duplicates that all resemble each other. For each candidate we compute its similarity to the S1's *other* high-scoring candidates, including cross-source S2↔S3 agreement. In round 2 we re-feed out-of-fold round-1 probabilities of sibling candidates (iterative collective stacking). This builds on the transitivity idea in collective ER.
- **N4 — Noise-channel alias mining.** Aliases are learned from training positive pairs by token alignment, using only the provided data:
  - transliteration aliases (`praivet → private`),
  - native-script and abbreviated state names,
  - city aliases (`calcutta ↔ kolkata`).

  The mined aliases are then fed back into normalisation.
- **N5 — Triangulated pseudo-labelling for unseen France.** A France pair becomes a pseudo-positive only when S1–S2, S1–S3 **and** S2–S3 all agree as mutual best matches with high probability. These pseudo-pairs are used to (i) mine French aliases (region ↔ department, street-type variants) and (ii) adapt the booster. The procedure is first validated by simulating India as the unseen country (leave-one-country-out).
- **N6 — Dual-budget compound-key TF-IDF blocking.** *Sparkly* (VLDB 2023) shows top-k TF-IDF blocking beats most learned blockers. We extend it with compound keys (token × locality, number × street, …) and separate name and address budgets.
- **N7 — Ditto-style cross-encoder on the ambiguous band.** Ditto (VLDB 2021) frames matching as sequence-pair classification with domain-knowledge markers. We fine-tune a small MIT-licensed multilingual encoder (`multilingual-e5-small`) on a free GPU, score only the pairs whose calibrated probability is uncertain, and stack its score into stage 2.

**References:** [Sparkly (VLDB'23)](https://www.vldb.org/pvldb/vol16/p1507-paulsen.pdf) · [Ditto (VLDB'21)](https://arxiv.org/abs/2004.00584) · [Exact F-measure maximisation](https://www.researchgate.net/publication/228532504_An_Exact_Algorithm_for_F-Measure_Maximization) · [Bayes-optimality of F-measure maximisers (JMLR'14)](https://jmlr.org/papers/volume15/waegeman14a/waegeman14a.pdf) · [ER with global constraints](https://arxiv.org/pdf/1108.6016) · [Collective ER in relational data](https://dl.acm.org/doi/pdf/10.1145/1217299.1217304)

### 2.5 Noise-generator inversion and learned aliases (`src/er/noise.py`, `src/er/aliases.py`, `src/er/auxfit.py`)
The corruptions are synthetic, so we learn the generator from train positives and undo it. Only the provided data is used.

| Learned component | What it captures | Result |
|---|---|---|
| **d/b/a marker split** | 105k S3 names (2% of S3) have the form `<pseudo-word> d/b/a|f/k/a|formerly: <real name>`. We keep the real name. Real names such as "DBA Organisers" or "Aka Technologies" are never split | unit-tested on real formats |
| **Name-replacement detector** | whole names replaced by syllable-built pseudo-words ("Korbelo", "Pyranovi") at the same address. Hashed char-n-gram logistic regression; S1 names of every split, including France, serve as label-free clean negatives | held-out AUC **0.9986** |
| **Filler tokens** | words the generator inserts ("service", "shri", "smt", "district"). Only tokens that are *common* in clean S1 names count as filler; rare ones (e.g. "rayal" = royal) carry identity | 11 filler tokens |
| **Token aliases** | transliterations (`kansalting→consulting`, `payoniyar→pioneer`), OCR-style digit swaps (`g1obal→global`, `h0tel→hotel`), address typos (`houstn→houston`, `5st→5th`). Tokens are aligned within positive pairs by phonetic skeleton + Jaro-Winkler, then filtered by frequency and purity | 722 name / 182 address aliases |

These give the matcher the following features:
- alias-mapped, filler-free name similarity (`ca_*`);
- alias-mapped address similarity (`aa_*`);
- the replacement score (`nrep_2`).

### 2.6 Cluster coherence (collective ER, README N3)
Each candidate is compared with the S1's best stage-1 candidate in the **other** source and in its **own** source. One business's S2 and S3 duplicates resemble each other, while distractors do not. Features: `coh_c_*`, `coh_a_*`, `coh_p_*`. Self-comparisons are masked.

### 2.7 Memory-bounded execution
Blocking all 2.2M train S1 (required for faithful competition features) needs ~39 GB if done naively. The pipeline instead streams everything:
- **Target keys:** built one family at a time. Document frequencies are counted over *all* targets, then keys that no S1 has are pruned.
- **Blocking:** candidates are processed in 50k-S1 chunks. Each chunk is featurised and written to disk immediately; per-S1 window features are exact because a chunk holds all of its S1's pairs.
- **Competition features:** computed from **streaming per-target aggregates** (count, max, runner-up) joined back chunk by chunk. They are exact, and memory scales with #targets, not #pairs.
- **Stage 1:** scored in two streaming passes. **Stage B:** only the records that occur in candidate pairs are loaded. **train2 / predict:** feature columns are read lazily and scored in chunks.
- **Resumability:** every stage writes `_SUCCESS` markers, so an interrupted run resumes where it stopped.

### 2.8 Compute decision (AWS analysis)
We evaluated moving heavy runs to AWS EC2:
- **Why it helps:** the pipeline is memory-bound, and a 64–512 GB instance would remove all subsampling.
- **Compliance:** only plain EC2/S3 compute is allowed. AWS Entity Resolution and Amazon Location Service are *excluded* because they violate the no-external-lookup rule.
- **Prepared:**
  - a private S3 bucket;
  - an S3-scoped instance role;
  - a self-terminating spot runner (`aws/launch.sh`, `aws/bootstrap.sh`) that resumes from checkpoints and hard-caps its runtime.
- **Status:** the account is on the AWS Free plan and the quota raise is pending, so the final pipeline runs locally with the memory-bounded design above.

**Status:** normalisation ✅ · blocking ✅ · stage A/1/B ✅ · stage 2 + calibration + expected-F0.5 decision ✅ · submission writer + official validator ✅ · full-data run 🚧

## 3. Engineering practices
- **Reproducible:** pinned `requirements.txt`, fixed seeds, deterministic S1 role split (`work/train_roles.parquet`).
- **Checkpointed:** every stage writes Parquet under `work/` and skips work already done, so it can resume after interruption.
- **Tested:** 42 `pytest` unit tests. They cover transliteration, normalisation, the address parser, blocking keys, the exact metric (including singleton rules), noise/alias mining, pair and coherence features, and the submission writer (format, determinism, invariants). There is also an **end-to-end regression test** (`pytest -m slow`): it builds a mini dataset from real records (`er.minidata`) and runs every stage. It then asserts that the official validator passes and that F0.5 stays above a floor. It has already caught three integration bugs before full-data runs, including a dtype upcast that would have crashed stage B.
- **Validated per layer:** blocking recall by country and key family, calibration curves, feature ablations, and a breakdown of errors by type. Every output file is checked with `utils/validate_submission.py`.
- **Versioned:** every milestone is committed to this repository.
- **Compliant:** no external data or lookups. The lexicons contain only general abbreviations. Models are MIT/Apache-2.0 licensed.

## 4. How to run
```bash
python -m venv .venv && .venv/Scripts/pip install -r code/business_entity_resolution/requirements.txt
cd code/business_entity_resolution/src
python -m er.run all --run v1    # prep -> aux -> stage_a -> stage1 -> stage_b -> train2 -> predict
                                 # resumable; writes output/matching_results.tsv + candidate_pairs.tsv
python -m er.run train2 --run v2 # retrain matcher + re-tune decision only (reuses cached stages)
cd .. && python -m pytest -q     # unit tests   (python -m pytest -m slow  -> end-to-end on mini data)
```

## Results log
| Date | Change | Metric |
|---|---|---|
| 2026-09-25 | Blocking v1, first-part locality, hno | recall (all candidates kept): US 96.2%, India 87.9% |
| 2026-09-25 | **Submission #1 (v0s1)**: stage-1 LightGBM matcher (31 features: blocking evidence, token-set name/address similarity, house-number equality, context) + tuned expected-F0.5 decision (ratio selector, m0=0.5, empty_bias=1.25, margin=0.05) | **val macro F0.5 0.8964** (US 0.9350, India 0.8382); P 0.934 / R 0.858; singleton F 0.607; stage-1 candidate recall 95.4%; official validator PASS |
| 2026-09-25 | **Submission #2 (v2)**: full stage-2 LightGBM matcher (125 features incl. learned-noise aliases/OCR repair, name-replacement score, cluster coherence, stage-1 score), 2-fold OOF + per-source isotonic calibration, refined expected-F0.5 decision (m0=0.5, empty_bias=2.5, margin=0.1) | **val macro F0.5 0.9724** (US 0.9829, India 0.9566); **P 0.9955** / R 0.9355; **singleton F 0.9828**; test: 1.62M/1.73M S1 matched, empty rate FR 7.1% / IN 7.3% / US 5.8%; official validator PASS; differs from v1 on 41% of S1 |
| 2026-09-25 | **Leakage audit (v2)**: learned-noise models were fit on all train positives (incl. validation). Retraining stage 2 **without** any of those features (ca_*, aa_*, nrep_*) | val F0.5 **0.9710** (P 0.9943, R 0.9337, singleton F 0.9746) vs 0.9724 with them: learned-noise features add +0.0014, so v2's +0.075 lead over v1 is not attributable to leakage |
| 2026-09-25 | **v3**: blocking v4 (full-name key, alphanumeric house-id keys, numeric-address budget, larger budgets, cap 600) + same stage-2 recipe; tuned decision (ratio, m0=0.75, empty_bias=2.5, margin=0.05) | **val F0.5 0.97630** (US 0.98363, India 0.96528); P 0.99527 / **R 0.94391**; singleton F 0.98438. **Acceptance gate vs v2 (paired bootstrap, 2000 resamples, same 150k S1): dF0.5 +0.0039 [0.0036, 0.0042] overall, India +0.0087 [0.0080, 0.0094], US +0.0008 [0.0005, 0.0010] → ACCEPT** |
| 2026-09-25 | **Leaderboard decomposition** (test mix US 38.3% / India 46.8% / France 15.0%): v1 LB 0.86 (val US 0.935, IN 0.838) implies France ~0.73; v2 LB 0.96 (val US 0.983, IN 0.957) implies France ~0.91 | Validation tracks LB for US/India; France is the swing country. 0.98+ requires India ~0.98 (recall) **and** France ~0.97 (adaptation). |
