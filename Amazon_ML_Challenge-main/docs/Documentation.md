# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** [Date]

---

## 1. Executive Summary
We built a multi-stage, memory-bounded entity-resolution cascade. It works in four steps:
1. **IDF-weighted multi-key blocking**: 11 compound key families across name and address.
2. **Out-of-fold LightGBM pruning ranker**: produces the audited candidate set.
3. **Rich pairwise matcher**: about 150 features, a LightGBM / lambdarank / CatBoost stack, calibrated per source.
4. **Bayes-optimal decision layer**: maximises *expected* per-entity F0.5 exactly, under a verified dataset constraint that every S2/S3 record belongs to at most one S1.

The main innovation is **noise-generator inversion**. The corruption operators in the data are learned from training positives and undone:
- transliteration aliases;
- OCR digit swaps;
- inserted filler words;
- "pseudo-word d/b/a real-name" wrappers;
- whole-name replacement by pseudo-words.

---

## 2. Methodology

### 2.1 Problem Analysis
EDA was done on 2.2M S1, 5.0M S2 and 5.3M S3 train records.

| Finding | Evidence | Consequence |
|---|---|---|
| Every S2/S3 record matches **at most one** S1 | 7,638,365 true pairs, all targets unique | target-side exclusivity + competition features |
| Singletons are rare | 5.6% of S1 have no match; median 3–4 matches | per-entity set selection, not a global threshold |
| Countries never cross | 0 cross-country pairs | blocking inside each country value (open set; handles France) |
| Names are heavily corrupted | only 14.5% of matched names equal after light normalisation | multi-view normalisation + fuzzy/phonetic features |
| Indic scripts | Devanagari, Bengali, Gujarati, Tamil, Telugu, Kannada, Malayalam, Odia names and states | unified Brahmic transliteration |
| d/b/a wrappers | ~105k S3 names `<pseudo-word> d/b/a <real name>` | keep the real name |
| Name replacement | ~4% of positive targets carry a pseudo-word name (e.g. "Korbelo") at the same address | replacement detector + address-only retrieval path |
| OCR swaps | `g1obal`, `c0astal`, `h0tel` | learned aliases + general 0/1/5 repair |
| Weak postcodes | no PIN codes in India, ZIP in ~10% of US records; 3.4% of target addresses null | component-level address features; postcodes never required |
| Generic names | e.g. "primary care group" ×253 | name-frequency features; address decides |
| France | appears only in test (15% of test S1) | country-agnostic features, label-free adaptation, LOCO validation |

### 2.2 Solution Strategy
**Approach Type:** Hybrid. Blocking + cascaded gradient-boosted matchers + a collective (cluster/competition) layer + a Bayes-optimal set decision.
**Core Innovation:** Noise-generator inversion, together with an exact expected-F0.5 decision under a one-owner-per-target constraint.

---

## 3. Candidate Generation (Blocking)
- **Normalisation before keying:**
  - Brahmic-script transliteration (all 9 scripts through one table, with schwa deletion);
  - legal-form canonicalisation;
  - d/b/a split;
  - address parsing into house number / numbers / street / locality (end-anchored) / canonical state.
- **Blocking keys (hashed, 11 families):** core name token, phonetic skeleton, name-token pair, skeleton pair, name×locality, skeleton×locality, 4-char prefix×locality, number×street, number×locality, number pair, and pooled address-token pairs.
- **Scoring:** pairs are scored by summed IDF of shared keys. Keys are frequency-capped (300 for single tokens, 1,500 for compound keys).
- **Dual budgets:** per S1 and source we keep the union of the top-40 by total score, the top-15 by name-only score, the top-20 by address-only score, and the top-10 by pure-name score. Name-only matches (null address) and name-swap matches (address only) therefore never compete with pairs that have both.
- **Meta-blocking edge features (GSM, VLDB'22):** ARCS, per-family key counts, family bitmask.
- **Competition across all S1 (training included):** every S1 is blocked, so "does this target have a better S1?" features match test conditions exactly.
- **Stage-1 pruning:** 2-fold out-of-fold LightGBM on 31 cheap features. It keeps ≤20 candidates per S1 per source with p ≥ 0.002. The output is exactly `candidate_pairs.tsv`.
- **Candidate pairs generated (test):** [pending]
- **How we ensured true matches were not lost:**
  - independent name and address retrieval paths;
  - separate budgets for each path;
  - phonetic keys;
  - pruning thresholds chosen from measured recall.

  Measured on validation: blocking recall [pending], after stage-1 [pending].
- **Scalability:** cost-aware adaptive chunking caps the number of joined rows per chunk. Per-target statistics are streaming aggregates. Peak RAM ≈ 7 GB on the full data.

---

## 4. Matching Model
**Features used (≈150):**
- **Name:** ratio, partial, token-sort, token-set, WRatio, Jaro-Winkler and Levenshtein on the full / core / skeleton name; first-token JW; set Jaccard and directional containment; IDF-weighted overlap and containment; legal-form agreement; name genericness (frequency).
- **Learned-noise features:**
  - similarity after alias mapping, OCR repair and filler removal;
  - address similarity after alias mapping;
  - the name-replacement probability (hashed char n-gram logistic model, held-out AUC 0.9986).
- **Address:** house-number equality, log-difference and digit edit distance; number-set overlap; street / locality / state agreement; full-address fuzzy scores; IDF-weighted address overlap; missing-component flags.
- **Context:** rank / gap / relative score within the S1's candidate list.
- **Competition:** rank, gap and margin of this S1 against every other S1 wanting the same target, from blocking and stage-1 scores. This operationalises the at-most-one-owner constraint.
- **Cluster coherence:** similarity of the candidate to the S1's best candidate from the *other* source and from the *same* source. True S2/S3 duplicates corroborate each other.
- **Excluded by design:** country is **not** a feature, so the model transfers to France.

**Model type:**
- LightGBM binary matcher (2-fold out-of-fold on fit S1).
- **Ensemble:** LightGBM lambdarank (listwise, per S1×source) and CatBoost, combined by a logistic stacker fit on OOF predictions, then per-source isotonic calibration. [adopted/rejected: pending validation]

**Threshold selection method:** there is no single threshold.
1. **Exclusivity:** each target is assigned only to its highest-probability S1, optionally with a margin.
2. **Set selection:** for every S1 we choose the prefix size k (k = 0 means "no match") that maximises **expected** per-entity F0.5. The exact computation uses Poisson-binomial distributions of true matches inside and outside the chosen set, verified against brute force.
3. **Tuning:** the knobs (missed-match mass m0, empty-set bias, exclusivity margin) are tuned on untouched validation S1 against the exact challenge metric.

---

## 5. Results & Error Analysis
- **F0.5 Score (macro, validation, 150k held-out train S1):** [pending], US [pending], India [pending]
- **Leave-one-country-out (France proxy):** [pending]
- **Common false positives (wrong merges):** [pending]
- **Common false negatives (missed matches):** [pending]

---

## 6. Conclusion
[pending]

---

## Appendix

### A. Code Artefacts
The full runnable pipeline is in `code/business_entity_resolution/`: source in `src/er/`, plus `README.md` and a pinned `requirements.txt`. Entry point: `python -m er.run all --run v1`. It is resumable and writes both output files. The official validator runs automatically at the end.

### B. Compliance
- No external data, APIs, geocoding or lookups.
- The lexicons contain only generic abbreviations (legal forms, street types, US/Indian state names).
- All models are MIT / Apache-2.0 / BSD licensed and tiny (≪ 8B parameters).
- AWS Entity Resolution and Location services were deliberately **not** used.

### C. Additional Results
[pending: layer recall table, calibration table, feature importance, ablations]
