# Research notes: ideas beyond README N1–N7

**Scope:** techniques *not* already in the README (Sparkly blocking, Ditto cross-encoder, GFM, one-to-one constraint, collective ER, triangulated France pseudo-labels). They are ranked by **expected gain ÷ implementation cost** for our 3-day, CPU-bound setting. Every idea uses only the provided data. Model licenses were checked where relevant.

---

## R1. Noise-generator inversion (data-specific; highest priority)
The corruptions are clearly **synthetic**, drawn from a finite set of operators. Seen in train:
- appended filler words ("Services", "Enterprises", "Co", "Center")
- legal-form moves and abbreviations
- inserted "(France)" / "(India)"
- junk prefixes ("--", "<<", "#")
- domain or hashtag instead of name
- word duplication
- accent injection
- Indic-script names
- random-looking character corruption ("Ibtdeiirofs")
- **full name replacement by a pronounceable gibberish word** ("Lumzeta", "Ariaevoveo", "Jaxrizagild") at the *same* address

Learning the generator from positive pairs turns noise into signal:
1. **Filler-token table.** For each train positive pair, compute `added = tokens(target) − tokens(S1)`. The frequency of a token in `added`, divided by its overall frequency, gives *P(token inserted by noise)*.
   - Down-weight these tokens in IDF overlap and blocking. "services" matching "services" is noise, not evidence.
   - Feature: similarity **after removing explainable filler tokens** (`n_resid_sim`).
2. **Gibberish-replacement detector.** Use a character n-gram language model (order 3–4, add-k), trained on S1 core tokens of the same country, plus out-of-vocabulary (OOV) rate against the S1 vocabulary. A single-token name with no legal form that is fully OOV and has no domain syntax gets `name_replaced=1`.
   - With this flag, the booster learns that name dissimilarity is uninformative and that it must rely on the address.
   - This targets exactly the address-only positives that pairwise name features currently punish.
3. **Operator-explainability score.** Count how many of the known operators are needed to turn S1's normalized name into the target's (0 = identical, 1 = one filler added, …). Few operators ⇒ strong positive. Precision-friendly.
4. **Typo channel.** Align positive pairs that differ by edit distance ≤ 3 and estimate a character substitution/insertion matrix. Feature: log-likelihood of the target given S1 under this channel, a learned stochastic edit distance (cf. discriminative finite-state edit distance, [CRF string edit distance](https://arxiv.org/pdf/1207.1406)).

*Cost:* one pass over ~7.6M positive pairs (sample 1M), plus a few dictionaries. *Transfer to France:* operator classes are language-independent. The filler table needs French entries; mine them from France pseudo-positives (README N5).

## R2. Supervised meta-blocking edge features
[Generalized Supervised Meta-blocking (VLDB'22)](https://www.vldb.org/pvldb/vol15/p1902-gagliardelli.pdf) shows that a classifier on **blocking-graph edge features** prunes candidates far better than raw scores. We already join on keys, so these are almost free:
- **ARCS** = Σ over shared keys of 1/(df_s1(key) · df_t(key))
- **ECBS / JS** = shared keys ÷ (keys of i + keys of j − shared), with log-IDF scaling
- **EJS** = JS · log(|E|/deg(i)) · log(|E|/deg(j)), where deg = number of candidates of the node (`cx_n` and `tx_n` already exist)
- per-family hit counts (`nkeys` per family instead of a single total)

They are stage-1 inputs and improve pruning, i.e. the recall ceiling of `candidate_pairs.tsv`. *Implementation:* in `block()`, also aggregate `(1/df).sum()` and per-family counts. Keys per record come from `make_keys` counts.

## R3. Better calibration: Venn-Abers / Beta instead of plain isotonic
A large 2026 benchmark ([Classifier Calibration at Scale](https://arxiv.org/html/2601.19944)) finds that **Venn-Abers and Beta calibration** most consistently improve log-loss for strong tabular models, while isotonic regression and Platt scaling *can degrade* them.
- This matters for us because the expected-F0.5 selector consumes probabilities directly.
- Fit the calibrators on out-of-fold predictions **per source (S2/S3) × candidate-count bucket**.
- Choose between isotonic, Beta and Venn-Abers by *validation macro F0.5 after the selector*, not by log-loss alone.

## R4. Listwise "Select" framing as an ensemble member
[Match, Compare, or Select? (COLING'25)](https://aclanthology.org/2025.coling-main.8.pdf) shows that choosing among **all candidates of a record at once** beats independent pairwise matching, because the model sees the competitors.
- **Cheap version (CPU):** a LightGBM **`lambdarank`** model grouped by (S1, source), with the same features. It is a second view whose errors are not the same as the binary model's. Blend its (per-group softmax-normalized) score with the binary probability in a logistic stacker on OOF predictions.
- **GPU version:** see R7.

## R5. Covariate-shift correction for zero-shot France (complements N5)
Pseudo-labelling (N5) needs a decent starting model. Two cheap shift defenses that make that starting model better:
1. **Adversarial importance weighting:** train a classifier "France-test pair vs train pair" on stage-B features. Reweight each train row by `p/(1−p)` (clipped) so training looks like France. This is standard covariate-shift correction; [DADER (VLDB'22)](https://www.vldb.org/pvldb/vol15/p3666-fan.pdf) is the deep-learning analogue, aligning feature distributions with MMD or gradient reversal.
2. **Shift-stable feature selection via LOCO:** drop features whose importance flips between the US-only and India-only models, or whose leave-one-country-out gain is negative. Examples are state-agreement and script flags, which do not exist in France in the same form.

[AnyMatch](https://arxiv.org/abs/2409.04073) (fine-tuned GPT-2, MIT) supports the same point: for zero-shot transfer across domains, *selecting diverse, hard training pairs* matters more than model size. So train the France-facing model on a hard-pair-enriched sample.

## R6. Learned character-level name encoder (SC-Block style)
[SC-Block (ESWC'24)](https://arxiv.org/abs/2303.03132) trains a supervised-contrastive encoder so that matching records cluster together.
- **CPU-feasible variant:** a hashed char-trigram bag (2^18 buckets) → 2-layer MLP → 128-d embedding. Train it with InfoNCE on train positives, using in-batch and blocking hard negatives. It takes minutes on CPU with PyTorch.
- **Uses:**
  - (a) a learned name-similarity feature that is robust to transliteration and filler words;
  - (b) an extra **FAISS blocking channel** for typo-heavy names that miss every exact key (e.g. "Red0x", "Heritage Ibtdeiirofs" with a null address).

## R7. ≤8B Apache-2.0 LLM on the hardest band (optional, GPU)
- **Model:** `Qwen2.5-7B-Instruct` is **Apache-2.0** ([license](https://huggingface.co/Qwen/Qwen2.5-7B/blob/main/LICENSE)), so it is compliant.
- **Setup:** run it in 4-bit on a Kaggle T4/P100 with a *listwise select* prompt: the S1 record plus its top candidates, asking which are the same business.
- **Scope:** only S1s whose selector decision is unstable, e.g. the difference in expected F between k and k±1 is below ε. This is where France zero-shot is most likely to break.
- **Output:** feed its selection back as a feature, stacked on a disjoint train fold, or use it as a tie-breaker.
- **Evidence:** 7B models with good prompts are competitive for entity matching ([Entity Matching with 7B LLMs](https://ceur-ws.org/Vol-3931/paper4.pdf)).
- **Cost:** high. Do this only if R1–R5 are done.

## R8. Link-strength categories (from CLIP multi-source clustering)
[CLIP (Saeedi et al., Leipzig)](https://dbs.uni-leipzig.de/research/publications/clustering-approaches-for-multi-source-entity-resolution) categorizes each link as:
- **strong:** mutual best in both directions, within the source;
- **normal:** best in one direction;
- **weak:** neither.

Clusters are made source-consistent by removing weak links first.
- **As a feature:** `link_class` ∈ {0, 1, 2}, computed from the `r1` and `tx_rank_p1` ranks we already produce. It costs one line.
- **As a rule:** a weak link is kept only if its probability exceeds a higher bar. Tune this on validation.

---

### Recommended order
**R1 (1+2)** → **R2** → **R8** → **R3** → **R4** → **R5** → R6 → R7

Every step is accepted only if validation macro F0.5 improves, with the per-country breakdown and US↔India LOCO checked as well.

### Sources
[GSM meta-blocking](https://arxiv.org/abs/2204.08801) · [SC-Block](https://arxiv.org/abs/2303.03132) · [Sparkly](https://www.vldb.org/pvldb/vol16/p1507-paulsen.pdf) · [AnyMatch](https://arxiv.org/abs/2409.04073) · [Match/Compare/Select](https://arxiv.org/pdf/2405.16884) · [DADER](https://dl.acm.org/doi/10.14778/3554821.3554870) · [Calibration at scale](https://arxiv.org/html/2601.19944) · [One-to-one matching for ER (VLDB J.)](https://link.springer.com/article/10.1007/s00778-023-00791-3) · [CLIP / multi-source clustering](https://dbs.uni-leipzig.de/research/publications/clustering-approaches-for-multi-source-entity-resolution) · [GFM](https://proceedings.neurips.cc/paper/2011/file/71ad16ad2c4d81f348082ff6c4b20768-Paper.pdf) · [7B LLM EM](https://ceur-ws.org/Vol-3931/paper4.pdf) · [Qwen2.5 license](https://huggingface.co/Qwen/Qwen2.5-7B/blob/main/LICENSE)
