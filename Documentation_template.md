# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Muse  
**Team Members:** Sruthi Sureshkumar (Team Leader), Thaarini Balagapandian  
**Submission Date:** 2 October 2026

---

## 1. Executive Summary

Team Muse links each Source-1 business to its Source-2 and Source-3 records in three phases:

1. **Candidate Generation.** Three retrieval rounds find possible matches. Rounds 1–2 form the **main candidate
   set**; round 3 adds a separate set of **address-only candidates**.
2. **Matching.** A 92-feature LightGBM matcher scores every candidate pair. Three multilingual transformer
   cross-encoders rescore the pairs the matcher is uncertain about, and a group-aware LightGBM stacker combines
   all the scores.
3. **Decision.** Each record goes only to its best Source-1 entity (exclusivity). Main threshold p ≥ 0.75;
   empty-entity rescue p ≥ 0.5; address-only matches are added only for records no entity has claimed.

| headline figure | value | type |
|---|---|---|
| reported public leaderboard macro F0.5 (final submission) | **0.974127** | public leaderboard |
| macro F0.5, final stacker + rescue rule | **0.9814** (0.98138) | held-out, test-like setting |
| macro F0.5, LightGBM matcher alone | 0.9745 | held-out, test-like setting |
| recall of the main candidate set | 96.6 % | held-out, test-like setting |
| predicted matches (`matching_results.tsv`) | 5,685,592 | test output |
| candidate pairs (`candidate_pairs.tsv`) | 7,386,698 (4.26 per Source-1 entity) | test output |

The pipeline uses only the provided training and test files, plus two openly licensed pretrained transformer
checkpoints fine-tuned on the training data (§10).

---

## 2. Problem Analysis and Data Characteristics

**Task.** For each Source-1 entity, list the Source-2/3 records describing the same business. The metric is
macro F0.5 averaged per Source-1 entity; an entity with no true match scores 1 if its prediction is empty.

| split | Source-1 entities | Source-2 + Source-3 records |
|---|---|---|
| train | ≈ 2.2M | ≈ 10.3M |
| test | 1,732,544 (US 663,106 · India 809,986 · France 259,452) | 9,969,589 |

**Findings from exploratory analysis on the training data:**

- **Cardinality.** Each Source-1 entity has 0–11 true matches (mean 3.46); 5.6 % are singletons. No Source-2/3
  record matches more than one Source-1 entity, which is the basis of the exclusivity rule (§7).
- **Unowned records.** About 26 % of training Source-2/3 records match no Source-1 entity. Test has 5.75 records
  per Source-1 entity against 4.68 in training, implying about 40 % unowned records in test; this motivates the
  test-like retraining (§6.2).
- **Country.** In training, a record's country label always equals its match's. Search is partitioned by
  whatever labels occur, with no list of countries in the code; France (test only) uses the same code path as
  the US and India.
- **Name noise:** legal-suffix variants (Pvt/Private, Ltd/Limited, LLC/L.L.C.); honorifics (Sri, Dr, M/s);
  transpositions and OCR/typing errors (Appare1s); accents; appended generic words; web-style names
  (`stargrandgold.com`); Indian names in Devanagari, Kannada, Tamil, Bengali or Gujarati script; names fully
  replaced by an unrelated token.
- **Address noise:** abbreviations (St/Street/"Saint"); state name ↔ code ↔ native script; reordered or dropped
  components; perturbed house numbers (5425 → 1425); landmark phrases; French formats in test (R./Rue, Bd, N°,
  Bis); empty addresses in 3.3 % of training Source-2/3 records.
- **Hard negatives.** Many unowned records are near-copies of a real entity (same address with a different legal
  form, or a sibling business at the same address), so precision is the main difficulty.

---

## 3. End-to-End Architecture

![Team Muse system architecture](code/business_entity_resolution/src/docs/system_architecture.png)

*Figure 1. End-to-end pipeline; arrow labels show what passes between stages. Full-resolution image:
`code/business_entity_resolution/src/docs/system_architecture.png`.*

| phase | stage | main script(s) | output |
|---|---|---|---|
| — | Normalisation | `normalize.py`, `prep.py` | normalised fields; ≈ 30 hashed blocking keys per record |
| **Candidate Generation** | Round 1: composite keys, top-k retrieval, reverse-rank filter, LightGBM filter (p ≥ 0.01) | `keys.py`, `block.py`, `pairs.py`, `model.py` | round-1 pairs |
| | Round 2: sibling expansion and empty-address pass, LightGBM filter (p ≥ 0.05) | `expand.py`, `round2.py` | new pairs; with round 1, the **main candidate set** (7,364,222 test pairs) |
| | Round 3: address-only retrieval and address LightGBM scorer (p ≥ 0.5) | `addr_pass.py`, `addr_score_v2.py` | **address-only candidates** (22,476 after the decision step's de-duplication) |
| **Matching** | Final 92-feature LightGBM matcher, test-like retrained | `round2.py`, `simulate.py` | p_LightGBM for every main pair |
| | Three cross-encoders on uncertain pairs (0.02 < p < 0.98) | `ce_prep.py`, `ce_prep_sib.py`, `ce_train.py` | three transformer scores per uncertain pair |
| | Group-aware LightGBM stacker | `ce_stack.py` | final probability for every main pair |
| **Decision** | Exclusivity, threshold 0.75, rescue 0.5, address matches for unclaimed records | `final_stack.py` | `matching_results.tsv`, `candidate_pairs.tsv` |

Round 3 is not fully independent: it drops pairs already in the main candidate set, and its scorer uses the main
set's probabilities as context features (Figure 1, "existing pairs + scores"). Address-only pairs do **not** pass
through the cross-encoders or the stacker; the decision step uses their own scorer probability directly.

---

## 4. Data Preparation and Normalisation

Every name and address goes through the same steps in `src/normalize.py`:

| step | what it does |
|---|---|
| Transliteration | Any script → ASCII with **`anyascii`**, an offline character table. |
| Basic cleaning | Lower-casing, `&` → `and`, punctuation split out, mixed tokens split (B20 → 20), ordinals stripped (152nd → 152, 2ème → 2). |
| Abbreviation canonicalisation | Street types and directions unified (street/st/saint → st, road → rd, rue/r → rue, boulevard/bd → blvd, …). |
| Legal suffixes and honorifics | Legal forms (Pvt, Ltd, LLC, Inc, …) and honorifics removed from the **name core**; the legal form is kept as a separate feature field. |
| Phonetic skeleton | Per token: ph → f, sh → s, w → v, z → s, c → k, m → n before a consonant; then the leading vowel and remaining vowels are dropped and repeats collapsed. "universal" and the transliterated "yunivrsl" both become `nvrsl`. |

All abbreviation, legal-suffix and honorific tables are hand-written rules in `normalize.py`; no external
dictionary or lookup service is used.

---

## 5. Candidate Generation / Blocking

### 5.1 Round 1: composite keys, retrieval, reverse-rank filter, LightGBM filter

- **Blocking keys** (`keys.py`), about 30 per record. Atomic keys: name-skeleton tokens `N`, concatenated-name
  prefixes `P`, address numbers `D`, address-word skeletons `W`. **Composite keys:** `N×N`, `N×D`, `P×D`, `D×W`,
  `P×W`, `N×W`; each token alone is common, but the combinations are rare. All keys are crc32-hashed into 2^26
  buckets; buckets occurring in more than 2000 Source-2/3 records are ignored, so every inverted list has at most
  2000 entries.
- **Retrieval** (`block.py`), per country label. `sparse_dot_topn` computes the top-k of
  Source-1 × (Source-2 ∪ Source-3)ᵀ, with IDF-weighted Source-1 vectors and binary Source-2/3 vectors. Two passes:
  **sum-IDF score**, top-20 per Source-1 entity; **containment score** (shared IDF / max(the record's total IDF,
  30)), top-10 per entity, which finds short records such as those with an empty address.
- **Reverse-rank filter.** A pair is kept only if the Source-1 entity is among the record's top-2 retrievers in
  either pass (reverse rank ≤ 2).
- **Stage-1 LightGBM filter** (`model.py`): 250 trees, 62 features (49 pairwise similarity + 13 blocking-context).
  Pairs with p < 0.01 are removed, keeping 99.98 % of the true pairs that reach the filter.

### 5.2 Round 2: sibling expansion and empty-address pass

Code: `expand.py`, `round2.py`.

- **Seeds** are round-1 pairs with stage-1 p ≥ 0.9.
- **Sibling expansion.** Each entity's query is re-run with its own keys plus its seeds' keys (sum top-10,
  containment top-5). This finds fragments resembling an already-matched fragment more than the Source-1 record,
  such as native-script names and truncated addresses.
- **Empty-address pass.** The entity's original keys are matched against empty-address Source-2/3 records only
  (containment top-3).
- **Candidate rule and filter.** New pairs are kept if the record ranks the entity in its top-3; a dedicated
  LightGBM filter (400 trees) then keeps pairs with p ≥ 0.05.

**Main candidate set** = round-1 + round-2 pairs:

| stage (training data, held-out) | candidates per Source-1 entity | recall of true pairs |
|---|---|---|
| sum top-20 only | 20.0 | 94.6 % |
| + containment top-10, reverse rank ≤ 2 | 8.58 | 95.3 % |
| + round-1 LightGBM filter | 3.77 | 95.3 % |
| + round-2 expansion, after its filter | 3.85 | 96.5 % |
| test-like setting, 20 % of entities hidden (`sim/validation_report.json`) | 3.94 | **96.6 %** |

On test, the main candidate set has **7,364,222 pairs** (4.25 per Source-1 entity).

### 5.3 Round 3: address-only retrieval

Code: `addr_pass.py`, `addr_score_v2.py`. **Purpose:** recover true matches whose name is unrecognisable (native
script or fully replaced) but whose address matches.

- **Retrieval.** Address tokens (normalised, stop tokens removed) are IDF-weighted and L2-normalised, per
  country. Only tokens found in ≤ 3000 records are used to retrieve, so city names never drive the search. The
  exact cosine is then computed on the full vectors and the top-5 per entity with cosine ≥ 0.5 are kept. Pairs
  already in the main candidate set are dropped.
- **Address LightGBM scorer (v2, used for the submission):** 400 trees, trained on held-out training entities
  (whose share of unowned records matches the test set). Features: cosine, its rank and gap, reverse ranks; name
  similarities (fuzzy, skeleton, partial); a native-script flag and house-number agreement; main-set context (the
  entity's confident-match count and best probability); similarity to the entity's best accepted record. Pairs
  with p ≥ 0.5 are kept.
- **On test,** the decision step keeps each record's single best address pair with p ≥ 0.5 and drops pairs
  already in the main set, leaving **22,476 address-only candidates**.

### 5.4 Main set and address-only candidates

| | main candidate set | address-only candidates |
|---|---|---|
| produced by | rounds 1–2 | round 3 |
| test pairs | 7,364,222 | 22,476 |
| scored by | LightGBM matcher → cross-encoders (uncertain pairs) → stacker | address LightGBM scorer only |
| included in `candidate_pairs.tsv` | all pairs | only pairs with p ≥ 0.5 (not the full retrieval pool) |
| decision rule | exclusivity + p ≥ 0.75, then rescue p ≥ 0.5 | p ≥ 0.5, only for records left unclaimed |

**`candidate_pairs.tsv`** is the union of the two: **7,386,698 pairs** (4.26 per Source-1 entity). Every pair in
`matching_results.tsv` is in it (`final_stack.py` asserts this). 37,899 Source-1 entities have no candidate.

---

## 6. Matching Models

### 6.1 Final LightGBM matcher (92 features)

Code: `features.py`, `pairs.py`, `model.py`, `round2.py`.

| feature group | count | examples |
|---|---|---|
| Name similarity | 31 | RapidFuzz ratio / token-set / token-sort / partial on the name core and skeleton; Jaro–Winkler; IDF-weighted token overlap and unexplained weight; legal-form agreement and conflict; non-ASCII and web-name flags |
| Address similarity | 18 | fuzzy ratios; house-number Jaccard, extra numbers, first-number equality, best number similarity; address-word IDF overlap |
| Blocking context | 13 | sum and containment scores, their ranks and reverse ranks, relative scores, candidate counts, Source-3 flag |
| Competition context (from out-of-fold stage-1 probabilities) | 13 | rank within the entity and within the record, best competitor, margin, agreement with the entity's other confident candidates |
| Sibling / group | 9 | best name / address similarity to the entity's confident round-1 matches, shared house number, empty-address flag |
| Round-2 | 7 + flag | expansion and empty-address scores and ranks, reverse rank, round-2 indicator |

The 49 pairwise and 13 blocking-context features make up stage 1's 62. The final list is `FINAL_FEATS` in
`round2.py` (92 entries). **Model:** LightGBM, binary objective, 127 leaves, learning rate 0.05, early-stopped on
the held-out split.

### 6.2 Test-like retraining (20 % hidden Source-1 entities)

`simulate.py` hides a random **20 % of the training Source-1 entities** not used for fitting. Their Source-2/3
records stay in the corpus with no owner, as decoys, raising the share of unowned records from about 26 % to about
41 % (test: an estimated 40 %). Everything after blocking is recomputed in this setting: reverse ranks, the
stage-1 filter, context features, round 2 and the final matcher. The submitted matcher is this retrained model; it
stopped at **2,379 trees** (`best_iteration` in `sim/validation_report.json`).

### 6.3 Cross-encoders on uncertain pairs

Code: `ce_prep.py`, `ce_prep_sib.py`, `ce_train.py`.

- **Which pairs.** Only pairs with **0.02 < p_LightGBM < 0.98** are rescored: **1,378,180 test pairs** (18.7 % of
  the main set). All other pairs keep their LightGBM probability.
- **Input.** `"<S1 name> | <S1 address>" [SEP] "<record name> | <record address>"`, raw text with no
  normalisation, so the models see native scripts directly.
- **Training pairs,** from training entities outside the held-out split: every uncertain round-1 pair; 300,000
  random easy pairs; round-2 positives plus up to 150,000 round-2 negatives.

| tag | checkpoint | licence | parameters | settings (README step 4) |
|---|---|---|---|---|
| `ce` | `distilbert-base-multilingual-cased` | Apache-2.0 | 134M | max length 96, batch 128 |
| `xlmr` | `xlm-roberta-base` | MIT | 278M | max length 96, batch 64, learning rate 2e-5, warm-up 6 % |
| `sib` | `distilbert-base-multilingual-cased`, sibling-aware | Apache-2.0 | 134M | max length 128; the record side also carries the entity's best *other* candidate (`<record> ## <best sibling>`) |

All three: 1 epoch, AdamW (weight decay 0.01), linear warm-up (5 %; 6 % for `xlmr`), gradient clipping 1.0, fp16, BCE
loss, learning rate 3e-5 (2e-5 for `xlmr`). These are the settings of the contest run, the Colab notebook
`src/notebooks/BER_crossencoder.ipynb` on one T4 GPU, which produced the submitted scores; README step 4 uses the
same settings via `ce_train.py`. The contest run had no fixed seed; `ce_train.py` fixes seed 42.

### 6.4 Group-aware LightGBM stacker

Code: `ce_stack.py`.

1. **Logistic blend** of logit(p_LightGBM), the three cross-encoder logits and each product
   logit(p_LightGBM)·logit(p_model), fitted on the held-out entities' uncertain pairs.
2. **Group features** on the blended probability, within the Source-1 entity and within the record: rank and
   candidate count; best, sum and confident count; best competitor.
3. **Source features:** a Source-3 flag; the entity's confident Source-2 and Source-3 counts; the best
   same-source probability; the gap to the entity's top candidate.
4. **Model:** LightGBM, 400 trees, learning rate 0.05, 31 leaves, trained on the held-out uncertain pairs. It
   rescores the uncertain test pairs and writes `scores/stack_test_p.npy`.

---

## 7. Final Decision Logic

`final_stack.py` applies four rules in order:

| # | rule | parameter | effect |
|---|---|---|---|
| 1 | **Exclusivity.** Each Source-2/3 record keeps only the Source-1 entity with its highest stacker probability. | — | one entity per record |
| 2 | **Main threshold.** Keep that pair if p ≥ 0.75. | `--thr 0.75` | main matches |
| 3 | **Empty-entity rescue.** An entity with no match after rule 2 keeps its single best candidate (still subject to exclusivity) if p ≥ 0.5. | `--rescue 0.5` | adds matches for otherwise-empty entities |
| 4 | **Address-only matches.** An address pair (best per record, not in the main set) with p ≥ 0.5 is added **only if its record was not claimed** by rules 1–3. | `--addr_thr 0.5` | adds matches only for unclaimed records |

**Result:** **5,685,592 matches** (3.28 per Source-1 entity); **105,953** Source-1 entities have no predicted
match.

**Threshold selection.** On the held-out split, the matcher's macro F0.5 is flat between 0.65 and 0.75
(0.97451–0.97434 in `sim/validation_report.json`); 0.75 was used for the stacker. On the public leaderboard,
address threshold 0.5 scored slightly higher than 0.7 (0.97405 vs 0.97401).

---

## 8. Validation and Results

**Validation design.**

- **Held-out split:** 200,000 training Source-1 entities (fixed seed), never used to fit the matcher.
- **Test-like conditions:** evaluation runs with 20 % of the other training entities hidden (§6.2).
- **Disjoint evaluation data:** the cross-encoders are trained on entities outside the held-out split; the
  stacker's and address scorer's held-out figures are out-of-fold by entity (5-fold and 4-fold).
- **Leaderboard use:** public submissions were used only to compare a few decision-rule variants and choose the
  final submission. No model was trained on leaderboard feedback.

### 8.1 Held-out results (training data, never used for fitting)

| system | candidates per Source-1 entity | held-out macro F0.5 |
|---|---|---|
| two-stage LightGBM cascade, round 1 only * | 3.76 | 0.9705 |
| + sibling expansion + group features * | 3.98 | 0.9733 |
| + full-competition training | 3.85 | 0.9755 |
| + test-like retraining † | 3.94 | 0.9745 |
| + three cross-encoders (logistic blend) † | 3.94 | 0.981 |
| **+ group-aware stacker with source features, threshold 0.75 + rescue (submitted) †** | 3.94 | **0.9814** (0.98138) |
| address-only round, v2 scorer, at p ≥ 0.5 (4-fold CV gain) † | — | +0.00079 |

† Evaluated with 20 % of Source-1 entities hidden; harder than the rows above, so not directly comparable.
\* Competitor context computed within the evaluation sample only; later rows use stage-1 scores for all training
entities, as on test.

**Held-out recall of the main candidate set (test-like): 96.6 %.**

### 8.2 Reported public leaderboard results

| submission | reported public macro F0.5 |
|---|---|
| LightGBM cascade, threshold 0.65 | 0.953 |
| + threshold 0.80 | 0.956 |
| + test-like retraining + 3 cross-encoders (logistic blend, threshold 0.85) | 0.97137 |
| + address-only round | 0.97229 |
| + group-aware stacker (threshold 0.80) | 0.97334 |
| + stacker threshold 0.75, empty-entity rescue, address scorer v2 | 0.97401 |
| + address threshold 0.5 | 0.97405 |
| **+ stacker source features (final submission)** | **0.974127** |

These are the public leaderboard scores of the team's uploads. The private / final leaderboard score is **not
known** to the team; only the public score (0.974127) is reported.

### 8.3 Error analysis (held-out)

- **False positives:** same name and address with a different legal form; name-only records whose name is shared
  by many businesses; name-replaced records at a neighbour's address.
- **False negatives:** a perturbed house number combined with a legal-form change; native-script names with a
  truncated address; name-replaced records with only a city left.

---

## 9. Reproducibility

### 9.1 Environment and hardware

- **Python 3.11**, every dependency pinned in `requirements.txt`. Core pins: numpy 2.4.4, pandas 3.0.2, pyarrow
  25.0.1, scipy 1.17.1, lightgbm 4.7.0, scikit-learn 1.8.0, rapidfuzz 3.14.6, anyascii 0.3.3, sparse-dot-topn
  1.2.0, torch 2.8.0, transformers 5.17.0, huggingface_hub 1.33.0, tokenizers 0.23.2, safetensors 0.8.0.
- **Hardware used:** CPU steps on 2–4 cores and 8–30 GB RAM (about 6 GB peak); step 4 on one NVIDIA T4 (16 GB)
  on Colab / Kaggle. Total about 6 h CPU plus 3 h GPU.

```
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 9.2 Exact commands

Data goes at `<DATA>/train/*.tsv` and `<DATA>/test/*.tsv`. Run every command from
`code/business_entity_resolution/`:

```
# 1. Normalisation, blocking (rounds 1-2), LightGBM filter + matcher                 CPU, ~3.3 h on 4 cores
python src/run_pipeline.py --data <DATA> --work work --out out_v2
# 2. Retrain everything after blocking with a test-like decoy rate                  CPU, ~1.3 h
python src/simulate.py     --data <DATA> --work work --sim sim --out out_sim
# 3. Package the uncertain pairs for the transformers                               CPU, ~15 min
python src/ce_prep.py      --work work --sim sim --out ce_pkg
python src/ce_prep_sib.py  --work work --sim sim --pkg ce_pkg --out ce_sib
# 4. Three cross-encoders on the uncertain pairs                                    GPU (T4), ~1 h each
python src/ce_train.py --pkg ce_pkg --out scores --tag ce   --model distilbert-base-multilingual-cased
python src/ce_train.py --pkg ce_pkg --out scores --tag xlmr --model xlm-roberta-base --bs 64 --lr 2e-5 --warmup 0.06
python src/ce_train.py --pkg ce_pkg --out scores --tag sib  --model distilbert-base-multilingual-cased --sib ce_sib --maxlen 128
# 5. Round 3: address-only retrieval + its pair scorer                              CPU, ~15 min
python src/addr_pass.py dev  --work work --sim sim --pkg ce_pkg --out addr
python src/addr_pass.py test --work work --sim sim --pkg ce_pkg --out addr
python src/addr_score_v2.py --work work --sim sim --pkg ce_pkg --addr addr
# 6. Group-aware stacker over LightGBM + the three cross-encoders                   CPU, ~2 min
python src/ce_stack.py   --pkg ce_pkg --scores scores --work work
# 7. Final decision -> both submission files                                        CPU, ~2 min
python src/final_stack.py --pkg ce_pkg --scores scores --addr addr --out ../../output
```

Step 7 writes both files to the archive's top-level `output/` folder. Held-out reports go to
`work/validation_report.json`, `sim/validation_report.json` and `scores/<tag>_report.json`.

### 9.3 Determinism

- **Fixed seeds:** the held-out split, the hidden-entity simulation, the crc32 hashing and the LightGBM seeds.
- **`ce_train.py`** fixes the Python, NumPy and PyTorch seeds (`--seed 42`), uses a seeded DataLoader, and enables
  deterministic cuDNN and PyTorch kernels (`CUBLAS_WORKSPACE_CONFIG=:4096:8`).
- **Steps 6–7 are deterministic.** Rerun on the saved outputs of steps 1–5, they regenerate both submitted files
  **byte for byte** (see §9.4).
- **Limits of a full rerun from raw data.** It reproduces the method and its scores, but not necessarily every
  byte: multithreaded LightGBM can differ in the last decimals with a different core count; the original GPU
  fine-tuning ran before the deterministic flags were added; GPU kernels can vary across GPU models and drivers.
- **Independent rerun of steps 1–2** on Kaggle (4 cores) reproduced the held-out macro F0.5 to four decimals
  (0.9755 standard, 0.9745 test-like), with a candidate set within 1 % in size.

### 9.4 Verification kit (CPU-only exact-output check)

An exact-output verification route. It does **not** retrain any model: it reruns only the final two steps
(stacker and decision rules) on saved intermediates from the contest run. A full retraining from raw data follows
§9.2.

`code/business_entity_resolution/src/verification_kit/` holds the saved contest-run outputs of steps 1–5
(candidate pairs with LightGBM probabilities, the held-out truth, the three cross-encoders' scores, the scored
address pairs) and `verify_submission.py`, which reruns steps 6–7 on them and compares SHA-256 checksums with the
submitted files. CPU only, about 2–3 minutes, about 4 GB of RAM, no GPU or internet (see `README_VERIFY.md`). Run
it from the kit folder with the official challenge dataset:

```
python verify_submission.py --data <challenge dataset dir containing test/>
```

It rebuilds the ID lists from `test/test_source{1,2,3}.tsv`, runs `ce_stack.py` and `final_stack.py`, and prints
`MATCH` or `DIFFERENT` for each TSV.

**Result of the team's run:** the kit as packaged in this archive, with the official test files
(`test_source1/2/3.tsv`), in the pinned environment from `requirements.txt`, printed
`matching_results.tsv: MATCH` and `candidate_pairs.tsv: MATCH` in about 1.5 minutes.

### 9.5 Checksums and output validation

```
SHA-256  6d8d4daf7ec9130ae01d85a69b9e0c6f88dcdb0bb4789309f60870f3b98e0d11  output/matching_results.tsv
SHA-256  2fc01359191ad95e7c9f4d94584fd3b8d9751b86a5f7a13b6ddbbf76b6c88fb4  output/candidate_pairs.tsv
```

1. **Checksums:** `sha256sum output/*.tsv` (or `shasum -a 256` on macOS).
2. **Format and IDs:** the challenge's validator, run from `student_resource/`:

   ```
   python3 utils/validate_submission.py --matching <archive>/output/matching_results.tsv \
       --candidate <archive>/output/candidate_pairs.tsv --test-dir dataset/test --check-ids
   ```

The team ran the challenge validator with `--check-ids` on both submitted files; it reported `PASS — no blocking issues found`.

Verified facts about the submitted files: both have 1,732,544 rows, one per test Source-1 entity;
`matching_results.tsv` contains 5,685,592 predicted matches; `candidate_pairs.tsv` contains 7,386,698 candidate
pairs; every predicted match is contained in the candidate set.

---

## 10. Fair-Play, External Resources, and Model Licences

| resource | role | licence |
|---|---|---|
| `distilbert-base-multilingual-cased` (134M parameters) | pretrained weights, fine-tuned on training pairs (`ce`, `sib`) | Apache-2.0 |
| `xlm-roberta-base` (278M parameters) | pretrained weights, fine-tuned on training pairs (`xlmr`) | MIT |
| `anyascii` 0.3.3 | offline Unicode → ASCII transliteration table | ISC |
| LightGBM 4.7.0 | filter, matcher, address scorer and stacker, all trained from scratch | MIT |
| scikit-learn 1.8.0 | logistic-regression blend | BSD-3 |
| RapidFuzz 3.14.6 | string-similarity features | MIT |
| sparse-dot-topn 1.2.0 | sparse top-n retrieval | Apache-2.0 |
| PyTorch 2.8.0 / Transformers 5.17.0 | cross-encoder fine-tuning | BSD-3 / Apache-2.0 |

- **Licences.** Both pretrained models are MIT / Apache-2.0 and far below the 8B-parameter limit. Their weights
  (about 1.5 GB) are downloaded from the public Hugging Face hub by `ce_train.py` and are not bundled; for offline
  use, see the README (`HF_HUB_OFFLINE=1`). The other pinned packages (numpy, pandas, pyarrow, scipy and the
  transitive dependencies) are not annotated with a licence in `requirements.txt`.
- **No external data.** No external business data, registry, geocoding, web lookup or API was used; the only data
  is the provided training and test files. Abbreviation and legal-suffix tables are hand-written rules in
  `normalize.py`.
- **No country hard-coding.** Country partitions are built from the labels present in the data.

---

## 11. Submission Artefacts

| path in `Muse_submission.zip` | content |
|---|---|
| `output/matching_results.tsv` | final matches: 1,732,544 rows, 5,685,592 matched IDs. Identical to the file uploaded to the leaderboard (SHA-256 above). |
| `output/candidate_pairs.tsv` | final candidate / blocking set: 7,386,698 pairs (7,364,222 main + 22,476 address-only), 4.26 per Source-1 entity |
| `code/business_entity_resolution/src/` | complete runnable pipeline (§3 table), plus `docs/system_architecture.png`, `notebooks/` (the Kaggle / Colab notebooks that ran steps 1–4) and `verification_kit/` |
| `code/business_entity_resolution/README.md` | end-to-end reproduction instructions (§9) |
| `code/business_entity_resolution/requirements.txt` | pinned Python 3.11 dependencies; checkpoint names and download instructions are documented in `README.md` |
| `Documentation_template.md` | this document |

`src/ce_blend.py` and `src/addr_score.py` are earlier versions replaced by steps 5–7, kept for reference only.
`src/notebooks/BER_crossencoder.ipynb` also contains superseded blending and threshold experiments (including a
France-specific threshold variant) that the final submission does not use; the final decision is `final_stack.py`.
