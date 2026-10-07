# Business Entity Resolution — reproducible pipeline

Blocking with an inverted index over phonetic and composite keys, a second sibling-expanded retrieval round,
and a LightGBM filter / matcher cascade with group-consistency features.
It uses no external data or lookup services. The models are LightGBM (MIT), trained from scratch, plus three
fine-tuned multilingual transformers (Apache-2.0 / MIT, at most 278M parameters) for uncertain pairs, a group-aware
LightGBM stacker and an address-only retrieval round.

## Layout

```
src/
  run_pipeline.py   entry point: data -> blocking -> matching -> output/*.tsv
  normalize.py      transliteration (anyascii), abbreviation canonicalisation, phonetic skeleton
  prep.py           stage 1: normalised fields per record -> parquet cache
  keys.py           blocking keys (atomic + composite), hashed to 2^26 buckets
  block.py          stage 2: inverted-index top-k retrieval per country (sparse top-n product)
  features.py       pairwise similarity features (49)
  pairs.py          final candidate rule + blocking-context features (13)
  model.py          stage-1 filter model, context features, stage-2 matcher
  expand.py         round-2 retrieval: queries expanded with keys of confident matches + empty-address pass
  round2.py         sibling features, round-2 filter, final matcher layout
  decide.py         exclusivity + threshold -> final match lists
  simulate.py       retrain with a test-like decoy rate (hides 20% of training Source1 entities)
  ce_prep.py        package uncertain pairs for the GPU stage
  ce_prep_sib.py    same, with each pair's best sibling candidate (sibling-aware cross-encoder)
  ce_train.py       GPU: fine-tune + score one multilingual cross-encoder (Apache-2.0 / MIT)
  addr_pass.py      round 3: address-only retrieval (rare address tokens, cosine top-5)
  addr_score.py     LightGBM for address-pass pairs; held-out evaluation; scores the test pairs
  addr_score_v2.py  round-3 scorer retrained on held-out entities (used for the submission)
  ce_stack.py       group-aware LightGBM stacker over LightGBM + cross-encoder scores
  final_stack.py    final decision (threshold, exclusivity, rescue, address pass) -> both TSVs
  ce_blend.py       blend LightGBM + cross-encoder scores (logistic regression), write both TSVs
  common.py         I/O helpers, macro F0.5
```

## Environment

```
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Hardware used: 2–4 CPU cores and 8–30 GB RAM (the code fits in about 6 GB) for the CPU steps, and one T4 GPU (Colab / Kaggle) for step 4.
The step-4 settings below are those of the contest run (`src/notebooks/BER_crossencoder.ipynb`, Colab, one T4 GPU).

## Run (full chain): data → blocking → matching → outputs

The challenge data must be at `<DATA>/train/*.tsv` and `<DATA>/test/*.tsv` (the `dataset/` folder of the
challenge's student resource). Run every command **from this folder** (`code/business_entity_resolution/`).
Intermediate results go to `work/`, `sim/`, `ce_pkg/`, `ce_sib/`, `scores/` and `addr/` inside this folder. Each
step caches its results there, so a rerun resumes. **Step 7 writes both submission files to `../../output/`,
which is the top-level `output/` folder of the submission archive.**

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

- Final outputs: `../../output/matching_results.tsv` and `../../output/candidate_pairs.tsv` (the archive's
  top-level `output/` folder). `candidate_pairs.tsv` is exactly the set of pairs the final models score: the main
  candidates plus the address-pass pairs kept by its scorer (p ≥ 0.5). Every matched ID is inside it.
- Held-out reports: `work/validation_report.json`, `sim/validation_report.json`, `scores/<tag>_report.json`.
  The log of `addr_score_v2.py` prints the address-pass held-out gain.
- `src/notebooks/` holds the Kaggle / Colab notebooks that ran steps 1–4 on the challenge hardware.
  `src/ce_blend.py` (earlier logistic-blend writer) and `src/addr_score.py` (first address scorer) are
  superseded by steps 5–7 and kept for reference only.

## External resources

- **Pretrained model weights**, downloaded automatically from the public Hugging Face hub the first time step 4
  runs:
  - `distilbert-base-multilingual-cased` (Apache-2.0, 134M parameters);
  - `xlm-roberta-base` (MIT, 278M parameters).
  Both are fine-tuned on the training pairs only. The challenge rules allow MIT / Apache-2.0 models of up to 8B
  parameters. The weights (~1.5 GB) are not bundled in the archive.
- **Offline use:** run `huggingface-cli download <model>` once on a machine with internet access, copy the
  cache folder (`~/.cache/huggingface`), and set `HF_HUB_OFFLINE=1`. You can also pass a local model directory
  to `--model`.
- **No other external data, API, database, geocoding or lookup service is used.** All abbreviation tables are
  hand-written rules in `src/normalize.py`, and transliteration uses the offline `anyascii` table.

## Expected outputs (checksums)

The submitted files, which are identical to the leaderboard upload (public macro F0.5 0.974127):

```
SHA-256  6d8d4daf7ec9130ae01d85a69b9e0c6f88dcdb0bb4789309f60870f3b98e0d11  output/matching_results.tsv
SHA-256  2fc01359191ad95e7c9f4d94584fd3b8d9751b86a5f7a13b6ddbbf76b6c88fb4  output/candidate_pairs.tsv
```

Check them with `sha256sum ../../output/*.tsv` (or `shasum -a 256` on macOS).

Validate the format with the challenge's validator, run from the challenge's `student_resource/` folder:

```
python3 utils/validate_submission.py --matching <archive>/output/matching_results.tsv \
    --candidate <archive>/output/candidate_pairs.tsv --test-dir dataset/test --check-ids
```

## Verify the submitted files in 3 minutes (no GPU)

`src/verification_kit/` holds the saved intermediate results of our contest run (outputs of steps 1–5: the
candidate pairs with their final LightGBM probabilities, the three cross-encoders' scores and the address-pass
scores). One command reruns steps 6–7 on them and checks the result against the submitted files:

```
cd src/verification_kit
python verify_submission.py --data <challenge dataset dir containing test/>
```

It prints `MATCH` for both `matching_results.tsv` and `candidate_pairs.tsv` (SHA-256 identical to the
submission), and needs CPU only, about 4 GB of RAM and 2–3 minutes. See `src/verification_kit/README_VERIFY.md`.

## Reproducibility

- **Fixed randomness.** The validation split, the hidden-entity simulation, the crc32 hashing and all LightGBM
  seeds are fixed. `ce_train.py` fixes the Python, NumPy and PyTorch seeds (`--seed 42`), uses a seeded
  DataLoader shuffle, and turns on deterministic cuDNN / PyTorch kernels (`CUBLAS_WORKSPACE_CONFIG=:4096:8`).
- **Steps 6–7 are deterministic.** They were checked to regenerate both submitted files **byte for byte**
  (the checksums above) from the saved outputs of steps 1–5.
- **A full rerun from raw data reproduces the method and its scores, but not necessarily every byte.**
  - Multithreaded LightGBM can differ in the last decimal places with a different CPU core count.
  - The original GPU fine-tuning ran before the deterministic flags above were added, and GPU kernels can
    still vary across GPU models and driver versions.
  - An independent end-to-end rerun of steps 1–2 on Kaggle (4 cores) reproduced the held-out macro F0.5 to
    four decimals (0.9755 standard setting, 0.9745 test-like), with a candidate set within 1 % in size.
- **Runtime used:** Python 3.11; steps 1–3 and 5–7 on 2–4 CPU cores / 8–30 GB RAM (about 6 GB peak); step 4
  on one NVIDIA T4 (16 GB) on Colab / Kaggle. Total about 6 h CPU + 3 h GPU.
