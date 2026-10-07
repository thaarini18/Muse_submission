# Verification kit — regenerate the exact submitted files

`verify_submission.py` reproduces `output/matching_results.tsv` and `output/candidate_pairs.tsv` **byte for
byte** (SHA-256 checked) from the saved intermediate results of our contest run. It runs on CPU only, takes
2–3 minutes, needs about 4 GB of RAM, and requires no GPU or internet.

```
pip install -r ../../requirements.txt     # from code/business_entity_resolution/src/verification_kit/
python verify_submission.py --data <challenge dataset dir containing test/>
```

`artifacts/` holds the contest-run outputs of pipeline steps 1–5 (`code/business_entity_resolution/README.md`):

| file | content | produced by |
|---|---|---|
| `val_pairs.parquet` | held-out candidate pairs: s1, r, label y, final LightGBM probability | steps 1–3 |
| `test_pairs.parquet` | test candidate pairs: s1, r, final LightGBM probability | steps 1–3 |
| `val_truth.parquet`, `val_meta.parquet` | held-out ground-truth pairs and entity list | step 3 |
| `{ce,xlmr,sib}_{val,test}.npy` | scores of the three fine-tuned cross-encoders on the uncertain pairs | step 4 (GPU) |
| `test_addr_scored_v2.parquet` | address-only round pairs with their scorer probability | step 5 |

The ID lists are rebuilt from the provided test files: the row order is the file order, with Source 2 followed
by Source 3. The full pipeline (README, steps 1–7) regenerates these artifacts from the raw data; because GPU
fine-tuning and multithreaded LightGBM are not bit-exact across machines, a full rerun reproduces the scores
within noise, while this kit reproduces the submitted files exactly.
