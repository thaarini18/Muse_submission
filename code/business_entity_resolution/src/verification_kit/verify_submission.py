"""Verification kit: regenerate the EXACT submitted files from the saved contest-run intermediates.

    python verify_submission.py --data <dataset dir with test/test_source{1,2,3}.tsv>

What it does (CPU only, ~2-3 min, ~4 GB RAM):
  1. Rebuilds the Source-1 / Source-2+3 ID lists from the provided test files (row order = file order).
  2. Runs src/ce_stack.py   on the saved LightGBM + cross-encoder scores from the contest run (artifacts/).
  3. Runs src/final_stack.py (threshold 0.75, exclusivity, empty-entity rescue, address-pass matches).
  4. Compares SHA-256 of both regenerated files with the submitted ones.
artifacts/ holds the contest-run outputs of pipeline steps 1-5 (see README): val/test candidate pairs with their
final LightGBM probability, the held-out truth, the three cross-encoders' scores and the address-pass scores.
"""
import argparse, hashlib, os, subprocess, sys
import pandas as pd, pyarrow as pa, pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
EXPECTED = {"matching_results.tsv": "6d8d4daf7ec9130ae01d85a69b9e0c6f88dcdb0bb4789309f60870f3b98e0d11",
            "candidate_pairs.tsv": "2fc01359191ad95e7c9f4d94584fd3b8d9751b86a5f7a13b6ddbbf76b6c88fb4"}
ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="challenge dataset dir (contains test/)")
ap.add_argument("--src", default=os.path.join(HERE, ".."))   # this kit lives in src/verification_kit/
ap.add_argument("--out", default=os.path.join(HERE, "regenerated_output"))
a = ap.parse_args()
A = os.path.join(HERE, "artifacts")


def ids(f):
    return pd.read_csv(f, sep="\t", usecols=["entity_id"], dtype=str, keep_default_na=False).entity_id


T = os.path.join(a.data, "test")
w = lambda s, n: pq.write_table(pa.Table.from_pandas(pd.DataFrame({"entity_id": s.values}), preserve_index=False),
                                os.path.join(A, n))
w(ids(f"{T}/test_source1.tsv"), "test_ids.parquet")
w(pd.concat([ids(f"{T}/test_source2.tsv"), ids(f"{T}/test_source3.tsv")], ignore_index=True), "test_r_ids.parquet")
print("1/4 ID lists rebuilt from the test files", flush=True)
run = lambda *args: subprocess.run([sys.executable, *args], check=True)
run(os.path.join(a.src, "ce_stack.py"), "--pkg", A, "--scores", A)
print("2/4 stacker done", flush=True)
run(os.path.join(a.src, "final_stack.py"), "--pkg", A, "--scores", A, "--addr", A, "--out", a.out)
print("3/4 decision rules done", flush=True)
ok = True
for f, h in EXPECTED.items():
    d = hashlib.sha256(open(os.path.join(a.out, f), "rb").read()).hexdigest()
    print(f"4/4 {f}: {'MATCH' if d == h else 'DIFFERENT'}  ({d[:16]}...)")
    ok &= d == h
print("\nRESULT:", "identical to the submitted files (public macro F0.5 0.974127)" if ok else "differs from the submission")
sys.exit(0 if ok else 1)
