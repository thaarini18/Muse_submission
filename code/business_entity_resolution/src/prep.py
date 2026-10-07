"""Stage 1: read raw TSVs and pre-compute normalised fields (parquet cache)."""
import os
import sys
import time
from multiprocessing import Pool

import pyarrow as pa
import pyarrow.csv as pc
import pyarrow.parquet as pq

from normalize import (to_ascii, name_core_tokens, skeleton, addr_tokens, ADDR_STOP,
                       basic_tokens, HONORIFIC)


def read_tsv(path):
    return pc.read_csv(
        path,
        parse_options=pc.ParseOptions(delimiter="\t", quote_char=False),
        convert_options=pc.ConvertOptions(strings_can_be_null=False,
                                          column_types={"entity_id": pa.string(), "business_name": pa.string(),
                                                        "business_address": pa.string(), "country": pa.string()}),
    )


def process_record(name, addr):
    nascii = to_ascii(name)
    core = name_core_tokens(name)
    skel = [skeleton(t) for t in core]
    atok = addr_tokens(addr)
    # blocking tokens
    bt = set()
    for s in skel:
        if len(s) >= 2:
            bt.add("n" + s)
    concat = "".join(t for t in basic_tokens(name) if t not in HONORIFIC)
    concat_core = "".join(core)
    if len(concat_core) >= 4:
        bt.add("c" + concat_core[:6])
    if len(concat) >= 4:
        bt.add("c" + concat[:6])
    for t in atok:
        if t.isdigit():
            bt.add("d" + t.lstrip("0") if t.strip("0") else "d0")
        elif t not in ADDR_STOP and len(t) >= 3:
            bt.add("w" + skeleton(t))
    return nascii, " ".join(core), " ".join(skel), " ".join(atok), " ".join(sorted(bt))


def _proc_chunk(args):
    names, addrs = args
    out = [process_record(n, a) for n, a in zip(names, addrs)]
    return list(zip(*out)) if out else [[], [], [], [], []]


def prep_file(src, dst, workers=2, chunk=25000):
    """Streaming: read TSV in record batches, normalise in a process pool, append to parquet."""
    t0 = time.time()
    reader = pc.open_csv(
        src,
        read_options=pc.ReadOptions(block_size=16 << 20),
        parse_options=pc.ParseOptions(delimiter="\t", quote_char=False),
        convert_options=pc.ConvertOptions(strings_can_be_null=False,
                                          column_types={"entity_id": pa.string(), "business_name": pa.string(),
                                                        "business_address": pa.string(), "country": pa.string()}),
    )
    writer = None
    n = 0
    with Pool(workers, maxtasksperchild=20) as p:
        for batch in reader:
            names = batch.column(1).to_pylist()
            addrs = batch.column(2).to_pylist()
            jobs = [(names[i:i + chunk], addrs[i:i + chunk]) for i in range(0, len(names), chunk)]
            cols = [[], [], [], [], []]
            for res in p.imap(_proc_chunk, jobs):
                for c, r in zip(cols, res):
                    c.extend(r)
            tb = pa.Table.from_batches([batch])
            for nm, c in zip(["nm_ascii", "nm_core", "nm_skel", "ad_tok", "btok"], cols):
                tb = tb.append_column(nm, pa.array(c, pa.string()))
            if writer is None:
                writer = pq.ParquetWriter(dst, tb.schema, compression="zstd")
            writer.write_table(tb)
            n += len(tb)
    writer.close()
    print(f"prep {src} -> {dst}: {n} rows in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    data_dir, cache_dir, split = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(cache_dir, exist_ok=True)
    for s in (1, 2, 3):
        src = os.path.join(data_dir, split, f"{split}_source{s}.tsv")
        prep_file(src, os.path.join(cache_dir, f"{split}_s{s}.parquet"))
