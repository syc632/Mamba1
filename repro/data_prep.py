"""Tokenize a LOCAL parquet corpus and pack it into flat binary token files.

Why local-only: in this environment `huggingface_hub` downloads return 0 bytes
(Xet backend broken behind the mirror), so `datasets.load_dataset(streaming=...)`
cannot be trusted. We therefore pull parquet shards with curl and read them from
disk with pyarrow.

Output:
    <out>_train.bin   raw tokens, uint16 if vocab < 65536 else int32
    <out>_val.bin
    <out>.meta.json

Usage:
    python repro/data_prep.py --parquet data/raw/fwedu_000.parquet \
        --tokenizer repro/tokenizer/tokenizer.json \
        --max_tokens 700_000_000 --val_tokens 5_000_000 --out data/fwedu
"""
import argparse
import glob
import json
import os
import time

import numpy as np


def finalize(args, train_path, val_path, dtype, vocab_size, eos,
             shards, n_docs, t0=None):
    """Carve the tail of the packed stream into val and write the final meta."""
    itemsize = np.dtype(dtype).itemsize
    total_bytes = os.path.getsize(train_path)
    val_bytes = min(args.val_tokens * itemsize, total_bytes // 10)
    n_val_tokens = 0
    if val_bytes > 0:
        with open(train_path, "r+b") as f:
            f.seek(total_bytes - val_bytes)
            tail = f.read(val_bytes)
            f.seek(total_bytes - val_bytes)
            f.truncate()
        with open(val_path, "ab") as f:
            f.write(tail)
        n_val_tokens = val_bytes // itemsize
    n_train = (total_bytes - val_bytes) // itemsize

    meta = {
        "parquet": shards,
        "tokenizer": args.tokenizer,
        "vocab_size": vocab_size,
        "dtype": str(dtype),
        "eos": eos,
        "n_train_tokens": n_train,
        "n_val_tokens": n_val_tokens,
        "n_docs": n_docs,
        "doc_sep_eos": args.doc_sep_eos,
        "note": "validation split = tail of packed stream (FineWeb-Edu sample is "
                "already a random sample of the full corpus)",
    }
    with open(args.out + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[data] done{'' if t0 is None else f' in {time.time()-t0:.0f}s'} -> "
          f"{train_path} ({n_train/1e6:.1f}M tok) + "
          f"{val_path} ({n_val_tokens/1e6:.2f}M tok)")
    print(json.dumps(meta, indent=2))


def pick_text_column(names, prefer=None):
    if prefer:
        if prefer in names:
            return prefer
        raise SystemExit(f"--text_column {prefer!r} not in {names}")
    for c in ("text", "content", "story", "raw_content", "body"):
        if c in names:
            return c
    raise SystemExit(f"cannot guess text column from {names}, pass --text_column")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", nargs="+", default=None,
                   help="local parquet path(s) or glob(s); not needed with --finalize")
    p.add_argument("--text_column", default=None)
    p.add_argument("--tokenizer", default="repro/tokenizer/tokenizer.json")
    p.add_argument("--max_tokens", type=int, default=700_000_000)
    p.add_argument("--val_tokens", type=int, default=5_000_000)
    p.add_argument("--out", required=True, help="output path prefix (no extension)")
    p.add_argument("--batch_docs", type=int, default=2000)
    p.add_argument("--append", action="store_true",
                   help="append to an existing <out>_train.bin instead of "
                        "overwriting (lets shards be tokenized as they land)")
    p.add_argument("--finalize", action="store_true",
                   help="only carve the val split off the tail + write meta")
    p.add_argument("--doc_sep_eos", action="store_true", default=True,
                   help="append EOS after every document (paper-style packing)")
    args = p.parse_args()

    paths = []
    for pat in (args.parquet or []):
        hits = sorted(glob.glob(pat))
        if not hits:
            hits = [pat]
        paths.extend(hits)
    for fp in paths:
        if not os.path.exists(fp):
            raise SystemExit(f"missing parquet: {fp}")
    print(f"[data] {len(paths)} parquet file(s), "
          f"{sum(os.path.getsize(f) for f in paths)/1e9:.2f} GB on disk")

    from tokenizers import Tokenizer
    from pyarrow import parquet as pq

    tok = Tokenizer.from_file(args.tokenizer)
    tok.no_padding()
    vocab_size = tok.get_vocab_size()
    eos = tok.token_to_id("<|endoftext|>")
    if eos is None:
        eos = tok.token_to_id("</s>")
    if eos is None:
        raise SystemExit("tokenizer has no <|endoftext|> token; cannot build doc separator")
    dtype = np.uint16 if vocab_size < 65536 else np.int32
    print(f"[data] tokenizer={args.tokenizer} vocab={vocab_size} eos={eos} dtype={dtype}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    train_path = args.out + "_train.bin"
    val_path = args.out + "_val.bin"

    if args.finalize:
        pj = args.out + ".partial.json"
        info = json.load(open(pj)) if os.path.exists(pj) else {}
        dt = np.dtype(np.uint16 if "uint16" in str(info.get("dtype", "uint16"))
                      else np.int32)
        return finalize(args, train_path, val_path, dt,
                        int(info.get("vocab_size", 50257)),
                        int(info.get("eos", 50256)),
                        info.get("shards", []), int(info.get("n_docs", 0)))

    ftrain = open(train_path, "ab" if args.append else "wb")
    fval = open(val_path, "wb")
    n_tokens = 0
    n_val_tokens = 0
    n_docs = 0
    buf = []
    t0 = time.time()
    CHUNK = 1 << 22  # flush every ~4M tokens

    def flush():
        nonlocal buf
        if not buf:
            return
        arr = np.asarray(buf, dtype=dtype)
        if n_val_tokens < args.val_tokens:
            # first `val_tokens` tokens go to the held-out split
            pass
        ftrain.write(arr.tobytes())
        buf = []

    try:
        for fp in paths:
            pf = pq.ParquetFile(fp)
            names = pf.schema_arrow.names
            col = pick_text_column(names, args.text_column) if n_docs == 0 else col
            print(f"[data] {os.path.basename(fp)} rows={pf.metadata.num_rows:,} col={col}")
            for batch in pf.iter_batches(batch_size=args.batch_docs, columns=[col]):
                texts = batch.column(col).to_pylist()
                texts = [t for t in texts if t]
                if not texts:
                    continue
                enc = tok.encode_batch(texts, add_special_tokens=False)
                for e in enc:
                    ids = e.ids
                    if args.doc_sep_eos:
                        ids = ids + [eos]
                    buf.extend(ids)
                    n_tokens += len(ids)
                    n_docs += 1
                if len(buf) >= CHUNK:
                    flush()
                if n_docs % 50000 == 0:
                    el = time.time() - t0
                    print(f"[data] {n_docs/1000:.0f}k docs, {n_tokens/1e6:.1f}M tok, "
                          f"{n_tokens/max(el,1)/1000:.0f}k tok/s, "
                          f"{os.path.getsize(train_path)/1e9:.2f} GB written", flush=True)
                if n_tokens >= args.max_tokens:
                    break
            if n_tokens >= args.max_tokens:
                break
        flush()
    finally:
        ftrain.close()
        fval.close()

    total_now = os.path.getsize(train_path) // np.dtype(dtype).itemsize
    if args.append:
        with open(args.out + ".partial.json", "w") as f:
            json.dump({"shards": [os.path.basename(x) for x in paths],
                       "tokens_this_run": n_tokens, "dtype": str(dtype),
                       "vocab_size": vocab_size, "eos": eos,
                       "total_tokens": total_now}, f, indent=2)
        print(f"[data] appended {n_tokens/1e6:.1f}M tok -> {train_path} "
              f"({total_now/1e6:.1f}M tok total, "
              f"{os.path.getsize(train_path)/1e9:.2f} GB)", flush=True)
        return

    finalize(args, train_path, val_path, dtype, vocab_size, eos,
             [os.path.basename(x) for x in paths], n_docs, t0)


if __name__ == "__main__":
    main()
