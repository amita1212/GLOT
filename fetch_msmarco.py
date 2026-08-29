#!/usr/bin/env python3
"""Fetch the MS MARCO triplets used for the MTEB contrastive stage.

``main.py --task=embedding`` reads ``--train_file`` (default
``./data/msmarco-triplets.jsonl``). That file was never present in this repo,
which is one reason the MTEB numbers were produced without any contrastive
training at all.

The published archive is gzipped JSON-lines of roughly 2.5 GB compressed. We
almost never need all of it: ``--num_train_samples subset`` slices
``train[:20000]``, so this script streams the archive and stops after
``--max_rows`` records by default instead of downloading the whole thing.

Usage
-----
    python fetch_msmarco.py                    # 200k rows -> ./data/msmarco-triplets.jsonl
    python fetch_msmarco.py --max_rows 0       # everything (slow, large)
    python fetch_msmarco.py --force            # re-download over an existing file

Each output line keeps only the fields the collate function reads --
``query`` and ``pos`` -- so the file stays small.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
import urllib.request

URL = (
    "https://huggingface.co/datasets/sentence-transformers/"
    "embedding-training-data/resolve/main/msmarco-triplets.jsonl.gz"
)
DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "data", "msmarco-triplets.jsonl")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--url", default=URL)
    p.add_argument("--max_rows", type=int, default=200_000,
                   help="Stop after this many usable records. 0 means no limit.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite an existing output file.")
    args = p.parse_args()

    if os.path.exists(args.out) and not args.force:
        n = sum(1 for _ in open(args.out, "r", encoding="utf-8"))
        print(f"{args.out} already exists ({n:,} rows). Use --force to re-download.")
        return 0

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + ".partial"

    kept = skipped = 0
    started = time.time()
    print(f"streaming {args.url}")
    try:
        with urllib.request.urlopen(args.url) as resp, \
                gzip.GzipFile(fileobj=resp) as gz, \
                open(tmp, "w", encoding="utf-8") as out:
            for raw in gz:
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError:
                    skipped += 1
                    continue
                # collate_embedding reads ex["query"] and ex["pos"][0]; anything
                # missing those would crash mid-training, so drop it here.
                query, pos = rec.get("query"), rec.get("pos")
                if not query or not pos:
                    skipped += 1
                    continue
                out.write(json.dumps({"query": query, "pos": pos}) + "\n")
                kept += 1
                if kept % 25_000 == 0:
                    print(f"  {kept:,} rows ({time.time() - started:.0f}s)")
                if args.max_rows and kept >= args.max_rows:
                    break
    except KeyboardInterrupt:
        if os.path.exists(tmp):
            os.remove(tmp)
        print("interrupted; partial file removed")
        return 130
    except Exception as exc:  # noqa: BLE001 - report and leave no partial file
        if os.path.exists(tmp):
            os.remove(tmp)
        print(f"download failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if kept == 0:
        os.remove(tmp)
        print("no usable records found; refusing to write an empty file", file=sys.stderr)
        return 1

    os.replace(tmp, args.out)
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"wrote {kept:,} rows ({size_mb:.1f} MB) to {args.out} "
          f"in {time.time() - started:.0f}s; skipped {skipped:,} malformed/incomplete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
