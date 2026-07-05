#!/usr/bin/env python3
"""Compare coverage between the extracted input and evaluation results.

Reads:
  - tmkp_edges_extracted.parquet  (the input)
  - results.db  (evaluations table)

Reports duplicate counts, coverage against the input, and lists any missing
or extra keys.
"""

import argparse
import sqlite3
import sys
from pathlib import Path

import polars as pl


def main():
    parser = argparse.ArgumentParser(
        description="Compare coverage between extracted input and results."
    )
    parser.add_argument(
        "--extracted", "-e",
        default="data/tmkp_kgx/tmkp_edges_extracted.parquet",
        help="Path to the extracted input Parquet file",
    )
    parser.add_argument(
        "--results-db", "-r",
        default="results.db",
        help="Path to the results SQLite database",
    )
    parser.add_argument(
        "--table", "-t",
        default="evaluations",
        help="Evaluations table name (default: evaluations)",
    )
    args = parser.parse_args()

    ext_path = Path(args.extracted)
    db_path = Path(args.results_db)

    if not ext_path.exists():
        sys.exit(f"Error: extracted file not found: {ext_path}")
    if not db_path.exists():
        sys.exit(f"Error: results database not found: {db_path}")

    key_cols = ['subject_curie', 'predicate', 'object_curie', 'supporting_text_id']

    # ---- Extracted input ----
    print("Loading extracted input ...")
    df_ext = pl.read_parquet(str(ext_path), columns=key_cols)
    ext_keys = set(df_ext.iter_rows())
    ext_total = df_ext.height
    ext_unique = len(ext_keys)
    print(f"  Total rows:    {ext_total:,}")
    print(f"  Unique 4-keys: {ext_unique:,}")
    print(f"  Duplicates:    {ext_total - ext_unique:,}")
    del df_ext

    # ---- Evaluations table ----
    print(f"\nLoading {args.table} table ...")
    conn = sqlite3.connect(str(db_path))
    eval_count = conn.execute(
        f'SELECT COUNT(*) FROM "{args.table}"'
    ).fetchone()[0]
    eval_keys = set()
    for row in conn.execute(
        f'SELECT subject_curie, predicate, object_curie, supporting_text_id '
        f'FROM "{args.table}"'
    ):
        eval_keys.add(row)
    conn.close()

    print(f"  Total rows:    {eval_count:,}")
    print(f"  Unique 4-keys: {len(eval_keys):,}")
    print(f"  Duplicates:    {eval_count - len(eval_keys):,}")

    # ---- Cross-check ----
    print(f"\n{'=' * 60}")
    print("Coverage Summary")
    print("=" * 60)
    print(f"  Extracted unique keys:          {ext_unique:,}")
    print(f"  Evaluated unique keys:          {len(eval_keys):,}")

    in_results_not_ext = eval_keys - ext_keys
    in_ext_not_results = ext_keys - eval_keys
    matched = eval_keys & ext_keys

    print(f"\n  In results but NOT in extracted: {len(in_results_not_ext):,}")
    print(f"  In extracted but NOT in results: {len(in_ext_not_results):,}")
    print(f"  Matched:                         {len(matched):,}")

    if ext_unique > 0:
        print(f"  Coverage: {len(matched):,} / {ext_unique:,} = "
              f"{len(matched) / ext_unique * 100:.4f}%")

    if in_ext_not_results:
        print(f"\n  Sample missing keys (up to 10):")
        for k in list(in_ext_not_results)[:10]:
            print(f"    {k}")

    if in_results_not_ext:
        print(f"\n  Sample extra keys (up to 10):")
        for k in list(in_results_not_ext)[:10]:
            print(f"    {k}")

    if not in_ext_not_results and not in_results_not_ext:
        print(f"\n  *** PERFECT MATCH: All rows accounted for ***")

    return 0 if not in_ext_not_results and not in_results_not_ext else 1


if __name__ == "__main__":
    sys.exit(main())
