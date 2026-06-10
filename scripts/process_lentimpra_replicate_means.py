#!/usr/bin/env python3
"""
Aggregate lentiMPRA element quantification replicates.

Input columns expected:
    condition
    replicate
    name
    dna_count
    rna_count
    ratio
    log2
    n_obs_bc

Output:
    One row per condition + name.
    Removes replicate column.
    Averages numeric columns across available replicates.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


NUMERIC_COLUMNS = [
    "dna_count",
    "rna_count",
    "ratio",
    "log2",
    "n_obs_bc",
]


REQUIRED_COLUMNS = [
    "condition",
    "replicate",
    "name",
    *NUMERIC_COLUMNS,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Average lentiMPRA element quantification values across replicates."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input element quantification TSV file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output processed TSV file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file does not exist: {args.input}")

    df = pd.read_csv(args.input, sep="\t")

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Input file is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    # Keep only required columns, in expected order.
    df = df[REQUIRED_COLUMNS].copy()

    # Convert replicate and numeric columns safely.
    df["replicate"] = pd.to_numeric(df["replicate"], errors="coerce")

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Optional sanity check: only expect replicate 1, 2, 3 if present.
    unexpected_reps = sorted(
        r for r in df["replicate"].dropna().unique().tolist() if r not in [1, 2, 3]
    )
    if unexpected_reps:
        raise ValueError(f"Unexpected replicate values found: {unexpected_reps}")

    # Average across available replicates for each condition + element name.
    # If an element has only 1 or 2 replicates, pandas mean uses available values.
    processed = (
        df.groupby(["condition", "name"], as_index=False, sort=False)[NUMERIC_COLUMNS]
        .mean()
    )

    # Preserve final column order.
    processed = processed[
        [
            "condition",
            "name",
            "dna_count",
            "rna_count",
            "ratio",
            "log2",
            "n_obs_bc",
        ]
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(args.output, sep="\t", index=False)

    print(f"[done] wrote {len(processed):,} rows to {args.output}")
    print(f"[input] {len(df):,} replicate-level rows")
    print(f"[output] {len(processed):,} element-level rows")


if __name__ == "__main__":
    main()