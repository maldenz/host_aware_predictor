#!/usr/bin/env python3
"""
Process joint_library.csv to keep only:
    - name
    - 230nt sequence column

Input:
    data/raw/joint_library.csv

Output:
    data/processed/joint_library_sequences.tsv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract name and 230nt sequence from joint_library.csv."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/joint_library.csv"),
        help="Input joint library CSV file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/joint_library_sequences.tsv"),
        help="Output TSV file.",
    )
    return parser.parse_args()


def read_joint_library(path: Path) -> pd.DataFrame:
    """
    The file may contain an extra first row like:
        Unnamed: 0, Unnamed: 1, ...

    The real header appears to start with:
        name, category, chr.hg19, ...

    This function detects and skips the bad first row if needed.
    """
    preview = pd.read_csv(path, nrows=3, header=None)

    first_cell_row0 = str(preview.iloc[0, 0]).strip()
    first_cell_row1 = str(preview.iloc[1, 0]).strip()

    if first_cell_row0 == "name":
        return pd.read_csv(path)

    if first_cell_row1 == "name":
        return pd.read_csv(path, skiprows=1)

    raise ValueError(
        "Could not detect header row. Expected a row starting with 'name'. "
        f"First two first-column values were: {first_cell_row0!r}, {first_cell_row1!r}"
    )


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file does not exist: {args.input}")

    df = read_joint_library(args.input)

    if "name" not in df.columns:
        raise ValueError(f"Missing required column 'name'. Columns: {list(df.columns)}")

    sequence_cols = [
        col for col in df.columns
        if str(col).startswith("230nt sequence")
    ]

    if len(sequence_cols) != 1:
        raise ValueError(
            "Expected exactly one column starting with '230nt sequence', "
            f"found {len(sequence_cols)}: {sequence_cols}"
        )

    seq_col = sequence_cols[0]

    processed = df[["name", seq_col]].copy()
    processed = processed.rename(columns={seq_col: "sequence_230nt"})

    processed["name"] = processed["name"].astype(str).str.strip()
    processed["sequence_230nt"] = processed["sequence_230nt"].astype(str).str.strip().str.upper()

    processed = processed.dropna(subset=["name", "sequence_230nt"])
    processed = processed[
        (processed["name"] != "")
        & (processed["sequence_230nt"] != "")
        & (processed["sequence_230nt"] != "NAN")
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(args.output, sep="\t", index=False)

    print(f"[done] wrote {len(processed):,} rows")
    print(f"[input]  {args.input}")
    print(f"[output] {args.output}")
    print(f"[sequence column] {seq_col}")


if __name__ == "__main__":
    main()