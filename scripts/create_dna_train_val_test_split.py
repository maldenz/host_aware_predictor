#!/usr/bin/env python3
"""Create train/val/test split for per-name DNA embeddings.

Input:
    data/processed/dna_emb/dna_embedding_index.tsv

Output:
    data/processed/splits/dna_train_val_test_split.tsv
    data/processed/splits/train_names.txt
    data/processed/splits/val_names.txt
    data/processed/splits/test_names.txt

Split:
    train: 80%
    val:   10%
    test:  10%

Only rows with a valid embedding_path are included.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for path in [start, *start.parents]:
        if (path / "pyproject.toml").exists() or (path / ".git").exists():
            return path
    return start


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="data/processed/dna_emb/dna_embedding_index.tsv",
        help="DNA embedding index TSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/splits",
        help="Output directory for split files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic split.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = find_repo_root(Path.cwd())
    input_path = (repo_root / args.input).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input split source does not exist: {input_path}")

    df = pd.read_csv(input_path, sep="\t", keep_default_na=False)

    required_cols = {"name", "embedding_path"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}. "
            f"Observed columns: {list(df.columns)}"
        )

    valid = df["embedding_path"].astype(str).str.len() > 0

    if "skip_reason" in df.columns:
        valid &= df["skip_reason"].astype(str).str.len().eq(0)

    split_df = df.loc[valid].copy().reset_index(drop=True)

    if split_df.empty:
        raise ValueError("No valid rows available for train/val/test split.")

    if split_df["name"].duplicated().any():
        dupes = split_df.loc[split_df["name"].duplicated(), "name"].head(20).tolist()
        raise ValueError(f"Duplicate names found in valid split rows: {dupes}")

    split_df = split_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    n_total = len(split_df)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    n_test = n_total - n_train - n_val

    split_labels = (
        ["train"] * n_train
        + ["val"] * n_val
        + ["test"] * n_test
    )

    split_df["split"] = split_labels

    # Stable final ordering: split first, then original shuffled order inside split.
    split_df.insert(0, "split_index", range(len(split_df)))

    output_split_path = output_dir / "dna_train_val_test_split.tsv"
    manifest_path = output_dir / "dna_train_val_test_split_manifest.json"

    split_df.to_csv(output_split_path, sep="\t", index=False)

    for split_name in ["train", "val", "test"]:
        names = split_df.loc[split_df["split"].eq(split_name), "name"].tolist()
        names_path = output_dir / f"{split_name}_names.txt"
        names_path.write_text("\n".join(names) + "\n", encoding="utf-8")

    counts = split_df["split"].value_counts().to_dict()

    manifest = {
        "source_index": str(input_path.relative_to(repo_root)),
        "output_split_file": str(output_split_path.relative_to(repo_root)),
        "seed": args.seed,
        "ratio": {
            "train": 0.8,
            "val": 0.1,
            "test": 0.1,
        },
        "counts": {
            "total_valid": int(n_total),
            "train": int(counts.get("train", 0)),
            "val": int(counts.get("val", 0)),
            "test": int(counts.get("test", 0)),
        },
        "excluded_rows": int(len(df) - len(split_df)),
    }

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("[done] DNA train/val/test split created")
    print(f"  source:        {input_path}")
    print(f"  output:        {output_split_path}")
    print(f"  manifest:      {manifest_path}")
    print(f"  total valid:   {n_total}")
    print(f"  train:         {counts.get('train', 0)}")
    print(f"  val:           {counts.get('val', 0)}")
    print(f"  test:          {counts.get('test', 0)}")
    print(f"  excluded rows: {len(df) - len(split_df)}")


if __name__ == "__main__":
    main()
