#!/usr/bin/env python
"""Preflight check for host-aware expression-head training inputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from <repo>/scripts without installing package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from host_aware_predictor.training import (  # noqa: E402
    ElementQuantificationDataset,
    discover_conditions,
    discover_quantification_files,
    load_quantification_table,
    make_records_for_split,
    read_split_names,
)
from host_aware_predictor.training.expression_head_trainer import to_serializable  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--condition", type=str, default=None, help="Deprecated single-condition filter. Prefer --conditions.")
    parser.add_argument("--conditions", type=str, nargs="*", default=None)
    parser.add_argument("--quantification-tsv", type=Path, nargs="*", default=None)
    parser.add_argument("--dna-embedding-dir", type=Path, default=None)
    parser.add_argument("--host-embedding-dir", type=Path, default=None)
    parser.add_argument("--split-dir", type=Path, default=None)
    parser.add_argument("--name-col", type=str, default="name")
    parser.add_argument("--condition-col", type=str, default="condition")
    parser.add_argument("--target-col", type=str, default="log2")
    parser.add_argument("--weight-col", type=str, default=None)
    parser.add_argument("--dna-pooling", choices=("mean", "first", "flatten"), default="mean")
    parser.add_argument("--host-pooling", choices=("mean", "first", "flatten"), default="mean")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed = args.processed_dir
    dna_embedding_dir = args.dna_embedding_dir or processed / "dna_emb" / "by_name"
    host_embedding_dir = args.host_embedding_dir or processed / "host_emb"
    split_dir = args.split_dir or processed / "splits"

    if args.condition is not None and args.conditions is not None:
        raise ValueError("Pass either --condition or --conditions, not both.")
    conditions = [args.condition] if args.condition is not None else args.conditions
    if conditions == []:
        conditions = None

    quantification_tsv = args.quantification_tsv
    if not quantification_tsv:
        discovery_condition = conditions[0] if conditions and len(conditions) == 1 else None
        quantification_tsv = discover_quantification_files(processed, condition=discovery_condition)

    known_conditions = discover_conditions(host_embedding_dir)
    df = load_quantification_table(
        quantification_tsv,
        condition=None,
        known_conditions=known_conditions,
        name_col=args.name_col,
        condition_col=args.condition_col,
        target_col=args.target_col,
        weight_col=args.weight_col,
    )
    if conditions:
        condition_set = {str(condition) for condition in conditions}
        df = df.loc[df[args.condition_col].astype(str).isin(condition_set)].copy()

    reports = {}
    datasets = {}
    for split in ["train", "val", "test"]:
        names = read_split_names(split_dir / f"{split}_names.txt")
        records, report = make_records_for_split(
            df,
            split=split,
            split_names=names,
            dna_embedding_dir=dna_embedding_dir,
            host_embedding_dir=host_embedding_dir,
            name_col=args.name_col,
            condition_col=args.condition_col,
            target_col=args.target_col,
            weight_col=args.weight_col,
            require_dna_embedding=True,
            require_host_embedding=True,
        )
        reports[split] = report.__dict__
        datasets[split] = ElementQuantificationDataset(records, dna_pooling=args.dna_pooling, host_pooling=args.host_pooling)

    dims = {}
    if len(datasets["train"]) > 0:
        dims = {
            "sequence_embedding_dim": datasets["train"].sequence_embedding_dim,
            "host_embedding_dim": datasets["train"].host_embedding_dim,
        }

    summary = {
        "processed_dir": processed,
        "quantification_tsv": quantification_tsv,
        "known_host_embedding_conditions": known_conditions,
        "requested_conditions": conditions,
        "observed_conditions_after_filter": sorted(df[args.condition_col].astype(str).unique().tolist()) if len(df) else [],
        "rows_after_filter": int(len(df)),
        "split_reports": reports,
        "embedding_dims_from_train_first_record": dims,
    }
    print(json.dumps(to_serializable(summary), indent=2))

    if len(datasets["train"]) == 0 or len(datasets["val"]) == 0:
        raise SystemExit("ERROR: train and val splits must both have at least one usable record.")


if __name__ == "__main__":
    main()
