#!/usr/bin/env python
"""Train concat, FiLM, query, sequence-only, or future expression heads on precomputed embeddings.

Default multi-condition layout from repository root:

    data/processed/
      host_emb/{condition}/{condition}_geneformer_embedding.pt
      dna_emb/by_name/{name}.pt
      splits/train_names.txt
      splits/val_names.txt
      splits/test_names.txt
      *_element_quantifications.tsv

The regression target is the ``log2`` column by default.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

# Allow running this file directly from <repo>/scripts without installing package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from host_aware_predictor.models.registry import available_head_names  # noqa: E402
from host_aware_predictor.training.expression_head_trainer import run_training  # noqa: E402


def _safe_run_name(value: str) -> str:
    """Return a filesystem-safe run/report name component."""

    value = str(value).strip()
    value = re.sub(r"[^A-Za-z0-9_.=-]+", "_", value)
    value = value.strip("._-")
    if not value:
        raise ValueError("Run name cannot be empty after sanitization.")
    return value


def default_run_name(*, head: str, timestamp_format: str) -> str:
    """Build the default timestamp+head run name used to prefix reports."""

    timestamp = datetime.now().strftime(timestamp_format)
    return _safe_run_name(f"{timestamp}_{head}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Data layout.
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--condition", type=str, default=None, help="Deprecated single-condition filter. Prefer --conditions.")
    parser.add_argument("--conditions", type=str, nargs="*", default=None, help="Conditions to include. Omit to include all conditions present in quantification tables.")
    parser.add_argument("--quantification-tsv", type=Path, nargs="*", default=None, help="Explicit quantification TSV(s). Omit to discover all *_element_quantifications.tsv files.")
    parser.add_argument("--dna-embedding-dir", type=Path, default=None)
    parser.add_argument("--host-embedding-dir", type=Path, default=None)
    parser.add_argument("--split-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None, help="Run/report filename prefix. Default: <timestamp>_<head>.")
    parser.add_argument("--timestamp-format", type=str, default="%Y%m%d-%H%M%S", help="strftime format used when --run-name is omitted.")

    # Table schema.
    parser.add_argument("--name-col", type=str, default="name")
    parser.add_argument("--condition-col", type=str, default="condition")
    parser.add_argument("--target-col", type=str, default="log2", help="Regression target column; this should be the orange log2 column.")
    parser.add_argument("--weight-col", type=str, default=None, help="Optional per-row loss weight, e.g. n_obs_bc. Disabled by default.")

    # Head selection and common head args.
    parser.add_argument("--head", "--head-type", dest="head", choices=available_head_names(), default="concat")
    parser.add_argument("--hidden-dims", type=int, nargs="*", default=(), help="Predictor MLP hidden dimensions. Empty means one linear output layer.")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--activation", choices=("gelu", "relu", "silu", "tanh"), default="gelu")

    # FiLM-specific args. These are ignored by concat and reserved-compatible with future heads.
    parser.add_argument("--fusion-dim", "--film-dim", dest="fusion_dim", type=int, default=256, help="FiLM/query projected sequence dimension.")
    parser.add_argument("--film-hidden-dims", "--modulation-hidden-dims", dest="film_hidden_dims", type=int, nargs="*", default=None, help="Host-to-FiLM MLP hidden dimensions. Omit for [fusion_dim]; pass flag with no values for linear host-to-FiLM.")
    parser.add_argument("--no-film-layer-norm", "--no-film-layernorm", dest="no_film_layer_norm", action="store_true", help="Disable LayerNorm on projected sequence features.")
    parser.add_argument("--film-gamma-scale", type=float, default=1.0)
    parser.add_argument("--film-include-host-skip", action="store_true", help="Concatenate a projected host skip vector after FiLM modulation.")
    parser.add_argument("--no-film-identity-init", action="store_true", help="Do not zero-init the final FiLM generator layer.")
    
    # Query-specific args. These are ignored by concat/FiLM/sequence_only.
    parser.add_argument("--query-num-heads", type=int, default=4, help="Number of attention heads for the query head.")
    parser.add_argument("--query-num-sequence-slots", "--query-num-slots", dest="query_num_sequence_slots", type=int, default=8, help="Number of learned latent sequence slots projected from each pooled sequence embedding.")
    parser.add_argument("--query-num-queries", type=int, default=4, help="Number of host-derived query vectors per sample.")
    parser.add_argument("--query-pooling", choices=("mean", "flatten"), default="mean", help="How to pool query attention outputs before prediction.")
    parser.add_argument("--query-no-layer-norm", action="store_true", help="Disable LayerNorm in the query head.")
    parser.add_argument("--query-no-sequence-skip", action="store_true", help="Disable the direct projected sequence skip path in the query head.")
    parser.add_argument("--query-include-host-skip", action="store_true", help="Concatenate a projected host skip vector after query attention.")

    parser.add_argument("--head-kwargs-json", type=str, default="{}", help="Extra JSON kwargs reserved for future/custom registered heads.")

    # Optimization.
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss", choices=("mse", "huber"), default="mse")
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--standardize-target", action="store_true", help="Train on z-scored target; metrics and predictions are written on original scale.")

    # Embedding loading and runtime.
    parser.add_argument("--dna-pooling", choices=("mean", "first", "flatten"), default="mean")
    parser.add_argument("--host-pooling", choices=("mean", "first", "flatten"), default="mean")
    parser.add_argument("--no-cache-embeddings", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amp", action="store_true", help="Use CUDA autocast mixed precision when CUDA is available.")
    parser.add_argument("--seed", type=int, default=1337)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.run_name is None:
        args.run_name = default_run_name(head=args.head, timestamp_format=args.timestamp_format)
    else:
        args.run_name = _safe_run_name(args.run_name)
    run_training(args)


if __name__ == "__main__":
    main()
