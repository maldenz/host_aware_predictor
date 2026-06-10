#!/usr/bin/env python3
"""Generate frozen Geneformer host transcriptome embeddings.

Input example:
    data/raw/Host/K562.tsv

Expected source columns:
    gene_id
    expected_count

Processing:
    - use expected_count as raw expression counts
    - strip Ensembl version through GeneformerTranscriptomeTokenizer
    - rank-value tokenize transcriptome for Geneformer V2
    - feed input_ids/attention_mask into local GeneformerEncoder
    - save tokenized transcriptome and host embedding

Outputs:
    data/processed/host_emb/K562_geneformer_tokens.pt
    data/processed/host_emb/K562_geneformer_tokens.jsonl
    data/processed/host_emb/K562_geneformer_embedding.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch


def find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for path in [start, *start.parents]:
        if (path / "pyproject.toml").exists() or (path / ".git").exists():
            return path
    return start


def add_src_to_pythonpath(repo_root: Path) -> None:
    src_dir = repo_root / "src"
    if src_dir.exists():
        sys.path.insert(0, str(src_dir))


def clean_host_name(path: Path) -> str:
    return path.stem.replace(" ", "_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="data/raw/Host/K562.tsv",
        help="Host gene quantification TSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/host_emb",
        help="Output directory for host transcriptome tokens and embeddings.",
    )
    parser.add_argument(
        "--host-id",
        default=None,
        help="Host/profile ID. Defaults to input filename stem, e.g. K562.",
    )
    parser.add_argument(
        "--gene-id-column",
        default="gene_id",
        help="Column containing Ensembl gene IDs.",
    )
    parser.add_argument(
        "--count-column",
        default="expected_count",
        help="Raw count column to feed Geneformer tokenizer.",
    )
    parser.add_argument(
        "--model-input-size",
        type=int,
        default=4096,
        help="Geneformer V2 input length.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
    )
    parser.add_argument(
        "--geneformer-repo",
        default="external/Geneformer",
        help="Local Geneformer repo containing geneformer dictionary pkl files.",
    )
    parser.add_argument(
        "--pooling",
        default="mean",
        choices=["mean", "cls", "first"],
        help="Pooling mode for GeneformerEncoder if supported.",
    )

    return parser.parse_args()


def instantiate_geneformer_encoder(
    *,
    pooling: str,
) -> torch.nn.Module:
    """Instantiate local Geneformer encoder.

    This intentionally avoids remote/HF hub use. It expects your local
    host_aware_predictor.models.geneformer wrapper to handle local checkpoint
    loading, just like nucleotide_transformer.py does.
    """
    from host_aware_predictor.external import geneformer as geneformer_module

    if not hasattr(geneformer_module, "GeneformerEncoder"):
        raise AttributeError(
            "Could not find GeneformerEncoder in host_aware_predictor.models.geneformer"
        )

    Encoder = geneformer_module.GeneformerEncoder

    # Most likely project pattern: GeneformerConfig + GeneformerEncoder(config).
    if hasattr(geneformer_module, "GeneformerConfig"):
        Config = geneformer_module.GeneformerConfig

        try:
            config = Config(
                pooling=pooling,
                freeze_encoder=True,
                require_weights=True,
            )
        except TypeError:
            config = Config()

        try:
            return Encoder(config)
        except TypeError:
            return Encoder()

    # Fallback if encoder accepts keyword args directly.
    try:
        return Encoder(
            pooling=pooling,
            freeze_encoder=True,
            require_weights=True,
        )
    except TypeError:
        return Encoder()


def extract_pooled_embedding(output: Any) -> torch.Tensor:
    """Handle likely GeneformerEncoder output shapes/containers."""
    if isinstance(output, torch.Tensor):
        if output.ndim != 2:
            raise ValueError(f"Expected 2D embedding tensor, got shape {tuple(output.shape)}")
        return output

    for attr in ("pooled_embedding", "embedding", "embeddings", "last_hidden_state"):
        value = getattr(output, attr, None)
        if value is None:
            continue

        if not torch.is_tensor(value):
            continue

        if attr == "last_hidden_state":
            if value.ndim != 3:
                raise ValueError(f"last_hidden_state must be 3D, got {tuple(value.shape)}")
            return value.mean(dim=1)

        if value.ndim == 2:
            return value

    if isinstance(output, dict):
        for key in ("pooled_embedding", "embedding", "embeddings"):
            value = output.get(key)
            if torch.is_tensor(value) and value.ndim == 2:
                return value

        value = output.get("last_hidden_state")
        if torch.is_tensor(value) and value.ndim == 3:
            return value.mean(dim=1)

    raise RuntimeError(
        "Could not extract pooled host embedding from GeneformerEncoder output. "
        "Check models/geneformer.py output field names."
    )


def main() -> None:
    args = parse_args()

    repo_root = find_repo_root(Path.cwd())
    add_src_to_pythonpath(repo_root)

    from host_aware_predictor.tokenizers.geneformer_transcriptome import (
        GeneformerTranscriptomeTokenizer,
        GeneformerTranscriptomeTokenizerConfig,
    )

    input_path = (repo_root / args.input).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input TSV does not exist: {input_path}")

    host_id = args.host_id or clean_host_name(input_path)

    df = pd.read_csv(input_path, sep="\t")

    required = {args.gene_id_column, args.count_column}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required column(s): {sorted(missing)}. "
            f"Observed columns: {list(df.columns)}"
        )

    # Keep only the two needed columns.
    # Rename expected_count to the host ID so tokenizer treats this as one profile/cell.
    expr_df = df[[args.gene_id_column, args.count_column]].copy()
    expr_df[args.count_column] = pd.to_numeric(expr_df[args.count_column], errors="coerce")
    expr_df = expr_df.dropna(subset=[args.gene_id_column, args.count_column])
    expr_df = expr_df[expr_df[args.count_column] > 0].copy()
    expr_df = expr_df.rename(columns={args.count_column: host_id})

    if expr_df.empty:
        raise ValueError(f"No positive raw counts found in {input_path}")

    tokenizer = GeneformerTranscriptomeTokenizer(
        GeneformerTranscriptomeTokenizerConfig(
            geneformer_repo=args.geneformer_repo,
            model_version="V2",
            model_input_size=args.model_input_size,
            special_token=True,
            strip_ensembl_version=True,
            uppercase_gene_ids=True,
            collapse_gene_ids=True,
            drop_empty_cells=True,
        )
    )

    tokenized = tokenizer.tokenize_dataframe(
        expr_df,
        orientation="genes_by_cells",
        gene_id_column=args.gene_id_column,
        count_columns=[host_id],
    )

    model_inputs = tokenized.to_model_inputs(
        pad_token_id=tokenizer.config.pad_token_id,
        pad_to_length=args.model_input_size,
        device=args.device,
    )

    encoder = instantiate_geneformer_encoder(pooling=args.pooling).to(args.device)
    encoder.eval()

    with torch.no_grad():
        output = encoder(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
        )

    host_embedding = extract_pooled_embedding(output).detach().cpu().contiguous()

    if host_embedding.shape[0] != 1:
        raise RuntimeError(
            f"Expected one host embedding row for {host_id}, got shape {tuple(host_embedding.shape)}"
        )

    tokens_pt_path = output_dir / f"{host_id}_geneformer_tokens.pt"
    tokens_jsonl_path = output_dir / f"{host_id}_geneformer_tokens.jsonl"
    emb_pt_path = output_dir / f"{host_id}_geneformer_embedding.pt"
    manifest_path = output_dir / f"{host_id}_geneformer_embedding_manifest.json"

    tokenized.save_pt(tokens_pt_path)
    tokenized.save_jsonl(tokens_jsonl_path)

    torch.save(
        {
            "host_id": host_id,
            "embedding": host_embedding,
            "embedding_dim": int(host_embedding.shape[1]),
            "input_ids": model_inputs["input_ids"].detach().cpu(),
            "attention_mask": model_inputs["attention_mask"].detach().cpu(),
            "token_lengths": tokenized.lengths,
            "source_file": str(input_path.relative_to(repo_root)),
            "gene_id_column": args.gene_id_column,
            "count_column": args.count_column,
            "model_input_size": args.model_input_size,
            "pooling": args.pooling,
        },
        emb_pt_path,
    )

    manifest = {
        "host_id": host_id,
        "source_file": str(input_path.relative_to(repo_root)),
        "gene_id_column": args.gene_id_column,
        "count_column": args.count_column,
        "num_positive_input_genes": int(len(expr_df)),
        "num_tokenized_profiles": int(len(tokenized)),
        "token_length": int(tokenized.lengths[0]),
        "model_input_size": int(args.model_input_size),
        "embedding_shape": list(host_embedding.shape),
        "embedding_dim": int(host_embedding.shape[1]),
        "outputs": {
            "tokens_pt": str(tokens_pt_path.relative_to(repo_root)),
            "tokens_jsonl": str(tokens_jsonl_path.relative_to(repo_root)),
            "embedding_pt": str(emb_pt_path.relative_to(repo_root)),
        },
    }

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print()
    print("[done] Host transcriptome embedding generated")
    print(f"  host_id:             {host_id}")
    print(f"  positive genes:      {len(expr_df)}")
    print(f"  token length:        {tokenized.lengths[0]}")
    print(f"  input_ids shape:     {tuple(model_inputs['input_ids'].shape)}")
    print(f"  attention shape:     {tuple(model_inputs['attention_mask'].shape)}")
    print(f"  embedding shape:     {tuple(host_embedding.shape)}")
    print(f"  tokens pt:           {tokens_pt_path}")
    print(f"  tokens jsonl:        {tokens_jsonl_path}")
    print(f"  embedding pt:        {emb_pt_path}")
    print(f"  manifest:            {manifest_path}")


if __name__ == "__main__":
    main()
