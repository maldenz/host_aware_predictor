#!/usr/bin/env python3
"""Generate one NTv3 DNA embedding file per sequence name.

Input:
    data/processed/joint_library_sequences.tsv

Expected columns:
    name
    sequence_230nt

Processing:
    - remove first 15 nt and last 15 nt
    - keep middle 200 nt
    - generate frozen NTv3 pooled embedding
    - save one .pt file per sequence name

Outputs:
    data/processed/dna_emb/by_name/{name}.pt
    data/processed/dna_emb/dna_embedding_index.tsv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

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


def normalize_sequence(seq: str) -> str:
    return "".join(str(seq).upper().replace("U", "T").split())


def crop_middle_200(seq_230nt: str) -> str:
    seq = normalize_sequence(seq_230nt)

    if len(seq) != 230:
        raise ValueError(
            f"Expected 230 nt before cropping, got {len(seq)}. "
            f"Prefix={seq[:30]!r}"
        )

    cropped = seq[15:-15]

    if len(cropped) != 200:
        raise ValueError(f"Expected 200 nt after cropping, got {len(cropped)}")

    return cropped


def safe_filename(name: str) -> str:
    """Make sequence name safe as a filename while keeping it readable."""
    name = str(name).strip()
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    if not name:
        raise ValueError("Empty sequence name after filename sanitization.")
    return name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="data/processed/joint_library_sequences.tsv",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/dna_emb",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
    )
    parser.add_argument(
        "--pooling",
        default="mean",
        choices=["mean", "cls", "first"],
    )
    parser.add_argument(
        "--model-dir",
        default="external/NTv3/NTv3_100M_pre",
    )
    parser.add_argument(
        "--base-model-dir",
        default="external/NTv3/ntv3_base_model",
    )
    parser.add_argument(
        "--torch-dtype",
        default=None,
        choices=[None, "float32", "float16", "bfloat16"],
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate embeddings even if per-name .pt files already exist.",
    )
    parser.add_argument(
        "--max-filename-stem-length",
        type=int,
        default=180,
        help=(
            "Skip sequence names whose sanitized filename stem exceeds this length. "
            "This avoids OSError: [Errno 36] File name too long."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = find_repo_root(Path.cwd())
    add_src_to_pythonpath(repo_root)

    from host_aware_predictor.external.nucleotide_transformer import (
        NucleotideTransformerConfig,
        NucleotideTransformerEncoder,
    )

    input_path = (repo_root / args.input).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    by_name_dir = output_dir / "by_name"

    output_dir.mkdir(parents=True, exist_ok=True)
    by_name_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    df = pd.read_csv(input_path, sep="\t")

    required_cols = {"name", "sequence_230nt"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}. "
            f"Observed columns: {list(df.columns)}"
        )

    df = df[["name", "sequence_230nt"]].copy()
    df["name"] = df["name"].astype(str)

    if df["name"].duplicated().any():
        dupes = df.loc[df["name"].duplicated(), "name"].head(20).tolist()
        raise ValueError(f"Duplicate names found. Example duplicates: {dupes}")

    df["sequence_200nt"] = df["sequence_230nt"].map(crop_middle_200)
    df["safe_name"] = df["name"].map(safe_filename)
    df["safe_name_length"] = df["safe_name"].str.len()

    df["skip_reason"] = ""
    too_long_mask = df["safe_name_length"] > int(args.max_filename_stem_length)
    df.loc[too_long_mask, "skip_reason"] = (
        "safe filename stem longer than "
        + str(args.max_filename_stem_length)
        + " characters"
    )

    df["embedding_path"] = ""
    valid_mask = df["skip_reason"].eq("")
    df.loc[valid_mask, "embedding_path"] = df.loc[valid_mask, "safe_name"].map(
        lambda x: str((by_name_dir / f"{x}.pt").relative_to(repo_root))
    )

    skipped_long_names_path = output_dir / "skipped_long_dna_names.tsv"
    skipped_df = df.loc[
        df["skip_reason"].ne(""),
        [
            "name",
            "safe_name",
            "safe_name_length",
            "skip_reason",
            "sequence_200nt",
        ],
    ].copy()
    skipped_df.to_csv(skipped_long_names_path, sep="\t", index=False)

    embed_df = df.loc[valid_mask].copy().reset_index(drop=True)

    if embed_df.empty:
        raise ValueError(
            "No DNA sequences left after filename-length filtering. "
            f"Skipped names were written to: {skipped_long_names_path}"
        )

    device = torch.device(args.device)

    nt_encoder = NucleotideTransformerEncoder(
        NucleotideTransformerConfig(
            model_dir=args.model_dir,
            base_model_dir=args.base_model_dir,
            pooling=args.pooling,
            freeze_encoder=True,
            require_weights=True,
            torch_dtype=args.torch_dtype,
        )
    ).to(device)

    nt_encoder.eval()

    names = embed_df["name"].tolist()
    safe_names = embed_df["safe_name"].tolist()
    seqs = embed_df["sequence_200nt"].tolist()

    written = 0
    skipped = 0

    with torch.no_grad():
        for start in range(0, len(embed_df), args.batch_size):
            end = min(start + args.batch_size, len(embed_df))

            batch_names = names[start:end]
            batch_safe_names = safe_names[start:end]
            batch_seqs = seqs[start:end]

            pending_indices = []
            pending_seqs = []

            for local_idx, safe_name in enumerate(batch_safe_names):
                out_path = by_name_dir / f"{safe_name}.pt"

                if out_path.exists() and not args.overwrite:
                    skipped += 1
                    continue

                pending_indices.append(local_idx)
                pending_seqs.append(batch_seqs[local_idx])

            if pending_seqs:
                out = nt_encoder(pending_seqs)

                if out.pooled_embedding is None:
                    raise RuntimeError("NTv3 returned no pooled embedding.")

                embeddings = out.pooled_embedding.detach().cpu().contiguous()

                for emb_row, local_idx in enumerate(pending_indices):
                    name = batch_names[local_idx]
                    safe_name = batch_safe_names[local_idx]
                    seq_200nt = batch_seqs[local_idx]
                    out_path = by_name_dir / f"{safe_name}.pt"

                    torch.save(
                        {
                            "name": name,
                            "safe_name": safe_name,
                            "sequence_200nt": seq_200nt,
                            "embedding": embeddings[emb_row],
                            "embedding_dim": int(embeddings.shape[1]),
                            "pooling": args.pooling,
                            "source_file": str(input_path.relative_to(repo_root)),
                            "crop": {
                                "remove_front_nt": 15,
                                "remove_back_nt": 15,
                                "input_nt": 230,
                                "output_nt": 200,
                            },
                            "model_dir": args.model_dir,
                            "base_model_dir": args.base_model_dir,
                        },
                        out_path,
                    )

                    written += 1

            print(
                f"[embed] processed {end}/{len(embed_df)} embeddable | written={written} skipped_existing={skipped} skipped_long_names={len(skipped_df)}",
                flush=True,
            )

    index_path = output_dir / "dna_embedding_index.tsv"

    df[
        [
            "name",
            "safe_name",
            "safe_name_length",
            "sequence_200nt",
            "embedding_path",
            "skip_reason",
        ]
    ].to_csv(index_path, sep="\t", index=False)

    print()
    print("[done] Per-name DNA embeddings generated")
    print(f"  input:        {input_path}")
    print(f"  output dir:   {by_name_dir}")
    print(f"  index:             {index_path}")
    print(f"  skipped names:     {skipped_long_names_path}")
    print(f"  total rows:        {len(df)}")
    print(f"  embeddable rows:   {len(embed_df)}")
    print(f"  skipped long name: {len(skipped_df)}")
    print(f"  written:           {written}")
    print(f"  skipped existing:  {skipped}")


if __name__ == "__main__":
    main()
