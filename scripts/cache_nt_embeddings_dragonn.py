from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from host_aware_predictor.data.dragonn_mpra import find_split_file, load_dragonn_mpra_split
from host_aware_predictor.models.nucleotide_transformer import (
    NucleotideTransformerConfig,
    NucleotideTransformerWrapper,
)


def _slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return value.strip("_") or "model"


def _auto_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cache frozen Nucleotide Transformer embeddings for MPRA-DragoNN splits."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/mpra_dragonn"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/mpra_dragonn/nt_embeddings"),
    )
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    parser.add_argument(
        "--nt-model-name",
        default="InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default=None, help="Example: float16, bfloat16, auto, or omit.")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--sequence-key", default=None)
    parser.add_argument("--target-key", default=None)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    missing_required: list[str] = []
    for split in args.splits:
        try:
            find_split_file(args.data_dir, split)
        except FileNotFoundError as exc:
            missing_required.append(str(exc))
    if missing_required:
        raise FileNotFoundError("\n".join(missing_required))

    model_slug = _slug(args.nt_model_name)
    max_length_slug = args.max_length if args.max_length is not None else "auto"
    output_dir = args.output_dir / model_slug / f"pool-{args.pooling}_layer-{args.layer}_maxlen-{max_length_slug}"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = _auto_device(args.device)
    print(f"Using device: {device}")
    print(f"Embedding output dir: {output_dir}")

    nt_config = NucleotideTransformerConfig(
        model_name_or_path=args.nt_model_name,
        max_length=args.max_length,
        pooling=args.pooling,
        freeze_encoder=True,
        load_mlm_head=True,
        torch_dtype=args.torch_dtype,
        local_files_only=args.local_files_only,
    )
    nt_model = NucleotideTransformerWrapper(nt_config).to(device)
    nt_model.eval()

    manifest: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(args.data_dir),
        "output_dir": str(output_dir),
        "nt_model_name": args.nt_model_name,
        "pooling": args.pooling,
        "layer": args.layer,
        "max_length": args.max_length,
        "torch_dtype": args.torch_dtype,
        "splits": {},
    }

    for split in args.splits:
        split_output_path = output_dir / f"{split}.pt"
        if split_output_path.exists() and not args.force:
            print(f"Skipping existing {split_output_path}; pass --force to overwrite.")
            continue

        split_path = find_split_file(args.data_dir, split)
        data = load_dragonn_mpra_split(
            split_path,
            split=split,
            sequence_key=args.sequence_key,
            target_key=args.target_key,
        )

        print(
            f"{split}: n={data.n_examples}, sequence_key={data.sequence_key}, "
            f"target_key={data.target_key}, targets={data.targets.shape}"
        )

        embeddings = nt_model.embed(
            data.sequences,
            batch_size=args.batch_size,
            layer=args.layer,
            pooling=args.pooling,
        )

        payload = {
            "split": split,
            "source_path": str(data.source_path),
            "sequence_key": data.sequence_key,
            "target_key": data.target_key,
            "sequence_ids": data.sequence_ids,
            "target_names": data.target_names,
            "targets": torch.from_numpy(data.targets).float(),
            "embeddings": embeddings.float().cpu(),
            "sequence_length": data.sequence_length,
            "nt_model_name": args.nt_model_name,
            "pooling": args.pooling,
            "layer": args.layer,
            "max_length": args.max_length,
        }
        torch.save(payload, split_output_path)

        manifest["splits"][split] = {
            "source_path": str(data.source_path),
            "output_path": str(split_output_path),
            "n_examples": data.n_examples,
            "n_targets": data.n_targets,
            "sequence_key": data.sequence_key,
            "target_key": data.target_key,
            "embedding_shape": list(embeddings.shape),
            "target_shape": list(data.targets.shape),
        }

        print(f"saved {split_output_path} shape={tuple(embeddings.shape)}")

    _save_json(output_dir / "manifest.json", manifest)
    print(f"saved {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
