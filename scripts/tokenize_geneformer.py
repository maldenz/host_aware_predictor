from __future__ import annotations

import argparse
from pathlib import Path

from host_aware_predictor.tokenizers import (
    GeneformerTranscriptomeTokenizer,
    GeneformerTranscriptomeTokenizerConfig,
)


def _optional_path(value: str | None) -> Path | None:
    return Path(value) if value else None


def _parse_custom_attrs(values: list[str]) -> dict[str, str] | None:
    if not values:
        return None

    parsed: dict[str, str] = {}
    for value in values:
        if ":" not in value:
            raise argparse.ArgumentTypeError(
                f"Invalid --custom-attr {value!r}; expected input_obs_col:output_dataset_col."
            )
        source, target = value.split(":", 1)
        parsed[source] = target

    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tokenize raw single-cell transcriptome counts with Geneformer."
    )
    parser.add_argument("--input-path", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("data/processed/geneformer"), type=Path)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--file-format", choices=["h5ad", "loom", "zarr"], default=None)
    parser.add_argument("--geneformer-repo", default="external/Geneformer")
    parser.add_argument("--gene-median-file", default=None)
    parser.add_argument("--token-dictionary-file", default=None)
    parser.add_argument("--gene-mapping-file", default=None)
    parser.add_argument("--model-version", choices=["V1", "V2"], default="V2")
    parser.add_argument("--nproc", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--ensembl-id-column", default="ensembl_id")
    parser.add_argument("--n-counts-column", default="n_counts")
    parser.add_argument("--count-layer", default=None)
    parser.add_argument(
        "--custom-attr",
        action="append",
        default=[],
        help="Preserve cell metadata as input_obs_col:output_dataset_col. Can be repeated.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    model_input_size = 2048 if args.model_version == "V1" else 4096
    special_token = args.model_version == "V2"

    config = GeneformerTranscriptomeTokenizerConfig(
        geneformer_repo=_optional_path(args.geneformer_repo),
        gene_median_file=_optional_path(args.gene_median_file),
        token_dictionary_file=_optional_path(args.token_dictionary_file),
        gene_mapping_file=_optional_path(args.gene_mapping_file),
        model_version=args.model_version,
        model_input_size=model_input_size,
        special_token=special_token,
        nproc=args.nproc,
        chunk_size=args.chunk_size,
        ensembl_id_column=args.ensembl_id_column,
        n_counts_column=args.n_counts_column,
        count_layer=args.count_layer,
        custom_attr_name_dict=_parse_custom_attrs(args.custom_attr),
    )

    tokenizer = GeneformerTranscriptomeTokenizer(config)

    dataset_path = tokenizer.tokenize(
        input_path=args.input_path,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        file_format=args.file_format,
    )

    print(dataset_path)


if __name__ == "__main__":
    main()