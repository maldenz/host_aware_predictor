from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


ENSEMBL_PREFIXES = ("ENSG", "ENSMUSG")


def _find_column_case_insensitive(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_to_original = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    return None


def _clean_ensembl_id(value: object) -> str:
    value = str(value).strip().upper()
    if "." in value:
        value = value.split(".", 1)[0]
    return value


def _load_gene_map(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None

    gene_map = pd.read_csv(path, sep="\t")

    ensembl_col = _find_column_case_insensitive(
        gene_map,
        ["ensembl_id", "gene_id", "ensembl_gene_id"],
    )
    symbol_col = _find_column_case_insensitive(
        gene_map,
        ["gene_symbol", "gene_name", "symbol", "external_gene_name"],
    )

    if ensembl_col is None or symbol_col is None:
        raise ValueError(
            "GRCh38 gene map must contain Ensembl and symbol columns, e.g. "
            "'ensembl_id' and 'gene_symbol'."
        )

    gene_map = gene_map[[ensembl_col, symbol_col]].copy()
    gene_map.columns = ["ensembl_id", "gene_symbol"]
    gene_map["ensembl_id"] = gene_map["ensembl_id"].map(_clean_ensembl_id)
    gene_map["gene_symbol"] = gene_map["gene_symbol"].astype(str).str.strip().str.upper()
    gene_map = gene_map.dropna().drop_duplicates("gene_symbol")

    return gene_map


def convert_tsv_to_geneformer_h5ad(
    input_tsv: Path,
    output_h5ad: Path,
    *,
    gene_id_column: str | None = None,
    gene_symbol_column: str | None = None,
    count_columns: list[str] | None = None,
    grch38_gene_map: Path | None = None,
    genome: str = "GRCh38",
) -> Path:
    try:
        import anndata as ad
    except ImportError as exc:
        raise ImportError("Install anndata first: pip install anndata") from exc

    df = pd.read_csv(input_tsv, sep="\t")

    if df.empty:
        raise ValueError(f"{input_tsv} is empty.")

    if gene_id_column is None:
        gene_id_column = _find_column_case_insensitive(
            df,
            ["ensembl_id", "gene_id", "gene", "Geneid", "genes", "id"],
        )

    if gene_id_column is None:
        gene_id_column = df.columns[0]

    if gene_symbol_column is None:
        gene_symbol_column = _find_column_case_insensitive(
            df,
            ["gene_symbol", "gene_name", "symbol", "external_gene_name"],
        )

    if gene_id_column not in df.columns:
        raise ValueError(f"gene_id_column={gene_id_column!r} not found in {input_tsv}")

    id_values = df[gene_id_column].astype(str).str.strip()
    id_values_clean = id_values.map(_clean_ensembl_id)

    looks_like_ensembl = id_values_clean.str.startswith(ENSEMBL_PREFIXES).mean() > 0.5

    if looks_like_ensembl:
        df["ensembl_id"] = id_values_clean
    else:
        gene_map = _load_gene_map(grch38_gene_map)
        if gene_map is None:
            raise ValueError(
                f"{input_tsv} does not appear to use Ensembl gene IDs. "
                "Because you said GRCh38, pass a GRCh38 symbol-to-Ensembl map with "
                "--grch38-gene-map data/reference/grch38_gene_map.tsv. "
                "The map should contain columns like gene_symbol and ensembl_id."
            )

        symbol_values = id_values.str.upper()
        df["_gene_symbol_for_map"] = symbol_values
        df = df.merge(
            gene_map,
            left_on="_gene_symbol_for_map",
            right_on="gene_symbol",
            how="inner",
        )

    if gene_symbol_column is not None and gene_symbol_column in df.columns:
        df["gene_symbol"] = df[gene_symbol_column].astype(str)
    elif "gene_symbol" not in df.columns:
        df["gene_symbol"] = df["ensembl_id"]

    identifier_columns = {
        gene_id_column,
        gene_symbol_column,
        "ensembl_id",
        "gene_symbol",
        "_gene_symbol_for_map",
    }
    identifier_columns = {col for col in identifier_columns if col is not None}

    if count_columns is None:
        candidate_columns = [col for col in df.columns if col not in identifier_columns]
        count_columns = [
            col for col in candidate_columns if pd.api.types.is_numeric_dtype(df[col])
        ]

    if not count_columns:
        raise ValueError(
            f"Could not detect expression/count columns in {input_tsv}. "
            "Pass them explicitly with --count-columns."
        )

    missing_count_cols = [col for col in count_columns if col not in df.columns]
    if missing_count_cols:
        raise ValueError(f"Missing count columns in {input_tsv}: {missing_count_cols}")

    expression = df[["ensembl_id", "gene_symbol", *count_columns]].copy()
    expression["ensembl_id"] = expression["ensembl_id"].map(_clean_ensembl_id)

    expression = expression[
        expression["ensembl_id"].notna()
        & expression["ensembl_id"].astype(str).str.startswith(ENSEMBL_PREFIXES)
    ]

    for col in count_columns:
        expression[col] = pd.to_numeric(expression[col], errors="coerce").fillna(0.0)

    expression = (
        expression.groupby("ensembl_id", as_index=False)
        .agg(
            {
                "gene_symbol": "first",
                **{col: "sum" for col in count_columns},
            }
        )
    )

    count_matrix = expression[count_columns].to_numpy(dtype=np.float32).T

    if count_matrix.shape[0] == 0 or count_matrix.shape[1] == 0:
        raise ValueError(f"No usable GRCh38 Ensembl genes/counts found in {input_tsv}")

    obs_names = [
        input_tsv.stem if len(count_columns) == 1 else f"{input_tsv.stem}_{col}"
        for col in count_columns
    ]

    adata = ad.AnnData(
        X=sparse.csr_matrix(count_matrix),
        obs=pd.DataFrame(index=obs_names),
        var=pd.DataFrame(index=expression["ensembl_id"].astype(str).to_numpy()),
    )

    adata.var["ensembl_id"] = expression["ensembl_id"].astype(str).to_numpy()
    adata.var["gene_symbol"] = expression["gene_symbol"].astype(str).to_numpy()

    adata.obs["n_counts"] = np.asarray(count_matrix.sum(axis=1), dtype=np.float64)
    adata.obs["cell_line"] = input_tsv.stem
    adata.obs["source_file"] = input_tsv.name
    adata.obs["genome"] = genome

    if np.any(adata.obs["n_counts"].to_numpy() <= 0):
        raise ValueError(f"{input_tsv} contains one or more zero-count profiles.")

    output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_h5ad)

    print(f"Wrote {output_h5ad}")
    print(f"Profiles: {adata.n_obs}")
    print(f"Genes: {adata.n_vars}")
    print(f"Genome: {genome}")

    return output_h5ad


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert GRCh38 TSV transcriptome quantifications to Geneformer-ready h5ad."
    )
    parser.add_argument("--input-tsv", required=True, type=Path)
    parser.add_argument("--output-h5ad", required=True, type=Path)
    parser.add_argument("--gene-id-column", default=None)
    parser.add_argument("--gene-symbol-column", default=None)
    parser.add_argument(
        "--count-columns",
        default=None,
        help="Comma-separated expression/count columns. If omitted, numeric columns are auto-detected.",
    )
    parser.add_argument("--grch38-gene-map", default=None, type=Path)
    parser.add_argument("--genome", default="GRCh38")

    args = parser.parse_args()

    count_columns = (
        [col.strip() for col in args.count_columns.split(",") if col.strip()]
        if args.count_columns
        else None
    )

    convert_tsv_to_geneformer_h5ad(
        input_tsv=args.input_tsv,
        output_h5ad=args.output_h5ad,
        gene_id_column=args.gene_id_column,
        gene_symbol_column=args.gene_symbol_column,
        count_columns=count_columns,
        grch38_gene_map=args.grch38_gene_map,
        genome=args.genome,
    )


if __name__ == "__main__":
    main()