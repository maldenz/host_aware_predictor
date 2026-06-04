"""Raw transcriptome counts -> Geneformer token IDs.

This module is the bridge between host transcriptome data and
models/geneformer.py.

It intentionally uses only local project files:

    external/Geneformer/geneformer/gene_median_dictionary_gc104M.pkl
    external/Geneformer/geneformer/token_dictionary_gc104M.pkl
    external/Geneformer/geneformer/ensembl_mapping_dict_gc104M.pkl

It does not modify external/Geneformer and does not download anything.

Expected model-side output:

    {
        "input_ids": LongTensor[batch, seq_len],
        "attention_mask": LongTensor[batch, seq_len],
    }

which can be passed directly into GeneformerEncoder from
host_aware_predictor.models.geneformer.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import torch

MatrixOrientation = Literal["cells_by_genes", "genes_by_cells"]
GeneformerModelVersion = Literal["V1", "V2"]


@dataclass(frozen=True)
class GeneformerTranscriptomeTokenizerConfig:
    """Configuration for local Geneformer transcriptome tokenization."""

    geneformer_repo: str | Path = Path("external/Geneformer")

    model_version: GeneformerModelVersion = "V2"
    model_input_size: int = 4096
    special_token: bool = True

    target_sum: float = 10_000.0
    pad_token_id: int = 0

    strip_ensembl_version: bool = True
    uppercase_gene_ids: bool = True
    collapse_gene_ids: bool = True

    drop_empty_cells: bool = True

    gene_median_file: str | Path | None = None
    token_dictionary_file: str | Path | None = None
    gene_mapping_file: str | Path | None = None


@dataclass
class TokenizedTranscriptomes:
    """Container returned by GeneformerTranscriptomeTokenizer."""

    input_ids: list[list[int]]
    cell_ids: list[str]
    lengths: list[int]
    metadata: dict[str, list[Any]]

    def __len__(self) -> int:
        return len(self.input_ids)

    def to_model_inputs(
        self,
        *,
        pad_token_id: int = 0,
        pad_to_length: int | None = None,
        device: torch.device | str | None = None,
    ) -> dict[str, torch.Tensor]:
        """Pad token IDs and return tensors accepted by GeneformerEncoder."""
        if not self.input_ids:
            raise ValueError("No tokenized transcriptomes are available.")

        max_length = max(len(ids) for ids in self.input_ids)
        if pad_to_length is not None:
            if pad_to_length <= 0:
                raise ValueError(f"pad_to_length must be positive, got {pad_to_length}")
            max_length = int(pad_to_length)

        input_ids = torch.full(
            (len(self.input_ids), max_length),
            fill_value=int(pad_token_id),
            dtype=torch.long,
        )
        attention_mask = torch.zeros_like(input_ids)

        for row_idx, ids in enumerate(self.input_ids):
            cropped = ids[:max_length]
            length = len(cropped)

            if length == 0:
                continue

            input_ids[row_idx, :length] = torch.tensor(cropped, dtype=torch.long)
            attention_mask[row_idx, :length] = 1

        # Defensive: explicit pad IDs should not contribute to pooled embeddings.
        attention_mask = attention_mask * input_ids.ne(int(pad_token_id)).long()

        if device is not None:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def save_jsonl(self, path: str | Path) -> Path:
        """Save tokenized rows as JSONL for inspection or lightweight reuse."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as handle:
            for idx, input_ids in enumerate(self.input_ids):
                row = {
                    "cell_id": self.cell_ids[idx],
                    "input_ids": input_ids,
                    "length": self.lengths[idx],
                }
                for key, values in self.metadata.items():
                    row[key] = values[idx]
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        return path

    def save_pt(self, path: str | Path) -> Path:
        """Save tokenized payload with torch.save."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "input_ids": self.input_ids,
                "cell_ids": self.cell_ids,
                "lengths": self.lengths,
                "metadata": self.metadata,
            },
            path,
        )

        return path


def _project_root() -> Path:
    # src/host_aware_predictor/tokenizers/geneformer_transcriptome.py -> repo root
    try:
        return Path(__file__).resolve().parents[3]
    except IndexError:
        return Path.cwd()


def _resolve_repo_path(path: str | Path) -> Path:
    path = Path(path).expanduser()

    if path.is_absolute():
        return path.resolve()

    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    return (_project_root() / path).resolve()


def _first_existing_path(package_dir: Path, candidates: tuple[str, ...]) -> Path | None:
    for candidate in candidates:
        candidate_path = package_dir / candidate
        if candidate_path.exists():
            return candidate_path
    return None


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _is_sparse_matrix(value: Any) -> bool:
    try:
        import scipy.sparse as sp
    except ImportError:
        return False

    return bool(sp.issparse(value))


def _row_sum(matrix: Any) -> np.ndarray:
    sums = matrix.sum(axis=1)

    if hasattr(sums, "A1"):
        return np.asarray(sums.A1, dtype=np.float64)

    return np.asarray(sums, dtype=np.float64).reshape(-1)


def _as_cells_by_genes_matrix(
    counts: Any,
    *,
    orientation: MatrixOrientation,
) -> Any:
    if _is_sparse_matrix(counts):
        matrix = counts.tocsr()
        if orientation == "genes_by_cells":
            matrix = matrix.T.tocsr()
        return matrix

    array = np.asarray(counts)

    if array.ndim == 1:
        array = array.reshape(1, -1)

    if array.ndim != 2:
        raise ValueError(f"counts must be 1D or 2D, got shape {array.shape}")

    if orientation == "genes_by_cells":
        array = array.T

    return np.asarray(array, dtype=np.float64)


def _get_nonzero_row_entries(matrix: Any, row_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Return nonzero column indices and values for one cell row."""
    if _is_sparse_matrix(matrix):
        row = matrix.getrow(row_idx)
        return row.indices.astype(np.int64), row.data.astype(np.float64)

    row = np.asarray(matrix[row_idx], dtype=np.float64).reshape(-1)
    nonzero_idx = np.flatnonzero(row)
    return nonzero_idx.astype(np.int64), row[nonzero_idx].astype(np.float64)


def _normalise_gene_id(
    value: Any,
    *,
    strip_ensembl_version: bool,
    uppercase: bool,
) -> str:
    gene_id = str(value).strip()

    if strip_ensembl_version and "." in gene_id:
        gene_id = gene_id.split(".", 1)[0]

    if uppercase:
        gene_id = gene_id.upper()

    return gene_id


def _coerce_bool_filter(filter_pass: Sequence[Any] | np.ndarray | None, n_cells: int) -> np.ndarray:
    if filter_pass is None:
        return np.ones(n_cells, dtype=bool)

    values = np.asarray(filter_pass)

    if values.shape[0] != n_cells:
        raise ValueError(f"filter_pass length {values.shape[0]} does not match n_cells={n_cells}")

    if values.dtype == bool:
        return values

    return values.astype(int) == 1


def _coerce_metadata(
    metadata: Mapping[str, Sequence[Any]] | None,
    *,
    n_cells: int,
) -> dict[str, list[Any]]:
    if metadata is None:
        return {}

    output: dict[str, list[Any]] = {}

    for key, values in metadata.items():
        values_list = list(values)
        if len(values_list) != n_cells:
            raise ValueError(
                f"metadata[{key!r}] length {len(values_list)} does not match n_cells={n_cells}"
            )
        output[str(key)] = values_list

    return output


class GeneformerTranscriptomeTokenizer:
    """Local direct tokenizer compatible with Geneformer V2 token IDs.

    The tokenization follows the Geneformer rank-value scheme:

    1. Map input gene IDs to Geneformer vocabulary Ensembl IDs.
    2. Drop genes absent from the Geneformer token dictionary or median dictionary.
    3. For each cell, use raw counts and total `n_counts`.
    4. Normalize counts by total counts and `target_sum`.
    5. Divide by Geneformer's per-gene median dictionary.
    6. Rank detected genes by the normalized values.
    7. Convert ranked genes to token IDs.
    8. For V2, prepend `<cls>` and append `<eos>`.
    """

    def __init__(
        self,
        config: GeneformerTranscriptomeTokenizerConfig | None = None,
        **overrides: Any,
    ) -> None:
        self.config = replace(config or GeneformerTranscriptomeTokenizerConfig(), **overrides)

        if self.config.model_version == "V1":
            self.config = replace(
                self.config,
                model_input_size=2048,
                special_token=False,
            )
        elif self.config.model_version == "V2":
            self.config = replace(
                self.config,
                model_input_size=int(self.config.model_input_size),
                special_token=bool(self.config.special_token),
            )
        else:
            raise ValueError(f"Unsupported Geneformer model_version: {self.config.model_version!r}")

        self.geneformer_repo = _resolve_repo_path(self.config.geneformer_repo)
        self.package_dir = self.geneformer_repo / "geneformer"

        if not self.package_dir.exists():
            raise FileNotFoundError(f"Geneformer package directory does not exist: {self.package_dir}")

        self.gene_median_file = self._resolve_dictionary_file(
            self.config.gene_median_file,
            default_v2=(
                "gene_median_dictionary_gc104M.pkl",
                "gene_median_dictionary_gc95M.pkl",
                "gene_median_dictionary.pkl",
            ),
            default_v1=(
                "gene_dictionaries_30m/gene_median_dictionary_gc30M.pkl",
                "gene_dictionaries_30m/gene_median_dictionary.pkl",
            ),
            name="gene median dictionary",
        )
        self.token_dictionary_file = self._resolve_dictionary_file(
            self.config.token_dictionary_file,
            default_v2=(
                "token_dictionary_gc104M.pkl",
                "token_dictionary_gc95M.pkl",
                "token_dictionary.pkl",
            ),
            default_v1=(
                "gene_dictionaries_30m/token_dictionary_gc30M.pkl",
                "gene_dictionaries_30m/token_dictionary.pkl",
            ),
            name="token dictionary",
        )
        self.gene_mapping_file = self._resolve_optional_dictionary_file(
            self.config.gene_mapping_file,
            default_v2=(
                "ensembl_mapping_dict_gc104M.pkl",
                "ensembl_mapping_dict_gc95M.pkl",
                "ensembl_mapping_dict.pkl",
            ),
            default_v1=(
                "gene_dictionaries_30m/ensembl_mapping_dict_gc30M.pkl",
                "gene_dictionaries_30m/ensembl_mapping_dict.pkl",
            ),
        )

        self.gene_median_dict = {
            str(key): float(value)
            for key, value in _load_pickle(self.gene_median_file).items()
        }
        self.gene_token_dict = {
            str(key): int(value)
            for key, value in _load_pickle(self.token_dictionary_file).items()
        }

        if self.gene_mapping_file is not None:
            loaded_mapping = _load_pickle(self.gene_mapping_file)
            self.gene_mapping_dict = {
                str(key): str(value)
                for key, value in loaded_mapping.items()
            }
        else:
            self.gene_mapping_dict = {key: key for key in self.gene_token_dict}

        # Keep only mappings whose target exists in the token dictionary.
        token_gene_keys = set(self.gene_token_dict)
        self.gene_mapping_dict = {
            key: value
            for key, value in self.gene_mapping_dict.items()
            if value in token_gene_keys
        }

        self.cls_token_id = self.gene_token_dict.get("<cls>")
        self.eos_token_id = self.gene_token_dict.get("<eos>")

        if self.config.special_token and (self.cls_token_id is None or self.eos_token_id is None):
            raise ValueError(
                "Geneformer V2 tokenization requires <cls> and <eos> in the token dictionary."
            )

    def _resolve_dictionary_file(
        self,
        explicit_path: str | Path | None,
        *,
        default_v2: tuple[str, ...],
        default_v1: tuple[str, ...],
        name: str,
    ) -> Path:
        if explicit_path is not None:
            path = _resolve_repo_path(explicit_path)
            if not path.exists():
                raise FileNotFoundError(f"Explicit {name} file does not exist: {path}")
            return path

        candidates = default_v1 if self.config.model_version == "V1" else default_v2
        path = _first_existing_path(self.package_dir, candidates)

        if path is None:
            raise FileNotFoundError(
                f"Could not locate local Geneformer {name} under {self.package_dir}. "
                "Check external/Geneformer LFS files."
            )

        return path

    def _resolve_optional_dictionary_file(
        self,
        explicit_path: str | Path | None,
        *,
        default_v2: tuple[str, ...],
        default_v1: tuple[str, ...],
    ) -> Path | None:
        if explicit_path is not None:
            path = _resolve_repo_path(explicit_path)
            if not path.exists():
                raise FileNotFoundError(f"Explicit gene mapping file does not exist: {path}")
            return path

        candidates = default_v1 if self.config.model_version == "V1" else default_v2
        return _first_existing_path(self.package_dir, candidates)

    def _canonical_gene_id(self, raw_gene_id: Any) -> str | None:
        gene_id = _normalise_gene_id(
            raw_gene_id,
            strip_ensembl_version=self.config.strip_ensembl_version,
            uppercase=self.config.uppercase_gene_ids,
        )

        if self.config.collapse_gene_ids:
            mapped_gene_id = self.gene_mapping_dict.get(gene_id)
            if mapped_gene_id is None and gene_id in self.gene_token_dict:
                mapped_gene_id = gene_id
        else:
            mapped_gene_id = gene_id

        if mapped_gene_id is None:
            return None

        if mapped_gene_id not in self.gene_token_dict:
            return None

        if mapped_gene_id not in self.gene_median_dict:
            return None

        median = self.gene_median_dict[mapped_gene_id]
        if not np.isfinite(median) or median <= 0:
            return None

        return mapped_gene_id

    def _canonicalize_gene_columns(self, gene_ids: Sequence[Any]) -> list[str | None]:
        if len(gene_ids) == 0:
            raise ValueError("gene_ids is empty.")

        return [self._canonical_gene_id(gene_id) for gene_id in gene_ids]

    def _tokenize_one_cell(
        self,
        *,
        col_indices: np.ndarray,
        count_values: np.ndarray,
        canonical_genes_by_col: Sequence[str | None],
        n_counts: float,
    ) -> list[int]:
        if not np.isfinite(n_counts) or n_counts <= 0:
            raise ValueError(f"n_counts must be finite and positive, got {n_counts!r}")

        collapsed_counts: dict[str, float] = {}

        for col_idx, count in zip(col_indices, count_values, strict=False):
            if not np.isfinite(count) or count <= 0:
                continue

            gene_id = canonical_genes_by_col[int(col_idx)]
            if gene_id is None:
                continue

            collapsed_counts[gene_id] = collapsed_counts.get(gene_id, 0.0) + float(count)

        if not collapsed_counts:
            return []

        gene_ids = list(collapsed_counts)
        raw_counts = np.asarray([collapsed_counts[gene_id] for gene_id in gene_ids], dtype=np.float64)
        medians = np.asarray([self.gene_median_dict[gene_id] for gene_id in gene_ids], dtype=np.float64)
        token_ids = np.asarray([self.gene_token_dict[gene_id] for gene_id in gene_ids], dtype=np.int64)

        scores = (raw_counts / float(n_counts) * float(self.config.target_sum)) / medians
        order = np.argsort(-scores, kind="stable")

        ranked_tokens = token_ids[order].astype(int).tolist()

        if self.config.special_token:
            # Leave room for CLS and EOS.
            ranked_tokens = ranked_tokens[: self.config.model_input_size - 2]
            ranked_tokens = [int(self.cls_token_id), *ranked_tokens, int(self.eos_token_id)]
        else:
            ranked_tokens = ranked_tokens[: self.config.model_input_size]

        return ranked_tokens

    def tokenize_counts(
        self,
        counts: Any,
        gene_ids: Sequence[Any],
        *,
        orientation: MatrixOrientation = "cells_by_genes",
        cell_ids: Sequence[Any] | None = None,
        n_counts: Sequence[float] | np.ndarray | None = None,
        filter_pass: Sequence[Any] | np.ndarray | None = None,
        metadata: Mapping[str, Sequence[Any]] | None = None,
    ) -> TokenizedTranscriptomes:
        """Tokenize a count matrix.

        Args:
            counts:
                Raw count matrix. Dense numpy array, scipy sparse matrix, or nested
                numeric sequence. Shape is controlled by `orientation`.
            gene_ids:
                Ensembl gene IDs corresponding to genes in the matrix.
            orientation:
                `"cells_by_genes"` means counts shape is [cells, genes].
                `"genes_by_cells"` means counts shape is [genes, cells].
            cell_ids:
                Optional cell/profile IDs. Defaults to cell_0, cell_1, ...
            n_counts:
                Optional total counts per cell. If omitted, row sums of raw counts
                are used.
            filter_pass:
                Optional boolean/0-1 inclusion mask, length n_cells.
            metadata:
                Optional per-cell metadata columns, each length n_cells.
        """
        matrix = _as_cells_by_genes_matrix(counts, orientation=orientation)

        n_cells, n_genes = matrix.shape

        if len(gene_ids) != n_genes:
            raise ValueError(f"gene_ids length {len(gene_ids)} does not match n_genes={n_genes}")

        if cell_ids is None:
            cell_id_values = [f"cell_{i}" for i in range(n_cells)]
        else:
            cell_id_values = [str(value) for value in cell_ids]
            if len(cell_id_values) != n_cells:
                raise ValueError(f"cell_ids length {len(cell_id_values)} does not match n_cells={n_cells}")

        if n_counts is None:
            n_count_values = _row_sum(matrix)
        else:
            n_count_values = np.asarray(n_counts, dtype=np.float64).reshape(-1)
            if n_count_values.shape[0] != n_cells:
                raise ValueError(
                    f"n_counts length {n_count_values.shape[0]} does not match n_cells={n_cells}"
                )

        include_mask = _coerce_bool_filter(filter_pass, n_cells)
        metadata_values = _coerce_metadata(metadata, n_cells=n_cells)
        canonical_genes_by_col = self._canonicalize_gene_columns(gene_ids)

        output_input_ids: list[list[int]] = []
        output_cell_ids: list[str] = []
        output_lengths: list[int] = []
        output_metadata: dict[str, list[Any]] = {key: [] for key in metadata_values}

        for cell_idx in range(n_cells):
            if not include_mask[cell_idx]:
                continue

            col_indices, count_values = _get_nonzero_row_entries(matrix, cell_idx)

            token_ids = self._tokenize_one_cell(
                col_indices=col_indices,
                count_values=count_values,
                canonical_genes_by_col=canonical_genes_by_col,
                n_counts=float(n_count_values[cell_idx]),
            )

            if not token_ids:
                if self.config.drop_empty_cells:
                    continue
                raise ValueError(
                    f"Cell {cell_id_values[cell_idx]!r} has no detected genes in the Geneformer vocabulary."
                )

            output_input_ids.append(token_ids)
            output_cell_ids.append(cell_id_values[cell_idx])
            output_lengths.append(len(token_ids))

            for key, values in metadata_values.items():
                output_metadata[key].append(values[cell_idx])

        if not output_input_ids:
            raise ValueError("No cells were tokenized. Check gene IDs, counts, and filters.")

        return TokenizedTranscriptomes(
            input_ids=output_input_ids,
            cell_ids=output_cell_ids,
            lengths=output_lengths,
            metadata=output_metadata,
        )

    def tokenize_dataframe(
        self,
        dataframe: Any,
        *,
        orientation: MatrixOrientation = "genes_by_cells",
        gene_id_column: str | None = None,
        count_columns: Sequence[str] | None = None,
        metadata: Mapping[str, Sequence[Any]] | None = None,
    ) -> TokenizedTranscriptomes:
        """Tokenize a pandas DataFrame.

        Default orientation expects one gene per row and one cell/profile per
        numeric count column:

            gene_id    host_A    host_B
            ENSG...    10        0
            ENSG...    2         5
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("tokenize_dataframe requires pandas.") from exc

        if not isinstance(dataframe, pd.DataFrame):
            dataframe = pd.DataFrame(dataframe)

        df = dataframe.copy()

        if orientation == "genes_by_cells":
            if gene_id_column is None:
                gene_id_column = str(df.columns[0])

            if gene_id_column not in df.columns:
                raise KeyError(f"gene_id_column={gene_id_column!r} not found in DataFrame.")

            gene_ids = df[gene_id_column].tolist()

            if count_columns is None:
                count_columns = [
                    col
                    for col in df.columns
                    if col != gene_id_column and pd.api.types.is_numeric_dtype(df[col])
                ]

            if not count_columns:
                raise ValueError("Could not infer count columns. Pass count_columns explicitly.")

            counts = df[list(count_columns)].to_numpy(dtype=np.float64)

            return self.tokenize_counts(
                counts,
                gene_ids=gene_ids,
                orientation="genes_by_cells",
                cell_ids=list(count_columns),
                metadata=metadata,
            )

        # cells_by_genes DataFrame: index is cells, columns are gene IDs.
        gene_ids = list(df.columns)
        cell_ids = [str(idx) for idx in df.index]
        counts = df.to_numpy(dtype=np.float64)

        return self.tokenize_counts(
            counts,
            gene_ids=gene_ids,
            orientation="cells_by_genes",
            cell_ids=cell_ids,
            metadata=metadata,
        )

    def tokenize_tsv(
        self,
        path: str | Path,
        *,
        sep: str = "\t",
        orientation: MatrixOrientation = "genes_by_cells",
        gene_id_column: str | None = None,
        count_columns: Sequence[str] | None = None,
    ) -> TokenizedTranscriptomes:
        """Read a TSV/CSV count table and tokenize it."""
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("tokenize_tsv requires pandas.") from exc

        path = Path(path)
        df = pd.read_csv(path, sep=sep)

        return self.tokenize_dataframe(
            df,
            orientation=orientation,
            gene_id_column=gene_id_column,
            count_columns=count_columns,
        )

    def tokenize_h5ad(
        self,
        path_or_adata: str | Path | Any,
        *,
        layer: str | None = None,
        ensembl_id_column: str = "ensembl_id",
        use_var_index_if_missing: bool = True,
        n_counts_column: str = "n_counts",
        filter_pass_column: str | None = "filter_pass",
        custom_attr_name_dict: Mapping[str, str] | None = None,
    ) -> TokenizedTranscriptomes:
        """Tokenize an AnnData h5ad object or file.

        Required/derived:
            - raw counts from adata.X or adata.layers[layer]
            - gene IDs from adata.var[ensembl_id_column] or adata.var_names
            - n_counts from adata.obs[n_counts_column] or row sums
        """
        try:
            import anndata as ad
        except ImportError as exc:
            raise ImportError("tokenize_h5ad requires anndata.") from exc

        if isinstance(path_or_adata, (str, Path)):
            adata = ad.read_h5ad(path_or_adata)
        else:
            adata = path_or_adata

        if layer is None:
            counts = adata.X
        else:
            if layer not in adata.layers:
                raise KeyError(f"layer={layer!r} not found in adata.layers.")
            counts = adata.layers[layer]

        if ensembl_id_column in adata.var.columns:
            gene_ids = adata.var[ensembl_id_column].tolist()
        elif use_var_index_if_missing:
            gene_ids = adata.var_names.tolist()
        else:
            raise KeyError(
                f"adata.var[{ensembl_id_column!r}] is missing and use_var_index_if_missing=False."
            )

        if n_counts_column in adata.obs.columns:
            n_counts = np.asarray(adata.obs[n_counts_column], dtype=np.float64)
        else:
            n_counts = None

        filter_pass = None
        if filter_pass_column is not None and filter_pass_column in adata.obs.columns:
            filter_pass = np.asarray(adata.obs[filter_pass_column])

        metadata: dict[str, list[Any]] = {}
        if custom_attr_name_dict is not None:
            for source_col, output_col in custom_attr_name_dict.items():
                if source_col not in adata.obs.columns:
                    raise KeyError(f"adata.obs[{source_col!r}] not found.")
                metadata[str(output_col)] = adata.obs[source_col].tolist()

        return self.tokenize_counts(
            counts,
            gene_ids=gene_ids,
            orientation="cells_by_genes",
            cell_ids=adata.obs_names.tolist(),
            n_counts=n_counts,
            filter_pass=filter_pass,
            metadata=metadata,
        )


GeneformerRawTranscriptomeTokenizer = GeneformerTranscriptomeTokenizer

__all__ = [
    "GeneformerTranscriptomeTokenizerConfig",
    "GeneformerTranscriptomeTokenizer",
    "GeneformerRawTranscriptomeTokenizer",
    "TokenizedTranscriptomes",
]