"""Geneformer transcriptome tokenization helpers.

This module prepares raw single-cell transcriptome count matrices for Geneformer
and delegates the actual rank-value encoding to geneformer.TranscriptomeTokenizer.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import inspect
import shutil
import sys
import types
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np

TranscriptomeFileFormat = Literal["h5ad", "loom", "zarr"]


@dataclass(frozen=True)
class GeneformerTranscriptomeTokenizerConfig:
    """Configuration for Geneformer transcriptome tokenization."""

    # Local Geneformer source checkout. Set to None if geneformer is pip-installed.
    geneformer_repo: Path | None = Path("external/Geneformer")

    # Geneformer V2 defaults match Geneformer-V2-316M.
    model_version: Literal["V1", "V2"] = "V2"
    model_input_size: int = 4096
    special_token: bool = True

    # Geneformer tokenizer behavior.
    collapse_gene_ids: bool = True
    use_h5ad_index: bool = False
    keep_counts: bool = False
    nproc: int = 1
    chunk_size: int = 512
    custom_attr_name_dict: Mapping[str, str] | None = None

    # Optional explicit Geneformer dictionary files. Leave as None to use the
    # defaults bundled with the imported geneformer package/source checkout.
    gene_median_file: Path | None = None
    token_dictionary_file: Path | None = None
    gene_mapping_file: Path | None = None

    # Preprocessing controls for raw h5ad input.
    work_dir: Path = Path("data/interim/geneformer_prepared")
    ensembl_id_column: str = "ensembl_id"
    use_var_index_if_missing: bool = True
    strip_ensembl_version: bool = True
    n_counts_column: str = "n_counts"
    count_layer: str | None = None
    overwrite_prepared: bool = True


def _first_existing_path(package_dir: Path, candidates: tuple[str, ...]) -> Path | None:
    """Return first existing Geneformer dictionary file from candidate names."""
    for candidate in candidates:
        candidate_path = package_dir / candidate
        if candidate_path.exists():
            return candidate_path
    return None


def _load_geneformer_transcriptome_tokenizer_direct(geneformer_repo: Path) -> type:
    """Load external/Geneformer/geneformer/tokenizer.py without importing geneformer/__init__.py.

    Geneformer's package initializer imports collator/classification modules that may be
    incompatible with the installed transformers version. The transcriptome tokenizer itself
    is independent, so we load only tokenizer.py and provide the constants it imports from ".".
    """
    repo = Path(geneformer_repo).expanduser().resolve()
    package_dir = repo / "geneformer"
    tokenizer_py = package_dir / "tokenizer.py"

    if not tokenizer_py.exists():
        raise ImportError(f"Could not find Geneformer tokenizer file: {tokenizer_py}")

    package_name = "_host_aware_geneformer_tokenizer_only"

    fake_package = types.ModuleType(package_name)
    fake_package.__file__ = str(package_dir / "__init__.py")
    fake_package.__package__ = package_name
    fake_package.__path__ = [str(package_dir)]  # type: ignore[attr-defined]
    fake_package.__spec__ = importlib.machinery.ModuleSpec(
        package_name,
        loader=None,
        is_package=True,
    )

    # Current Geneformer V2 / Genecorpus-104M dictionary files.
    fake_package.GENE_MEDIAN_FILE = _first_existing_path(
        package_dir,
        (
            "gene_median_dictionary_gc104M.pkl",
            "gene_median_dictionary_gc95M.pkl",
            "gene_median_dictionary_95m.pkl",
            "gene_median_dictionary.pkl",
        ),
    )
    fake_package.TOKEN_DICTIONARY_FILE = _first_existing_path(
        package_dir,
        (
            "token_dictionary_gc104M.pkl",
            "token_dictionary_gc95M.pkl",
            "token_dictionary_95m.pkl",
            "token_dictionary.pkl",
        ),
    )
    fake_package.ENSEMBL_MAPPING_FILE = _first_existing_path(
        package_dir,
        (
            "ensembl_mapping_dict_gc104M.pkl",
            "ensembl_mapping_dict_gc95M.pkl",
            "ensembl_mapping_dict_95m.pkl",
            "ensembl_mapping_dict.pkl",
        ),
    )

    # Optional names used only if model_version="V1".
    v1_dir = package_dir / "gene_dictionaries_30m"
    fake_package.GENE_MEDIAN_FILE_30M = _first_existing_path(
        v1_dir,
        (
            "gene_median_dictionary_gc30M.pkl",
            "gene_median_dictionary.pkl",
        ),
    )
    fake_package.TOKEN_DICTIONARY_FILE_30M = _first_existing_path(
        v1_dir,
        (
            "token_dictionary_gc30M.pkl",
            "token_dictionary.pkl",
        ),
    )
    fake_package.ENSEMBL_MAPPING_FILE_30M = _first_existing_path(
        v1_dir,
        (
            "ensembl_mapping_dict_gc30M.pkl",
            "ensembl_mapping_dict.pkl",
        ),
    )

    missing_required = [
        name
        for name in ("GENE_MEDIAN_FILE", "TOKEN_DICTIONARY_FILE")
        if getattr(fake_package, name) is None
    ]
    if missing_required:
        raise ImportError(
            "Could not locate required Geneformer dictionary files in "
            f"{package_dir}. Missing: {missing_required}. "
            "Run `git lfs pull` inside external/Geneformer if these are missing."
        )

    sys.modules[package_name] = fake_package

    module_name = f"{package_name}.tokenizer"
    sys.modules.pop(module_name, None)

    spec = importlib.util.spec_from_file_location(module_name, tokenizer_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {tokenizer_py}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except ImportError as exc:
        raise ImportError(
            "Geneformer tokenizer.py was found, but one tokenizer-specific dependency "
            "is missing. Install dependencies with: "
            "pip install scanpy anndata loompy datasets tqdm"
        ) from exc

    try:
        return module.TranscriptomeTokenizer
    except AttributeError as exc:
        raise ImportError(f"{tokenizer_py} does not define TranscriptomeTokenizer.") from exc


def _load_geneformer_transcriptome_tokenizer(geneformer_repo: Path | None) -> type:
    """Import Geneformer TranscriptomeTokenizer.

    Prefer direct tokenizer.py loading because package-level Geneformer imports can fail
    on newer transformers versions due to unrelated classification/collator modules.
    """
    if geneformer_repo is None:
        try:
            from geneformer import TranscriptomeTokenizer  # type: ignore

            return TranscriptomeTokenizer
        except ImportError as exc:
            raise ImportError(
                "Could not import geneformer.TranscriptomeTokenizer. "
                "Set GeneformerTranscriptomeTokenizerConfig(geneformer_repo=...) "
                "or pass --geneformer-repo external/Geneformer."
            ) from exc

    repo = Path(geneformer_repo).expanduser().resolve()

    if not repo.exists():
        raise ImportError(f"Geneformer repo does not exist: {repo}")

    return _load_geneformer_transcriptome_tokenizer_direct(repo)

def _filter_kwargs_for_callable(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop kwargs unsupported by older Geneformer releases."""
    signature = inspect.signature(callable_obj)
    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_var_kwargs:
        return kwargs

    supported = set(signature.parameters)
    return {key: value for key, value in kwargs.items() if key in supported}


def infer_geneformer_file_format(
    input_path: Path | str,
    file_format: TranscriptomeFileFormat | None = None,
) -> TranscriptomeFileFormat:
    """Infer Geneformer input format from a file or directory path."""
    if file_format is not None:
        return file_format

    path = Path(input_path)
    suffix = path.suffix.lower().lstrip(".")
    if suffix in {"h5ad", "loom", "zarr"}:
        return suffix  # type: ignore[return-value]

    if path.is_dir():
        for candidate_format in ("h5ad", "loom", "zarr"):
            if any(path.glob(f"*.{candidate_format}")):
                return candidate_format  # type: ignore[return-value]

    raise ValueError(
        f"Could not infer Geneformer file format from {path}. "
        "Pass file_format='h5ad', 'loom', or 'zarr'."
    )


def _sum_counts_by_cell(matrix: Any) -> np.ndarray:
    """Return total counts per cell from dense or sparse AnnData.X."""
    counts = matrix.sum(axis=1)
    if hasattr(counts, "A1"):
        return np.asarray(counts.A1, dtype=np.float64)
    return np.asarray(counts, dtype=np.float64).reshape(-1)


def prepare_h5ad_for_geneformer(
    input_h5ad: Path | str,
    output_h5ad: Path | str,
    *,
    ensembl_id_column: str = "ensembl_id",
    use_var_index_if_missing: bool = True,
    strip_ensembl_version: bool = True,
    n_counts_column: str = "n_counts",
    count_layer: str | None = None,
) -> Path:
    """Prepare a raw-count h5ad file for Geneformer tokenization.

    The output h5ad will contain:
      - adata.var["ensembl_id"]
      - adata.obs["n_counts"]

    Geneformer expects raw counts in adata.X. If your raw counts are stored in a
    layer, pass count_layer="counts" or the relevant layer name.
    """
    try:
        import anndata as ad
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "prepare_h5ad_for_geneformer requires anndata and pandas. "
            "Install them before preparing .h5ad input."
        ) from exc

    input_h5ad = Path(input_h5ad)
    output_h5ad = Path(output_h5ad)
    output_h5ad.parent.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(input_h5ad)

    if count_layer is not None:
        if count_layer not in adata.layers:
            raise KeyError(f"count_layer={count_layer!r} not found in adata.layers.")
        adata.X = adata.layers[count_layer].copy()

    if ensembl_id_column in adata.var.columns:
        ensembl_ids = pd.Series(
            adata.var[ensembl_id_column].to_numpy(),
            index=adata.var.index,
            dtype="string",
        )
    elif use_var_index_if_missing:
        ensembl_ids = pd.Series(
            adata.var_names.to_numpy(),
            index=adata.var.index,
            dtype="string",
        )
    else:
        raise KeyError(
            f"adata.var[{ensembl_id_column!r}] is missing and "
            "use_var_index_if_missing=False."
        )

    ensembl_ids = ensembl_ids.fillna("")
    if strip_ensembl_version:
        ensembl_ids = ensembl_ids.str.replace(r"\.\d+$", "", regex=True)
    adata.var["ensembl_id"] = ensembl_ids.str.upper().astype(str).to_numpy()

    if "ensembl_id_collapsed" in adata.var.columns:
        del adata.var["ensembl_id_collapsed"]

    if n_counts_column in adata.obs.columns:
        adata.obs["n_counts"] = np.asarray(adata.obs[n_counts_column], dtype=np.float64)
    elif "n_counts" not in adata.obs.columns:
        adata.obs["n_counts"] = _sum_counts_by_cell(adata.X)

    n_counts = np.asarray(adata.obs["n_counts"], dtype=np.float64)
    if np.any(~np.isfinite(n_counts)) or np.any(n_counts <= 0):
        raise ValueError(
            "adata.obs['n_counts'] contains non-finite or non-positive values. "
            "Remove empty cells or provide a valid n_counts column before tokenization."
        )

    adata.write_h5ad(output_h5ad)
    return output_h5ad


class GeneformerTranscriptomeTokenizer:
    """Small project wrapper around geneformer.TranscriptomeTokenizer."""

    def __init__(
        self,
        config: GeneformerTranscriptomeTokenizerConfig | None = None,
        **overrides: Any,
    ) -> None:
        self.config = replace(config or GeneformerTranscriptomeTokenizerConfig(), **overrides)

    def _build_tokenizer(self) -> Any:
        tokenizer_cls = _load_geneformer_transcriptome_tokenizer(self.config.geneformer_repo)

        kwargs: dict[str, Any] = {
            "custom_attr_name_dict": dict(self.config.custom_attr_name_dict)
            if self.config.custom_attr_name_dict is not None
            else None,
            "nproc": self.config.nproc,
            "chunk_size": self.config.chunk_size,
            "model_input_size": self.config.model_input_size,
            "special_token": self.config.special_token,
            "collapse_gene_ids": self.config.collapse_gene_ids,
            "use_h5ad_index": self.config.use_h5ad_index,
            "keep_counts": self.config.keep_counts,
            "model_version": self.config.model_version,
        }

        optional_path_kwargs = {
            "gene_median_file": self.config.gene_median_file,
            "token_dictionary_file": self.config.token_dictionary_file,
            "gene_mapping_file": self.config.gene_mapping_file,
        }
        for key, value in optional_path_kwargs.items():
            if value is not None:
                kwargs[key] = Path(value)

        kwargs = _filter_kwargs_for_callable(tokenizer_cls.__init__, kwargs)
        return tokenizer_cls(**kwargs)

    def _prepare_h5ad_input(self, input_path: Path) -> Path:
        prepared_root = Path(self.config.work_dir)
        prepared_root.mkdir(parents=True, exist_ok=True)

        if input_path.is_dir():
            source_files = sorted(input_path.glob("*.h5ad"))
            if not source_files:
                raise FileNotFoundError(f"No .h5ad files found in {input_path}.")

            output_dir = prepared_root / input_path.name
            output_dir.mkdir(parents=True, exist_ok=True)

            for source_file in source_files:
                target_file = output_dir / source_file.name
                if target_file.exists() and not self.config.overwrite_prepared:
                    continue

                prepare_h5ad_for_geneformer(
                    source_file,
                    target_file,
                    ensembl_id_column=self.config.ensembl_id_column,
                    use_var_index_if_missing=self.config.use_var_index_if_missing,
                    strip_ensembl_version=self.config.strip_ensembl_version,
                    n_counts_column=self.config.n_counts_column,
                    count_layer=self.config.count_layer,
                )

            return output_dir

        if not input_path.exists():
            raise FileNotFoundError(input_path)

        output_dir = prepared_root / input_path.stem
        output_file = output_dir / input_path.name

        if output_file.exists() and not self.config.overwrite_prepared:
            return output_dir

        prepare_h5ad_for_geneformer(
            input_path,
            output_file,
            ensembl_id_column=self.config.ensembl_id_column,
            use_var_index_if_missing=self.config.use_var_index_if_missing,
            strip_ensembl_version=self.config.strip_ensembl_version,
            n_counts_column=self.config.n_counts_column,
            count_layer=self.config.count_layer,
        )

        return output_dir

    def _prepare_passthrough_input(
        self,
        input_path: Path,
        file_format: TranscriptomeFileFormat,
    ) -> Path:
        """Return a directory containing loom/zarr input for Geneformer."""
        if input_path.is_dir() and input_path.suffix.lower() != ".zarr":
            return input_path

        if not input_path.exists():
            raise FileNotFoundError(input_path)

        output_dir = Path(self.config.work_dir) / input_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / input_path.name

        if output_path.exists() and self.config.overwrite_prepared:
            if output_path.is_dir():
                shutil.rmtree(output_path)
            else:
                output_path.unlink()

        if not output_path.exists():
            if input_path.is_dir():
                shutil.copytree(input_path, output_path)
            else:
                shutil.copy2(input_path, output_path)

        return output_dir

    def _prepare_input(
        self,
        input_path: Path | str,
        file_format: TranscriptomeFileFormat | None,
    ) -> tuple[Path, TranscriptomeFileFormat]:
        path = Path(input_path)
        inferred_format = infer_geneformer_file_format(path, file_format)

        if inferred_format == "h5ad":
            return self._prepare_h5ad_input(path), inferred_format

        return self._prepare_passthrough_input(path, inferred_format), inferred_format

    def tokenize(
        self,
        input_path: Path | str,
        output_dir: Path | str = Path("data/processed/geneformer"),
        output_prefix: str | None = None,
        *,
        file_format: TranscriptomeFileFormat | None = None,
        input_identifier: str = "",
        use_generator: bool = False,
    ) -> Path:
        """Tokenize raw transcriptome files into a Geneformer .dataset directory."""
        input_path = Path(input_path)
        prepared_input_dir, resolved_format = self._prepare_input(input_path, file_format)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if output_prefix is None:
            output_prefix = input_path.stem if input_path.is_file() else input_path.name

        output_dataset_path = (output_dir / output_prefix).with_suffix(".dataset")

        if output_dataset_path.exists() and self.config.overwrite_prepared:
            shutil.rmtree(output_dataset_path)

        tokenizer = self._build_tokenizer()

        kwargs: dict[str, Any] = {
            "data_directory": str(prepared_input_dir),
            "output_directory": str(output_dir),
            "output_prefix": output_prefix,
            "file_format": resolved_format,
            "input_identifier": input_identifier,
            "use_generator": use_generator,
        }

        kwargs = _filter_kwargs_for_callable(tokenizer.tokenize_data, kwargs)
        tokenizer.tokenize_data(**kwargs)

        return output_dataset_path

    def tokenize_to_dataset(self, *args: Any, **kwargs: Any) -> Any:
        """Tokenize and return a datasets.Dataset loaded from disk."""
        try:
            from datasets import load_from_disk
        except ImportError as exc:
            raise ImportError("tokenize_to_dataset requires the datasets package.") from exc

        dataset_path = self.tokenize(*args, **kwargs)
        return load_from_disk(str(dataset_path))


__all__ = [
    "GeneformerTranscriptomeTokenizerConfig",
    "GeneformerTranscriptomeTokenizer",
    "prepare_h5ad_for_geneformer",
    "infer_geneformer_file_format",
]