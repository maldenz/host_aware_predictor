"""Dataset plumbing for host-aware expression-head regression.

Each sample is:

    sequence_embedding = data/processed/dna_emb/by_name/{name}.pt
    host_embedding     = data/processed/host_emb/{condition}/{condition}_geneformer_embedding.pt
    target             = element_quantifications.tsv["log2"] by default

Split membership is controlled by ``data/processed/splits/{train,val,test}_names.txt``.
The split files contain element names only, so multi-condition training naturally
uses the same element split across every condition.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import torch
from torch.utils.data import Dataset

from .embedding_io import Pooling, load_embedding_vector


@dataclass(frozen=True)
class ElementRecord:
    name: str
    condition: str
    target: float
    dna_embedding_path: Path
    host_embedding_path: Path
    weight: float = 1.0


@dataclass(frozen=True)
class SplitBuildReport:
    split: str
    requested_names: int
    quantification_rows: int
    usable_records: int
    missing_in_quantification: int
    missing_dna_embeddings: int
    missing_host_embeddings: int
    condition_counts: dict[str, int]


def read_split_names(path: str | Path) -> list[str]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip() and not line.lstrip().startswith("#")]


def discover_quantification_files(processed_dir: str | Path, *, condition: str | None = None) -> list[Path]:
    """Discover processed ``*_element_quantifications.tsv`` files."""

    processed_dir = Path(processed_dir)
    patterns: list[str]
    if condition:
        patterns = [
            f"*_{condition}_*_element_quantifications.tsv",
            f"*{condition}*element_quantifications.tsv",
            "*element_quantifications.tsv",
        ]
    else:
        patterns = ["*element_quantifications.tsv"]

    files: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for path in sorted(processed_dir.glob(pattern)):
            if path not in seen:
                files.append(path)
                seen.add(path)
    return files


def discover_conditions(host_embedding_dir: str | Path) -> list[str]:
    """Return condition names that have a host embedding subdirectory."""

    host_embedding_dir = Path(host_embedding_dir)
    if not host_embedding_dir.exists():
        return []
    return sorted(path.name for path in host_embedding_dir.iterdir() if path.is_dir())


def default_host_embedding_path(host_embedding_dir: str | Path, condition: str) -> Path:
    condition = str(condition)
    return Path(host_embedding_dir) / condition / f"{condition}_geneformer_embedding.pt"


def infer_condition_from_quantification_path(
    path: str | Path,
    *,
    known_conditions: Sequence[str] | None = None,
) -> str | None:
    """Infer condition from names like ``ENCSR203UFY_K562_ENCFF068BWG_element_quantifications.tsv``.

    If known host-embedding condition names are available, exact separator-aware
    matching is preferred.  Otherwise the second underscore-delimited token is
    used as a fallback for the ENCSR_condition_ENCFF naming pattern.
    """

    name = Path(path).name
    padded = f"_{name}_"
    for condition in sorted({str(c) for c in known_conditions or []}, key=len, reverse=True):
        if f"_{condition}_" in padded:
            return condition

    stem = name
    for suffix in ("_element_quantifications.tsv", "_element_quantifications.txt", ".tsv", ".txt"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    parts = [part for part in stem.split("_") if part]
    if len(parts) >= 3:
        return parts[1]
    return None


def load_quantification_table(
    paths: Sequence[str | Path],
    *,
    condition: str | None = None,
    known_conditions: Sequence[str] | None = None,
    name_col: str = "name",
    condition_col: str = "condition",
    target_col: str = "log2",
    weight_col: str | None = None,
) -> pd.DataFrame:
    """Load and normalize one or more element quantification TSVs."""

    if not paths:
        raise FileNotFoundError("No element_quantifications.tsv files were provided or discovered.")

    frames: list[pd.DataFrame] = []
    for path in paths:
        path = Path(path)
        frame = pd.read_csv(path, sep="\t")
        inferred_condition = infer_condition_from_quantification_path(path, known_conditions=known_conditions)
        if condition_col in frame.columns:
            frame[condition_col] = frame[condition_col].astype(str)
            missing_condition = frame[condition_col].isna() | frame[condition_col].isin(["", "nan", "None"])
            if inferred_condition is not None and bool(missing_condition.any()):
                frame.loc[missing_condition, condition_col] = inferred_condition
        else:
            fallback_condition = inferred_condition or condition
            if fallback_condition is None:
                raise ValueError(
                    f"Could not infer condition for {path}. Add a {condition_col!r} column, "
                    "name host embeddings so the condition appears in the file name, or pass --conditions."
                )
            frame[condition_col] = str(fallback_condition)
        frame["__source_tsv"] = str(path)
        frames.append(frame)

    df = pd.concat(frames, axis=0, ignore_index=True)

    required = [name_col, condition_col, target_col]
    if weight_col is not None:
        required.append(weight_col)
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Quantification table is missing required columns {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.copy()
    df[name_col] = df[name_col].astype(str)
    df[condition_col] = df[condition_col].astype(str)
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.loc[df[target_col].notna()].copy()

    if condition is not None:
        df = df.loc[df[condition_col].astype(str) == str(condition)].copy()

    if weight_col is not None:
        df[weight_col] = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0)
        df.loc[df[weight_col] <= 0, weight_col] = 0.0

    return df


def make_records_for_split(
    df: pd.DataFrame,
    *,
    split: str,
    split_names: Iterable[str],
    dna_embedding_dir: str | Path,
    host_embedding_dir: str | Path,
    name_col: str = "name",
    condition_col: str = "condition",
    target_col: str = "log2",
    weight_col: str | None = None,
    require_dna_embedding: bool = True,
    require_host_embedding: bool = True,
) -> tuple[list[ElementRecord], SplitBuildReport]:
    split_name_list = [str(name) for name in split_names]
    split_name_set = set(split_name_list)
    dna_embedding_dir = Path(dna_embedding_dir)
    host_embedding_dir = Path(host_embedding_dir)

    split_df = df.loc[df[name_col].astype(str).isin(split_name_set)].copy()
    observed_name_set = set(split_df[name_col].astype(str).tolist())

    records: list[ElementRecord] = []
    missing_dna = 0
    missing_host = 0

    for _, row in split_df.iterrows():
        name = str(row[name_col])
        condition = str(row[condition_col])

        dna_path = dna_embedding_dir / f"{name}.pt"
        if not dna_path.exists():
            missing_dna += 1
            if require_dna_embedding:
                continue

        host_path = default_host_embedding_path(host_embedding_dir, condition)
        if not host_path.exists():
            missing_host += 1
            if require_host_embedding:
                continue

        target = float(row[target_col])
        weight = 1.0
        if weight_col is not None:
            weight = float(row[weight_col])
            if weight <= 0.0:
                weight = 0.0

        records.append(
            ElementRecord(
                name=name,
                condition=condition,
                target=target,
                dna_embedding_path=dna_path,
                host_embedding_path=host_path,
                weight=weight,
            )
        )

    condition_counts = dict(sorted(Counter(record.condition for record in records).items()))
    report = SplitBuildReport(
        split=split,
        requested_names=len(split_name_set),
        quantification_rows=int(len(split_df)),
        usable_records=len(records),
        missing_in_quantification=len(split_name_set - observed_name_set),
        missing_dna_embeddings=missing_dna,
        missing_host_embeddings=missing_host,
        condition_counts=condition_counts,
    )
    return records, report


class ElementQuantificationDataset(Dataset[dict[str, object]]):
    """PyTorch dataset over precomputed DNA embeddings and per-condition host embeddings."""

    def __init__(
        self,
        records: Sequence[ElementRecord],
        *,
        dna_pooling: Pooling = "mean",
        host_pooling: Pooling = "mean",
        cache_embeddings: bool = True,
    ) -> None:
        self.records = list(records)
        self.dna_pooling = dna_pooling
        self.host_pooling = host_pooling
        self.cache_embeddings = cache_embeddings
        self._dna_cache: dict[Path, torch.Tensor] = {}
        self._host_cache: dict[Path, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.records)

    @property
    def sequence_embedding_dim(self) -> int:
        if not self.records:
            raise ValueError("Cannot infer sequence embedding dim from an empty dataset.")
        return int(self._load_dna_embedding(self.records[0].dna_embedding_path).numel())

    @property
    def host_embedding_dim(self) -> int:
        if not self.records:
            raise ValueError("Cannot infer host embedding dim from an empty dataset.")
        return int(self._load_host_embedding(self.records[0].host_embedding_path).numel())

    @property
    def conditions(self) -> list[str]:
        return sorted({record.condition for record in self.records})

    @property
    def condition_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(record.condition for record in self.records).items()))

    def _load_dna_embedding(self, path: Path) -> torch.Tensor:
        if self.cache_embeddings and path in self._dna_cache:
            return self._dna_cache[path]
        embedding = load_embedding_vector(path, pooling=self.dna_pooling)
        if self.cache_embeddings:
            self._dna_cache[path] = embedding
        return embedding

    def _load_host_embedding(self, path: Path) -> torch.Tensor:
        if self.cache_embeddings and path in self._host_cache:
            return self._host_cache[path]
        embedding = load_embedding_vector(path, pooling=self.host_pooling)
        if self.cache_embeddings:
            self._host_cache[path] = embedding
        return embedding

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        sequence_embedding = self._load_dna_embedding(record.dna_embedding_path)
        host_embedding = self._load_host_embedding(record.host_embedding_path)
        return {
            "name": record.name,
            "condition": record.condition,
            "sequence_embedding": sequence_embedding,
            "host_embedding": host_embedding,
            "target": torch.tensor([record.target], dtype=torch.float32),
            "weight": torch.tensor([record.weight], dtype=torch.float32),
        }


def collate_element_batch(batch: Sequence[dict[str, object]]) -> dict[str, object]:
    if not batch:
        raise ValueError("Cannot collate an empty batch.")
    return {
        "name": [str(item["name"]) for item in batch],
        "condition": [str(item["condition"]) for item in batch],
        "sequence_embedding": torch.stack([item["sequence_embedding"] for item in batch]),
        "host_embedding": torch.stack([item["host_embedding"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
        "weight": torch.stack([item["weight"] for item in batch]),
    }


__all__ = [
    "ElementQuantificationDataset",
    "ElementRecord",
    "SplitBuildReport",
    "collate_element_batch",
    "default_host_embedding_path",
    "discover_conditions",
    "discover_quantification_files",
    "infer_condition_from_quantification_path",
    "load_quantification_table",
    "make_records_for_split",
    "read_split_names",
]
