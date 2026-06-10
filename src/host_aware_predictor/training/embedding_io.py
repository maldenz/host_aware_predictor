"""Utilities for loading precomputed host and DNA embeddings.

The training pipeline expects each ``.pt`` file to resolve to one vector.  This
loader accepts common checkpoint shapes:

- a tensor directly
- a dict with keys such as ``embedding``, ``embeddings`` or ``pooled_embedding``
- a singleton batch tensor shaped ``[1, dim]``
- a token/cell matrix shaped ``[n, dim]``; by default this is mean pooled
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

Pooling = Literal["mean", "first", "flatten"]

_TENSOR_KEYS = (
    "pooled_embedding",
    "embedding",
    "embeddings",
    "sequence_embedding",
    "host_embedding",
    "dna_embedding",
    "last_hidden_state",
    "hidden_state",
    "hidden_states",
)


def _first_tensor_from_mapping(obj: Mapping[str, Any]) -> torch.Tensor | None:
    for key in _TENSOR_KEYS:
        value = obj.get(key)
        if torch.is_tensor(value):
            return value
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value)

    tensor_values: list[torch.Tensor] = []
    for value in obj.values():
        if torch.is_tensor(value):
            tensor_values.append(value)
        elif isinstance(value, np.ndarray):
            tensor_values.append(torch.from_numpy(value))

    if len(tensor_values) == 1:
        return tensor_values[0]

    return None


def _as_tensor(obj: Any, *, path: Path) -> torch.Tensor:
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj)
    if isinstance(obj, Mapping):
        tensor = _first_tensor_from_mapping(obj)
        if tensor is not None:
            return tensor
        available = ", ".join(str(key) for key in obj.keys())
        raise TypeError(
            f"Could not find a tensor in {path}. Tried keys {_TENSOR_KEYS}; "
            f"available keys: [{available}]."
        )
    if isinstance(obj, (list, tuple)):
        try:
            return torch.as_tensor(obj)
        except Exception as exc:  # pragma: no cover - defensive path
            raise TypeError(f"Could not convert list/tuple in {path} to a tensor.") from exc

    raise TypeError(f"Unsupported object type in {path}: {type(obj).__name__}.")


def coerce_embedding_vector(
    tensor: torch.Tensor,
    *,
    pooling: Pooling = "mean",
    path: str | Path | None = None,
) -> torch.Tensor:
    """Convert a loaded tensor to a 1-D floating point embedding vector."""

    source = f" from {path}" if path is not None else ""

    if not torch.is_tensor(tensor):
        raise TypeError(f"Expected tensor{source}, got {type(tensor).__name__}.")

    if tensor.numel() == 0:
        raise ValueError(f"Empty embedding tensor{source}.")

    tensor = tensor.detach().cpu()
    if not tensor.is_floating_point():
        tensor = tensor.float()
    else:
        tensor = tensor.to(dtype=torch.float32)

    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)

    if tensor.ndim == 1:
        return tensor.contiguous()

    if tensor.ndim == 2 and tensor.shape[0] == 1:
        return tensor.squeeze(0).contiguous()

    if pooling == "flatten":
        return tensor.reshape(-1).contiguous()

    if tensor.ndim >= 2:
        matrix = tensor.reshape(-1, tensor.shape[-1])
        if pooling == "mean":
            return matrix.mean(dim=0).contiguous()
        if pooling == "first":
            return matrix[0].contiguous()

    raise ValueError(f"Unsupported pooling mode {pooling!r} for tensor shape {tuple(tensor.shape)}{source}.")


def load_embedding_vector(
    path: str | Path,
    *,
    pooling: Pooling = "mean",
    map_location: str | torch.device = "cpu",
) -> torch.Tensor:
    """Load a ``.pt`` embedding and return a 1-D float32 CPU tensor."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    try:
        obj = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # Older PyTorch versions do not accept weights_only.
        obj = torch.load(path, map_location=map_location)

    tensor = _as_tensor(obj, path=path)
    return coerce_embedding_vector(tensor, pooling=pooling, path=path)


def infer_embedding_dim(path: str | Path, *, pooling: Pooling = "mean") -> int:
    """Return the vector length produced by ``load_embedding_vector``."""

    return int(load_embedding_vector(path, pooling=pooling).numel())
