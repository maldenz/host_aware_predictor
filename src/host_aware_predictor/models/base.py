"""Base interface and validation for embedding fusion heads."""

from __future__ import annotations

import torch
from torch import nn


class EmbeddingFusionHead(nn.Module):
    """Base class for heads that consume precomputed sequence and host vectors."""

    sequence_embedding_dim: int
    host_embedding_dim: int

    def _validate_embedding(
        self,
        embedding: torch.Tensor,
        *,
        name: str,
        expected_dim: int,
    ) -> torch.Tensor:
        if not torch.is_tensor(embedding):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(embedding).__name__}.")

        if embedding.ndim != 2:
            raise ValueError(f"{name} must be shaped [batch, dim], got {tuple(embedding.shape)}.")

        if embedding.shape[-1] != expected_dim:
            raise ValueError(f"{name} has embedding dim {embedding.shape[-1]}, expected {expected_dim}.")

        if not embedding.is_floating_point():
            raise TypeError(f"{name} must be floating-point, got {embedding.dtype}.")

        return embedding

    def _validate_pair(
        self,
        sequence_embedding: torch.Tensor,
        host_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence_embedding = self._validate_embedding(
            sequence_embedding,
            name="sequence_embedding",
            expected_dim=self.sequence_embedding_dim,
        )
        host_embedding = self._validate_embedding(
            host_embedding,
            name="host_embedding",
            expected_dim=self.host_embedding_dim,
        )

        if sequence_embedding.shape[0] != host_embedding.shape[0]:
            raise ValueError(
                "sequence_embedding and host_embedding batch sizes must match; "
                f"got {sequence_embedding.shape[0]} and {host_embedding.shape[0]}."
            )

        return sequence_embedding, host_embedding


__all__ = ["EmbeddingFusionHead"]
