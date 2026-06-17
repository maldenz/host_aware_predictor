"""Sequence-only expression baseline head."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .base import EmbeddingFusionHead
from .utils import build_mlp, normalise_hidden_dims, validate_dropout, validate_positive_int


@dataclass(frozen=True)
class SequenceOnlyExpressionHeadConfig:
    """Configuration for a host-compatible sequence-only baseline head."""

    sequence_embedding_dim: int
    host_embedding_dim: int
    hidden_dims: tuple[int, ...] = ()
    dropout: float = 0.0
    output_dim: int = 1
    activation: str = "gelu"


class SequenceOnlyExpressionHead(EmbeddingFusionHead):
    """Trainable MLP over sequence embeddings only.

    The shared forward signature is preserved for training-loop compatibility,
    but host_embedding is deliberately ignored.
    """

    def __init__(
        self,
        sequence_embedding_dim: int,
        host_embedding_dim: int,
        *,
        hidden_dims: Sequence[int] | None = None,
        dropout: float = 0.0,
        output_dim: int = 1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()

        self.sequence_embedding_dim = validate_positive_int(
            sequence_embedding_dim,
            name="sequence_embedding_dim",
        )
        self.host_embedding_dim = validate_positive_int(host_embedding_dim, name="host_embedding_dim")
        output_dim = validate_positive_int(output_dim, name="output_dim")
        hidden_dims_tuple = normalise_hidden_dims(hidden_dims)
        dropout = validate_dropout(dropout)

        self.config = SequenceOnlyExpressionHeadConfig(
            sequence_embedding_dim=self.sequence_embedding_dim,
            host_embedding_dim=self.host_embedding_dim,
            hidden_dims=hidden_dims_tuple,
            dropout=dropout,
            output_dim=output_dim,
            activation=activation,
        )
        self.network = build_mlp(
            self.sequence_embedding_dim,
            hidden_dims_tuple,
            output_dim,
            dropout=dropout,
            activation=activation,
        )

    @property
    def input_dim(self) -> int:
        return self.sequence_embedding_dim

    def forward(
        self,
        sequence_embedding: torch.Tensor,
        host_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del host_embedding
        sequence_embedding = self._validate_embedding(
            sequence_embedding,
            name="sequence_embedding",
            expected_dim=self.sequence_embedding_dim,
        )
        return self.network(sequence_embedding)


__all__ = ["SequenceOnlyExpressionHead", "SequenceOnlyExpressionHeadConfig"]
