"""Host-query attention expression head."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from .base import EmbeddingFusionHead
from .utils import build_mlp, normalise_hidden_dims, validate_dropout, validate_positive_int


@dataclass(frozen=True)
class QueryExpressionHeadConfig:
    """Configuration for host-query attention over sequence latent slots."""

    sequence_embedding_dim: int
    host_embedding_dim: int
    fusion_dim: int
    num_heads: int = 4
    num_sequence_slots: int = 8
    num_queries: int = 4
    hidden_dims: tuple[int, ...] = ()
    dropout: float = 0.0
    output_dim: int = 1
    activation: str = "gelu"
    use_layer_norm: bool = True
    include_sequence_skip: bool = True
    include_host_skip: bool = False
    query_pooling: str = "mean"


@dataclass(frozen=True)
class QueryExpressionOutput:
    """Optional rich query-head output for debugging and attribution."""

    expression: torch.Tensor
    sequence_slots: torch.Tensor
    host_queries: torch.Tensor
    attention_context: torch.Tensor
    pooled_context: torch.Tensor
    attention_weights: torch.Tensor | None


class QueryExpressionHead(EmbeddingFusionHead):
    """Host-query attention over learned sequence latent slots."""

    def __init__(
        self,
        sequence_embedding_dim: int,
        host_embedding_dim: int,
        *,
        fusion_dim: int | None = None,
        num_heads: int = 4,
        num_sequence_slots: int = 8,
        num_queries: int = 4,
        hidden_dims: Sequence[int] | None = None,
        dropout: float = 0.0,
        output_dim: int = 1,
        activation: str = "gelu",
        use_layer_norm: bool = True,
        include_sequence_skip: bool = True,
        include_host_skip: bool = False,
        query_pooling: str = "mean",
    ) -> None:
        super().__init__()

        self.sequence_embedding_dim = validate_positive_int(
            sequence_embedding_dim,
            name="sequence_embedding_dim",
        )
        self.host_embedding_dim = validate_positive_int(host_embedding_dim, name="host_embedding_dim")
        self.fusion_dim = validate_positive_int(
            self.sequence_embedding_dim if fusion_dim is None else fusion_dim,
            name="fusion_dim",
        )
        self.num_heads = validate_positive_int(num_heads, name="num_heads")
        if self.fusion_dim % self.num_heads != 0:
            raise ValueError(
                "fusion_dim must be divisible by num_heads for query attention; "
                f"got fusion_dim={self.fusion_dim}, num_heads={self.num_heads}."
            )

        self.num_sequence_slots = validate_positive_int(num_sequence_slots, name="num_sequence_slots")
        self.num_queries = validate_positive_int(num_queries, name="num_queries")
        output_dim = validate_positive_int(output_dim, name="output_dim")
        hidden_dims_tuple = normalise_hidden_dims(hidden_dims)
        dropout = validate_dropout(dropout)

        query_pooling = str(query_pooling).lower()
        if query_pooling not in {"mean", "flatten"}:
            raise ValueError(f"query_pooling must be 'mean' or 'flatten', got {query_pooling!r}.")

        include_sequence_skip = bool(include_sequence_skip)
        include_host_skip = bool(include_host_skip)
        use_layer_norm = bool(use_layer_norm)

        self.config = QueryExpressionHeadConfig(
            sequence_embedding_dim=self.sequence_embedding_dim,
            host_embedding_dim=self.host_embedding_dim,
            fusion_dim=self.fusion_dim,
            num_heads=self.num_heads,
            num_sequence_slots=self.num_sequence_slots,
            num_queries=self.num_queries,
            hidden_dims=hidden_dims_tuple,
            dropout=dropout,
            output_dim=output_dim,
            activation=activation,
            use_layer_norm=use_layer_norm,
            include_sequence_skip=include_sequence_skip,
            include_host_skip=include_host_skip,
            query_pooling=query_pooling,
        )

        self.sequence_to_slots = nn.Linear(self.sequence_embedding_dim, self.num_sequence_slots * self.fusion_dim)
        self.host_to_queries = nn.Linear(self.host_embedding_dim, self.num_queries * self.fusion_dim)
        self.sequence_slot_norm = nn.LayerNorm(self.fusion_dim) if use_layer_norm else nn.Identity()
        self.host_query_norm = nn.LayerNorm(self.fusion_dim) if use_layer_norm else nn.Identity()
        self.attention = nn.MultiheadAttention(
            embed_dim=self.fusion_dim,
            num_heads=self.num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_output_norm = nn.LayerNorm(self.fusion_dim) if use_layer_norm else nn.Identity()

        self.sequence_skip_projection: nn.Module | None = None
        if include_sequence_skip:
            self.sequence_skip_projection = nn.Sequential(
                nn.Linear(self.sequence_embedding_dim, self.fusion_dim),
                nn.LayerNorm(self.fusion_dim) if use_layer_norm else nn.Identity(),
            )

        self.host_skip_projection: nn.Module | None = None
        if include_host_skip:
            self.host_skip_projection = nn.Sequential(
                nn.Linear(self.host_embedding_dim, self.fusion_dim),
                nn.LayerNorm(self.fusion_dim) if use_layer_norm else nn.Identity(),
            )

        self.predictor = build_mlp(
            self.input_dim,
            hidden_dims_tuple,
            output_dim,
            dropout=dropout,
            activation=activation,
        )

    @property
    def input_dim(self) -> int:
        input_dim = self.fusion_dim if self.config.query_pooling == "mean" else self.num_queries * self.fusion_dim
        if self.config.include_sequence_skip:
            input_dim += self.fusion_dim
        if self.config.include_host_skip:
            input_dim += self.fusion_dim
        return input_dim

    def forward(
        self,
        sequence_embedding: torch.Tensor,
        host_embedding: torch.Tensor,
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | QueryExpressionOutput:
        sequence_embedding, host_embedding = self._validate_pair(sequence_embedding, host_embedding)
        batch_size = sequence_embedding.shape[0]

        sequence_slots = self.sequence_to_slots(sequence_embedding).view(
            batch_size,
            self.num_sequence_slots,
            self.fusion_dim,
        )
        sequence_slots = self.sequence_slot_norm(sequence_slots)

        host_queries = self.host_to_queries(host_embedding).view(
            batch_size,
            self.num_queries,
            self.fusion_dim,
        )
        host_queries = self.host_query_norm(host_queries)

        attention_context, attention_weights = self.attention(
            host_queries,
            sequence_slots,
            sequence_slots,
            need_weights=return_aux,
            average_attn_weights=False,
        )
        attention_context = self.attention_output_norm(attention_context)

        if self.config.query_pooling == "mean":
            pooled_context = attention_context.mean(dim=1)
        else:
            pooled_context = attention_context.reshape(batch_size, -1)

        predictor_parts = [pooled_context]
        if self.sequence_skip_projection is not None:
            predictor_parts.append(self.sequence_skip_projection(sequence_embedding))
        if self.host_skip_projection is not None:
            predictor_parts.append(self.host_skip_projection(host_embedding))

        predictor_input = predictor_parts[0] if len(predictor_parts) == 1 else torch.cat(predictor_parts, dim=-1)
        expression = self.predictor(predictor_input)

        if return_aux:
            return QueryExpressionOutput(
                expression=expression,
                sequence_slots=sequence_slots,
                host_queries=host_queries,
                attention_context=attention_context,
                pooled_context=pooled_context,
                attention_weights=attention_weights,
            )
        return expression


__all__ = [
    "QueryExpressionHead",
    "QueryExpressionHeadConfig",
    "QueryExpressionOutput",
]
