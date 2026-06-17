"""Expression-head registry and factory."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .base import EmbeddingFusionHead
from .concat_head import ConcatExpressionHead
from .film_head import FiLMExpressionHead
from .query_head import QueryExpressionHead
from .sequence_only_head import SequenceOnlyExpressionHead


HEAD_REGISTRY: Mapping[str, type[EmbeddingFusionHead]] = {
    "concat": ConcatExpressionHead,
    "film": FiLMExpressionHead,
    "query": QueryExpressionHead,
    "sequence_only": SequenceOnlyExpressionHead,
}


def available_head_names() -> tuple[str, ...]:
    return tuple(sorted(HEAD_REGISTRY.keys()))


def build_expression_head(
    head: str,
    *,
    sequence_embedding_dim: int,
    host_embedding_dim: int,
    hidden_dims: Sequence[int] | None = None,
    dropout: float = 0.0,
    output_dim: int = 1,
    activation: str = "gelu",
    fusion_dim: int | None = None,
    film_hidden_dims: Sequence[int] | None = None,
    film_use_layer_norm: bool = True,
    film_gamma_scale: float = 1.0,
    film_include_host_skip: bool = False,
    film_identity_init: bool = True,
    query_num_heads: int = 4,
    query_num_sequence_slots: int = 8,
    query_num_queries: int = 4,
    query_use_layer_norm: bool = True,
    query_include_sequence_skip: bool = True,
    query_include_host_skip: bool = False,
    query_pooling: str = "mean",
    **extra_head_kwargs: Any,
) -> EmbeddingFusionHead:
    """Instantiate a registered expression head by name."""

    head = str(head).lower()
    if head == "concat":
        _reject_extra_kwargs(head, extra_head_kwargs)
        return ConcatExpressionHead(
            sequence_embedding_dim=sequence_embedding_dim,
            host_embedding_dim=host_embedding_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            output_dim=output_dim,
            activation=activation,
        )

    if head == "sequence_only":
        _reject_extra_kwargs(head, extra_head_kwargs)
        return SequenceOnlyExpressionHead(
            sequence_embedding_dim=sequence_embedding_dim,
            host_embedding_dim=host_embedding_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            output_dim=output_dim,
            activation=activation,
        )

    if head == "film":
        _reject_extra_kwargs("FiLM", extra_head_kwargs)
        return FiLMExpressionHead(
            sequence_embedding_dim=sequence_embedding_dim,
            host_embedding_dim=host_embedding_dim,
            fusion_dim=fusion_dim,
            hidden_dims=hidden_dims,
            film_hidden_dims=film_hidden_dims,
            dropout=dropout,
            output_dim=output_dim,
            activation=activation,
            use_layer_norm=film_use_layer_norm,
            gamma_scale=film_gamma_scale,
            include_host_skip=film_include_host_skip,
            identity_init=film_identity_init,
        )

    if head == "query":
        _reject_extra_kwargs(head, extra_head_kwargs)
        return QueryExpressionHead(
            sequence_embedding_dim=sequence_embedding_dim,
            host_embedding_dim=host_embedding_dim,
            fusion_dim=fusion_dim,
            num_heads=query_num_heads,
            num_sequence_slots=query_num_sequence_slots,
            num_queries=query_num_queries,
            hidden_dims=hidden_dims,
            dropout=dropout,
            output_dim=output_dim,
            activation=activation,
            use_layer_norm=query_use_layer_norm,
            include_sequence_skip=query_include_sequence_skip,
            include_host_skip=query_include_host_skip,
            query_pooling=query_pooling,
        )

    raise ValueError(f"Unknown head {head!r}. Available heads: {available_head_names()}.")


def _reject_extra_kwargs(head: str, extra_head_kwargs: Mapping[str, Any]) -> None:
    if extra_head_kwargs:
        raise ValueError(f"Unsupported extra kwargs for {head} head: {sorted(extra_head_kwargs)}")


__all__ = [
    "HEAD_REGISTRY",
    "available_head_names",
    "build_expression_head",
]
