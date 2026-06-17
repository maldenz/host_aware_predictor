"""Expression heads for frozen DNA and host embeddings.

All heads expose the same forward contract::

    prediction = head(sequence_embedding=seq, host_embedding=host)

where ``seq`` and ``host`` are rank-2 floating-point tensors shaped
``[batch, embedding_dim]``.
"""

from __future__ import annotations

from .base import EmbeddingFusionHead
from .concat_head import ConcatExpressionHead, ConcatExpressionHeadConfig
from .film_head import FiLMExpressionHead, FiLMExpressionHeadConfig, FiLMExpressionOutput
from .query_head import QueryExpressionHead, QueryExpressionHeadConfig, QueryExpressionOutput
from .registry import HEAD_REGISTRY, available_head_names, build_expression_head
from .sequence_only_head import SequenceOnlyExpressionHead, SequenceOnlyExpressionHeadConfig
from .utils import expression_head_config_dict


__all__ = [
    "ConcatExpressionHead",
    "ConcatExpressionHeadConfig",
    "EmbeddingFusionHead",
    "FiLMExpressionHead",
    "FiLMExpressionHeadConfig",
    "FiLMExpressionOutput",
    "QueryExpressionHead",
    "QueryExpressionHeadConfig",
    "QueryExpressionOutput",
    "HEAD_REGISTRY",
    "SequenceOnlyExpressionHead",
    "SequenceOnlyExpressionHeadConfig",
    "available_head_names",
    "build_expression_head",
    "expression_head_config_dict",
]
