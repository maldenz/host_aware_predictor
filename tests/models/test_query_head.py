from __future__ import annotations

import pytest
import torch
from torch import nn

from host_aware_predictor.models.query_head import QueryExpressionHead, QueryExpressionOutput
from host_aware_predictor.models.registry import available_head_names, build_expression_head


def test_query_head_shapes_and_aux_outputs():
    torch.manual_seed(1337)

    head = QueryExpressionHead(
        sequence_embedding_dim=5,
        host_embedding_dim=3,
        fusion_dim=8,
        num_heads=2,
        num_sequence_slots=4,
        num_queries=2,
        hidden_dims=(6,),
        dropout=0.0,
        output_dim=1,
        include_sequence_skip=True,
        include_host_skip=False,
    )

    sequence_embedding = torch.randn(7, 5)
    host_embedding = torch.randn(7, 3)

    output = head(sequence_embedding, host_embedding, return_aux=True)

    assert isinstance(output, QueryExpressionOutput)
    assert output.expression.shape == (7, 1)
    assert output.sequence_slots.shape == (7, 4, 8)
    assert output.host_queries.shape == (7, 2, 8)
    assert output.attention_context.shape == (7, 2, 8)
    assert output.pooled_context.shape == (7, 8)

    assert output.attention_weights is not None
    assert output.attention_weights.shape == (7, 2, 2, 4)

    prediction = head(sequence_embedding, host_embedding)

    assert prediction.shape == (7, 1)


def test_query_head_can_flatten_query_outputs():
    head = QueryExpressionHead(
        sequence_embedding_dim=5,
        host_embedding_dim=3,
        fusion_dim=8,
        num_heads=2,
        num_sequence_slots=4,
        num_queries=3,
        dropout=0.0,
        output_dim=1,
        include_sequence_skip=False,
        include_host_skip=False,
        query_pooling="flatten",
    )

    assert head.input_dim == 3 * 8

    output = head(
        torch.randn(2, 5),
        torch.randn(2, 3),
        return_aux=True,
    )

    assert isinstance(output, QueryExpressionOutput)
    assert output.pooled_context.shape == (2, 3 * 8)
    assert output.expression.shape == (2, 1)


def test_query_head_host_query_controls_attention_and_prediction():
    head = QueryExpressionHead(
        sequence_embedding_dim=1,
        host_embedding_dim=2,
        fusion_dim=2,
        num_heads=1,
        num_sequence_slots=2,
        num_queries=1,
        dropout=0.0,
        output_dim=1,
        use_layer_norm=False,
        include_sequence_skip=False,
        include_host_skip=False,
    )

    with torch.no_grad():
        # Create two fixed sequence slots:
        # slot 0 = [1, 0]
        # slot 1 = [0, 1]
        head.sequence_to_slots.weight.zero_()
        head.sequence_to_slots.bias.copy_(torch.tensor([1.0, 0.0, 0.0, 1.0]))

        # Host embedding directly becomes the query.
        head.host_to_queries.weight.copy_(torch.eye(2))
        head.host_to_queries.bias.zero_()

        # Make MultiheadAttention projections identity for Q, K, V, and output.
        eye = torch.eye(2)
        head.attention.in_proj_weight.copy_(torch.cat((eye, eye, eye), dim=0))
        head.attention.in_proj_bias.zero_()
        head.attention.out_proj.weight.copy_(eye)
        head.attention.out_proj.bias.zero_()

        predictor = head.predictor[0]
        assert isinstance(predictor, nn.Linear)
        predictor.weight.copy_(torch.tensor([[1.0, -1.0]]))
        predictor.bias.zero_()

    sequence_embedding = torch.zeros(2, 1)
    host_embedding = torch.tensor(
        [
            [10.0, 0.0],
            [0.0, 10.0],
        ]
    )

    output = head(sequence_embedding, host_embedding, return_aux=True)

    assert isinstance(output, QueryExpressionOutput)
    assert output.attention_weights is not None

    # First host query attends to slot [1, 0], producing positive prediction.
    assert output.expression[0, 0] > 0.9

    # Second host query attends to slot [0, 1], producing negative prediction.
    assert output.expression[1, 0] < -0.9


def test_query_head_builds_from_registry():
    assert "query" in available_head_names()

    head = build_expression_head(
        "query",
        sequence_embedding_dim=5,
        host_embedding_dim=3,
        fusion_dim=8,
        query_num_heads=2,
        query_num_sequence_slots=4,
        query_num_queries=2,
        hidden_dims=(6,),
        dropout=0.0,
        output_dim=1,
    )

    assert isinstance(head, QueryExpressionHead)

    prediction = head(torch.randn(4, 5), torch.randn(4, 3))

    assert prediction.shape == (4, 1)


def test_query_head_requires_fusion_dim_divisible_by_heads():
    with pytest.raises(ValueError, match="fusion_dim must be divisible"):
        QueryExpressionHead(
            sequence_embedding_dim=5,
            host_embedding_dim=3,
            fusion_dim=10,
            num_heads=4,
        )


def test_query_head_rejects_invalid_pooling():
    with pytest.raises(ValueError, match="query_pooling"):
        QueryExpressionHead(
            sequence_embedding_dim=5,
            host_embedding_dim=3,
            fusion_dim=8,
            num_heads=2,
            query_pooling="max",
        )


@pytest.mark.parametrize(
    ("sequence_embedding", "host_embedding", "match"),
    [
        (
            torch.randn(2, 6),
            torch.randn(2, 3),
            "sequence_embedding has embedding dim 6, expected 5",
        ),
        (
            torch.randn(2, 5),
            torch.randn(2, 4),
            "host_embedding has embedding dim 4, expected 3",
        ),
        (
            torch.randn(2, 5, 1),
            torch.randn(2, 3),
            "sequence_embedding must be shaped",
        ),
        (
            torch.randn(2, 5),
            torch.randn(3, 3),
            "batch sizes must match",
        ),
    ],
)
def test_query_head_validates_inputs(
    sequence_embedding: torch.Tensor,
    host_embedding: torch.Tensor,
    match: str,
):
    head = QueryExpressionHead(
        sequence_embedding_dim=5,
        host_embedding_dim=3,
        fusion_dim=8,
        num_heads=2,
    )

    with pytest.raises(ValueError, match=match):
        head(sequence_embedding, host_embedding)