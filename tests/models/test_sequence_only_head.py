from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from host_aware_predictor.models.fusion_heads import (
    SequenceOnlyExpressionHead,
    available_head_names,
    build_expression_head,
)
from host_aware_predictor.training.expression_head_trainer import enforce_host_specific_baseline


def test_sequence_only_head_uses_only_sequence_embeddings():
    head = SequenceOnlyExpressionHead(
        sequence_embedding_dim=2,
        host_embedding_dim=3,
        output_dim=1,
    )

    linear = head.network[0]
    assert isinstance(linear, nn.Linear)

    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[2.0, -1.0]]))
        linear.bias.copy_(torch.tensor([0.25]))

    sequence_embedding = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    host_embedding = torch.tensor(
        [
            [10.0, 20.0, 30.0],
            [40.0, 50.0, 60.0],
        ]
    )

    changed_host_embedding = host_embedding + 1000.0

    prediction = head(sequence_embedding, host_embedding)
    prediction_with_changed_host = head(sequence_embedding, changed_host_embedding)

    expected = sequence_embedding @ linear.weight.T + linear.bias

    assert prediction.shape == (2, 1)
    torch.testing.assert_close(prediction, expected)
    torch.testing.assert_close(prediction_with_changed_host, expected)


def test_sequence_only_head_accepts_missing_host_embedding_placeholder():
    head = SequenceOnlyExpressionHead(
        sequence_embedding_dim=2,
        host_embedding_dim=3,
        output_dim=1,
    )

    sequence_embedding = torch.randn(5, 2)

    prediction = head(sequence_embedding, host_embedding=None)

    assert prediction.shape == (5, 1)


def test_sequence_only_head_builds_from_registry_with_small_mlp():
    assert "sequence_only" in available_head_names()

    head = build_expression_head(
        "sequence_only",
        sequence_embedding_dim=2,
        host_embedding_dim=3,
        hidden_dims=(4,),
        dropout=0.0,
        output_dim=1,
    )

    assert isinstance(head, SequenceOnlyExpressionHead)
    assert isinstance(head.network[0], nn.Linear)
    assert isinstance(head.network[1], nn.GELU)
    assert isinstance(head.network[2], nn.Linear)

    prediction = head(torch.randn(5, 2), torch.randn(5, 3))

    assert prediction.shape == (5, 1)


@pytest.mark.parametrize("conditions", [None, [], ["K562", "HepG2"]])
def test_sequence_only_training_requires_exactly_one_condition(conditions):
    args = SimpleNamespace(head="sequence_only", conditions=conditions)

    with pytest.raises(ValueError, match="requires exactly one cell"):
        enforce_host_specific_baseline(args)


def test_sequence_only_training_accepts_exactly_one_condition():
    enforce_host_specific_baseline(SimpleNamespace(head="sequence_only", conditions=["K562"]))


def test_host_specific_guard_does_not_affect_other_heads():
    enforce_host_specific_baseline(SimpleNamespace(head="concat", conditions=None))