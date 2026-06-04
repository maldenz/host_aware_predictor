from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch import nn

from host_aware_predictor.models.concat_head import (
    ConcatExpressionHead,
    ConcatExpressionOutput,
    FrozenConcatExpressionPredictor,
)


@dataclass(frozen=True)
class DummyEncoderOutput:
    pooled_embedding: torch.Tensor


class DummyFrozenEncoder(nn.Module):
    """Small encoder double used to verify frozen-embedder behavior."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.ones(embedding_dim))
        self.freeze_called = False
        self.forward_calls = 0
        self.grad_enabled_during_forward: list[bool] = []

    def freeze(self) -> "DummyFrozenEncoder":
        self.freeze_called = True
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad = False
        return self

    def forward(
        self,
        inputs: torch.Tensor | None = None,
        *,
        offset: float = 0.0,
    ) -> DummyEncoderOutput:
        self.forward_calls += 1
        self.grad_enabled_during_forward.append(torch.is_grad_enabled())

        if inputs is None:
            inputs = torch.zeros(2, self.embedding_dim)

        pooled_embedding = inputs.float() + self.weight + float(offset)
        return DummyEncoderOutput(pooled_embedding=pooled_embedding)


class DummyNoPooledEmbeddingEncoder(nn.Module):
    embedding_dim = 2

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, inputs: torch.Tensor):
        del inputs
        return object()


def test_concat_head_concatenates_embeddings_and_predicts_deterministically():
    head = ConcatExpressionHead(
        sequence_embedding_dim=2,
        host_embedding_dim=3,
        output_dim=1,
    )

    linear = head.network[0]
    assert isinstance(linear, nn.Linear)

    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]))
        linear.bias.copy_(torch.tensor([0.5]))

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

    prediction = head(sequence_embedding, host_embedding)

    expected_fused = torch.cat((sequence_embedding, host_embedding), dim=-1)
    expected = expected_fused @ linear.weight.T + linear.bias

    assert prediction.shape == (2, 1)
    torch.testing.assert_close(prediction, expected)


def test_concat_head_supports_small_mlp():
    head = ConcatExpressionHead(
        sequence_embedding_dim=2,
        host_embedding_dim=3,
        hidden_dims=(4,),
        dropout=0.0,
        output_dim=1,
    )

    sequence_embedding = torch.randn(5, 2)
    host_embedding = torch.randn(5, 3)

    prediction = head(sequence_embedding, host_embedding)

    assert prediction.shape == (5, 1)
    assert isinstance(head.network[0], nn.Linear)
    assert isinstance(head.network[1], nn.GELU)
    assert isinstance(head.network[2], nn.Linear)


@pytest.mark.parametrize(
    ("sequence_embedding", "host_embedding", "match"),
    [
        (
            torch.randn(2, 3),
            torch.randn(2, 3),
            "sequence_embedding has embedding dim 3, expected 2",
        ),
        (
            torch.randn(2, 2),
            torch.randn(2, 4),
            "host_embedding has embedding dim 4, expected 3",
        ),
        (
            torch.randn(2, 2, 1),
            torch.randn(2, 3),
            "sequence_embedding must be shaped",
        ),
        (
            torch.randn(2, 2),
            torch.randn(3, 3),
            "batch sizes must match",
        ),
    ],
)
def test_concat_head_validates_embedding_shapes(
    sequence_embedding: torch.Tensor,
    host_embedding: torch.Tensor,
    match: str,
):
    head = ConcatExpressionHead(
        sequence_embedding_dim=2,
        host_embedding_dim=3,
    )

    with pytest.raises(ValueError, match=match):
        head(sequence_embedding, host_embedding)


def test_concat_head_requires_floating_point_embeddings():
    head = ConcatExpressionHead(
        sequence_embedding_dim=2,
        host_embedding_dim=3,
    )

    with pytest.raises(TypeError, match="sequence_embedding must be floating-point"):
        head(
            torch.tensor([[1, 2]], dtype=torch.long),
            torch.randn(1, 3),
        )

    with pytest.raises(TypeError, match="host_embedding must be floating-point"):
        head(
            torch.randn(1, 2),
            torch.tensor([[1, 2, 3]], dtype=torch.long),
        )


def test_predictor_freezes_both_encoders_and_only_head_is_trainable():
    sequence_encoder = DummyFrozenEncoder(embedding_dim=2)
    host_encoder = DummyFrozenEncoder(embedding_dim=3)

    model = FrozenConcatExpressionPredictor(
        sequence_encoder=sequence_encoder,
        host_encoder=host_encoder,
    )

    assert sequence_encoder.freeze_called is True
    assert host_encoder.freeze_called is True

    assert all(not parameter.requires_grad for parameter in sequence_encoder.parameters())
    assert all(not parameter.requires_grad for parameter in host_encoder.parameters())

    assert model.trainable_parameter_names
    assert all(name.startswith("head.") for name in model.trainable_parameter_names)

    model.assert_only_head_trainable()

    model.train()

    assert model.training is True
    assert model.head.training is True
    assert sequence_encoder.training is False
    assert host_encoder.training is False
    assert all(not parameter.requires_grad for parameter in sequence_encoder.parameters())
    assert all(not parameter.requires_grad for parameter in host_encoder.parameters())


def test_predictor_runs_frozen_encoders_under_no_grad_and_backprops_only_into_head():
    sequence_encoder = DummyFrozenEncoder(embedding_dim=2)
    host_encoder = DummyFrozenEncoder(embedding_dim=3)

    model = FrozenConcatExpressionPredictor(
        sequence_encoder=sequence_encoder,
        host_encoder=host_encoder,
    )

    linear = model.head.network[0]
    assert isinstance(linear, nn.Linear)

    with torch.no_grad():
        linear.weight.fill_(1.0)
        linear.bias.zero_()

    sequence_inputs = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )
    host_inputs = torch.tensor(
        [
            [10.0, 20.0, 30.0],
            [40.0, 50.0, 60.0],
        ]
    )

    prediction = model(
        sequence_inputs=sequence_inputs,
        host_inputs=host_inputs,
    )

    expected_sequence_embedding = sequence_inputs + 1.0
    expected_host_embedding = host_inputs + 1.0
    expected = torch.cat(
        (expected_sequence_embedding, expected_host_embedding),
        dim=-1,
    ).sum(dim=-1, keepdim=True)

    torch.testing.assert_close(prediction, expected)

    loss = prediction.sum()
    loss.backward()

    assert sequence_encoder.forward_calls == 1
    assert host_encoder.forward_calls == 1
    assert sequence_encoder.grad_enabled_during_forward == [False]
    assert host_encoder.grad_enabled_during_forward == [False]

    assert sequence_encoder.weight.grad is None
    assert host_encoder.weight.grad is None
    assert linear.weight.grad is not None
    assert linear.bias.grad is not None


def test_predictor_accepts_precomputed_embeddings_but_detaches_them():
    sequence_encoder = DummyFrozenEncoder(embedding_dim=2)
    host_encoder = DummyFrozenEncoder(embedding_dim=3)

    model = FrozenConcatExpressionPredictor(
        sequence_encoder=sequence_encoder,
        host_encoder=host_encoder,
    )

    sequence_embedding = torch.randn(4, 2, requires_grad=True)
    host_embedding = torch.randn(4, 3, requires_grad=True)

    prediction = model(
        sequence_embedding=sequence_embedding,
        host_embedding=host_embedding,
    )

    assert prediction.shape == (4, 1)

    prediction.sum().backward()

    assert sequence_encoder.forward_calls == 0
    assert host_encoder.forward_calls == 0
    assert sequence_embedding.grad is None
    assert host_embedding.grad is None

    assert any(
        parameter.grad is not None
        for parameter in model.head.parameters()
    )


def test_predictor_forwards_encoder_kwargs_and_can_return_embeddings():
    sequence_encoder = DummyFrozenEncoder(embedding_dim=2)
    host_encoder = DummyFrozenEncoder(embedding_dim=3)

    model = FrozenConcatExpressionPredictor(
        sequence_encoder=sequence_encoder,
        host_encoder=host_encoder,
    )

    output = model(
        sequence_inputs=torch.zeros(2, 2),
        host_inputs=torch.zeros(2, 3),
        sequence_kwargs={"offset": 2.0},
        host_kwargs={"offset": 4.0},
        return_embeddings=True,
    )

    assert isinstance(output, ConcatExpressionOutput)
    assert output.expression.shape == (2, 1)
    torch.testing.assert_close(output.sequence_embedding, torch.full((2, 2), 3.0))
    torch.testing.assert_close(output.host_embedding, torch.full((2, 3), 5.0))


def test_predictor_raises_when_encoder_does_not_return_pooled_embedding():
    sequence_encoder = DummyNoPooledEmbeddingEncoder()
    host_encoder = DummyFrozenEncoder(embedding_dim=3)

    model = FrozenConcatExpressionPredictor(
        sequence_encoder=sequence_encoder,
        host_encoder=host_encoder,
        sequence_embedding_dim=2,
    )

    with pytest.raises(ValueError, match="sequence encoder did not return pooled_embedding"):
        model(
            sequence_inputs=torch.zeros(2, 2),
            host_inputs=torch.zeros(2, 3),
        )


def test_predictor_rejects_ambiguous_raw_inputs_and_precomputed_embeddings():
    model = FrozenConcatExpressionPredictor(
        sequence_encoder=DummyFrozenEncoder(embedding_dim=2),
        host_encoder=DummyFrozenEncoder(embedding_dim=3),
    )

    with pytest.raises(ValueError, match="either sequence_inputs or sequence_embedding"):
        model(
            sequence_inputs=torch.zeros(2, 2),
            sequence_embedding=torch.zeros(2, 2),
            host_inputs=torch.zeros(2, 3),
        )

    with pytest.raises(ValueError, match="either host_inputs or host_embedding"):
        model(
            sequence_inputs=torch.zeros(2, 2),
            host_inputs=torch.zeros(2, 3),
            host_embedding=torch.zeros(2, 3),
        )