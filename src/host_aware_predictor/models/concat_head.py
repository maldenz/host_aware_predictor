"""Concat-fusion expression head and frozen-encoder predictor wrapper.

The concat head implementation now lives in ``fusion_heads.py`` so the same
training stack can build concat, FiLM, and future query heads from one registry.
This module is kept as a backwards-compatible import location.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn

from .fusion_heads import ConcatExpressionHead, ConcatExpressionHeadConfig


@dataclass(frozen=True)
class ConcatExpressionOutput:
    """Optional rich output for debugging or downstream analysis."""

    expression: torch.Tensor
    sequence_embedding: torch.Tensor
    host_embedding: torch.Tensor


def _freeze_encoder(encoder: nn.Module) -> None:
    """Freeze an encoder and keep stochastic layers disabled."""

    freeze = getattr(encoder, "freeze", None)
    if callable(freeze):
        freeze()

    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False


def _infer_embedding_dim(encoder: nn.Module, *, name: str) -> int:
    embedding_dim = getattr(encoder, "embedding_dim", None)
    if embedding_dim is None:
        raise ValueError(
            f"Could not infer {name} embedding dimension. "
            f"Pass {name}_embedding_dim explicitly."
        )

    embedding_dim = int(embedding_dim)
    if embedding_dim <= 0:
        raise ValueError(f"{name}_embedding_dim must be positive, got {embedding_dim!r}.")
    return embedding_dim


def _extract_pooled_embedding(output: Any, *, name: str) -> torch.Tensor:
    """Accept either a tensor directly or an encoder output with pooled_embedding."""

    if torch.is_tensor(output):
        pooled = output
    else:
        pooled = getattr(output, "pooled_embedding", None)

    if pooled is None:
        raise ValueError(
            f"{name} encoder did not return pooled_embedding. "
            "Use an encoder pooling mode such as 'mean', 'cls', or 'first'."
        )

    if not torch.is_tensor(pooled):
        raise TypeError(
            f"{name} pooled_embedding must be a torch.Tensor, "
            f"got {type(pooled).__name__}."
        )

    return pooled


def _run_frozen_encoder(
    encoder: nn.Module,
    inputs: Any,
    *,
    kwargs: Mapping[str, Any] | None,
    name: str,
) -> torch.Tensor:
    """Run a frozen encoder under no_grad and return detached pooled embeddings."""

    call_kwargs = dict(kwargs or {})

    with torch.no_grad():
        if inputs is None:
            if not call_kwargs:
                raise ValueError(
                    f"Pass either {name}_inputs or {name}_kwargs to compute "
                    f"{name} embeddings."
                )
            output = encoder(**call_kwargs)
        else:
            output = encoder(inputs, **call_kwargs)

    return _extract_pooled_embedding(output, name=name).detach()


class FrozenConcatExpressionPredictor(nn.Module):
    """Frozen embedders plus trainable concat expression head.

    This wrapper enforces the intended training contract:

    - sequence encoder: frozen
    - host encoder: frozen
    - concat head: trainable
    """

    def __init__(
        self,
        sequence_encoder: nn.Module,
        host_encoder: nn.Module,
        *,
        sequence_embedding_dim: int | None = None,
        host_embedding_dim: int | None = None,
        hidden_dims: tuple[int, ...] | list[int] | None = None,
        dropout: float = 0.0,
        output_dim: int = 1,
        head: ConcatExpressionHead | None = None,
    ) -> None:
        super().__init__()

        self.sequence_encoder = sequence_encoder
        self.host_encoder = host_encoder

        _freeze_encoder(self.sequence_encoder)
        _freeze_encoder(self.host_encoder)

        if head is None:
            if sequence_embedding_dim is None:
                sequence_embedding_dim = _infer_embedding_dim(
                    self.sequence_encoder,
                    name="sequence",
                )
            if host_embedding_dim is None:
                host_embedding_dim = _infer_embedding_dim(
                    self.host_encoder,
                    name="host",
                )

            head = ConcatExpressionHead(
                sequence_embedding_dim=sequence_embedding_dim,
                host_embedding_dim=host_embedding_dim,
                hidden_dims=hidden_dims,
                dropout=dropout,
                output_dim=output_dim,
            )

        self.head = head
        self.assert_only_head_trainable()

    @property
    def trainable_parameter_names(self) -> list[str]:
        return [
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        ]

    def assert_only_head_trainable(self) -> None:
        trainable_non_head = [
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad and not name.startswith("head.")
        ]

        if trainable_non_head:
            raise RuntimeError(
                "Only concat head parameters may be trainable. "
                f"Found trainable non-head parameters: {trainable_non_head}."
            )

    def train(self, mode: bool = True) -> "FrozenConcatExpressionPredictor":
        super().train(mode)

        # Calling parent.train() would normally put submodules in train mode.
        # Immediately re-freeze/re-eval the embedders so only the head trains.
        _freeze_encoder(self.sequence_encoder)
        _freeze_encoder(self.host_encoder)

        return self

    def _head_device_and_dtype(self) -> tuple[torch.device, torch.dtype]:
        parameter = next(self.head.parameters())
        return parameter.device, parameter.dtype

    def _prepare_embedding_for_head(self, embedding: torch.Tensor) -> torch.Tensor:
        device, dtype = self._head_device_and_dtype()
        return embedding.detach().to(device=device, dtype=dtype)

    def forward(
        self,
        sequence_inputs: Any = None,
        host_inputs: Any = None,
        *,
        sequence_embedding: torch.Tensor | None = None,
        host_embedding: torch.Tensor | None = None,
        sequence_kwargs: Mapping[str, Any] | None = None,
        host_kwargs: Mapping[str, Any] | None = None,
        return_embeddings: bool = False,
    ) -> torch.Tensor | ConcatExpressionOutput:
        """Predict expression from raw inputs or precomputed embeddings."""

        if sequence_embedding is not None and sequence_inputs is not None:
            raise ValueError("Pass either sequence_inputs or sequence_embedding, not both.")

        if host_embedding is not None and host_inputs is not None:
            raise ValueError("Pass either host_inputs or host_embedding, not both.")

        if sequence_embedding is None:
            sequence_embedding = _run_frozen_encoder(
                self.sequence_encoder,
                sequence_inputs,
                kwargs=sequence_kwargs,
                name="sequence",
            )

        if host_embedding is None:
            host_embedding = _run_frozen_encoder(
                self.host_encoder,
                host_inputs,
                kwargs=host_kwargs,
                name="host",
            )

        sequence_embedding = self._prepare_embedding_for_head(sequence_embedding)
        host_embedding = self._prepare_embedding_for_head(host_embedding)

        expression = self.head(sequence_embedding, host_embedding)

        if return_embeddings:
            return ConcatExpressionOutput(
                expression=expression,
                sequence_embedding=sequence_embedding,
                host_embedding=host_embedding,
            )

        return expression


__all__ = [
    "ConcatExpressionHead",
    "ConcatExpressionHeadConfig",
    "ConcatExpressionOutput",
    "FrozenConcatExpressionPredictor",
]
