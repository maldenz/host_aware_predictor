"""FiLM expression head."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from .base import EmbeddingFusionHead
from .utils import (
    build_mlp,
    normalise_hidden_dims,
    validate_dropout,
    validate_positive_int,
    zero_init_last_linear,
)


@dataclass(frozen=True)
class FiLMExpressionHeadConfig:
    """Configuration for host-conditioned FiLM fusion."""

    sequence_embedding_dim: int
    host_embedding_dim: int
    fusion_dim: int
    hidden_dims: tuple[int, ...] = ()
    film_hidden_dims: tuple[int, ...] = ()
    dropout: float = 0.0
    output_dim: int = 1
    activation: str = "gelu"
    use_layer_norm: bool = True
    gamma_scale: float = 1.0
    include_host_skip: bool = False
    identity_init: bool = True


@dataclass(frozen=True)
class FiLMExpressionOutput:
    """Optional rich FiLM output for debugging and attribution."""

    expression: torch.Tensor
    sequence_features: torch.Tensor
    gamma: torch.Tensor
    beta: torch.Tensor
    modulated_sequence_features: torch.Tensor


class FiLMExpressionHead(EmbeddingFusionHead):
    """Host-conditioned feature-wise linear modulation head."""

    def __init__(
        self,
        sequence_embedding_dim: int,
        host_embedding_dim: int,
        *,
        fusion_dim: int | None = None,
        hidden_dims: Sequence[int] | None = None,
        film_hidden_dims: Sequence[int] | None = None,
        dropout: float = 0.0,
        output_dim: int = 1,
        activation: str = "gelu",
        use_layer_norm: bool = True,
        gamma_scale: float = 1.0,
        include_host_skip: bool = False,
        identity_init: bool = True,
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

        output_dim = validate_positive_int(output_dim, name="output_dim")
        hidden_dims_tuple = normalise_hidden_dims(hidden_dims)
        film_hidden_dims_tuple = (
            (self.fusion_dim,) if film_hidden_dims is None else normalise_hidden_dims(film_hidden_dims)
        )
        dropout = validate_dropout(dropout)
        gamma_scale = float(gamma_scale)
        if gamma_scale < 0.0:
            raise ValueError(f"gamma_scale must be non-negative, got {gamma_scale!r}.")

        self.config = FiLMExpressionHeadConfig(
            sequence_embedding_dim=self.sequence_embedding_dim,
            host_embedding_dim=self.host_embedding_dim,
            fusion_dim=self.fusion_dim,
            hidden_dims=hidden_dims_tuple,
            film_hidden_dims=film_hidden_dims_tuple,
            dropout=dropout,
            output_dim=output_dim,
            activation=activation,
            use_layer_norm=bool(use_layer_norm),
            gamma_scale=gamma_scale,
            include_host_skip=bool(include_host_skip),
            identity_init=bool(identity_init),
        )

        self.sequence_projection = nn.Linear(self.sequence_embedding_dim, self.fusion_dim)
        self.sequence_norm = nn.LayerNorm(self.fusion_dim) if use_layer_norm else nn.Identity()
        self.host_to_film = build_mlp(
            self.host_embedding_dim,
            film_hidden_dims_tuple,
            2 * self.fusion_dim,
            dropout=dropout,
            activation=activation,
        )
        if identity_init:
            zero_init_last_linear(self.host_to_film)

        predictor_input_dim = self.fusion_dim
        self.host_skip_projection: nn.Module | None = None
        if include_host_skip:
            self.host_skip_projection = nn.Sequential(
                nn.Linear(self.host_embedding_dim, self.fusion_dim),
                nn.LayerNorm(self.fusion_dim) if use_layer_norm else nn.Identity(),
            )
            predictor_input_dim += self.fusion_dim

        self.predictor = build_mlp(
            predictor_input_dim,
            hidden_dims_tuple,
            output_dim,
            dropout=dropout,
            activation=activation,
        )

    @property
    def input_dim(self) -> int:
        host_skip_dim = self.fusion_dim if self.config.include_host_skip else 0
        return self.fusion_dim + host_skip_dim

    def forward(
        self,
        sequence_embedding: torch.Tensor,
        host_embedding: torch.Tensor,
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | FiLMExpressionOutput:
        sequence_embedding, host_embedding = self._validate_pair(sequence_embedding, host_embedding)

        sequence_features = self.sequence_norm(self.sequence_projection(sequence_embedding))
        gamma, beta = self.host_to_film(host_embedding).chunk(2, dim=-1)
        modulated = sequence_features * (1.0 + self.config.gamma_scale * gamma) + beta

        predictor_input = modulated
        if self.host_skip_projection is not None:
            predictor_input = torch.cat((predictor_input, self.host_skip_projection(host_embedding)), dim=-1)

        expression = self.predictor(predictor_input)
        if return_aux:
            return FiLMExpressionOutput(
                expression=expression,
                sequence_features=sequence_features,
                gamma=gamma,
                beta=beta,
                modulated_sequence_features=modulated,
            )
        return expression


__all__ = [
    "FiLMExpressionHead",
    "FiLMExpressionHeadConfig",
    "FiLMExpressionOutput",
]
