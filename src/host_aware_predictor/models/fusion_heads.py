"""Shared expression heads for frozen DNA and host embeddings.

All heads in this module expose the same forward contract:

    prediction = head(sequence_embedding=seq, host_embedding=host)

where ``seq`` and ``host`` are rank-2 floating point tensors shaped
``[batch, embedding_dim]``.  This keeps the training loop independent of the
fusion mechanism.  A future query/cross-attention head should only need to add a
new class plus one entry in ``HEAD_REGISTRY``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import nn


ActivationFactory = Callable[[], nn.Module]


def _validate_positive_int(value: int, *, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}.")
    return value


def _normalise_hidden_dims(hidden_dims: Sequence[int] | None) -> tuple[int, ...]:
    if hidden_dims is None:
        return ()

    dims = tuple(int(dim) for dim in hidden_dims)
    for index, dim in enumerate(dims):
        _validate_positive_int(dim, name=f"hidden_dims[{index}]")
    return dims


def _validate_dropout(dropout: float) -> float:
    dropout = float(dropout)
    if not 0.0 <= dropout < 1.0:
        raise ValueError(f"dropout must be in [0, 1), got {dropout!r}.")
    return dropout


def _activation_factory(name: str) -> ActivationFactory:
    name = str(name).lower()
    if name == "gelu":
        return nn.GELU
    if name == "relu":
        return nn.ReLU
    if name == "silu":
        return nn.SiLU
    if name == "tanh":
        return nn.Tanh
    raise ValueError(f"Unsupported activation {name!r}. Use gelu, relu, silu, or tanh.")


def _build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int] | None,
    output_dim: int,
    *,
    dropout: float = 0.0,
    activation: str = "gelu",
) -> nn.Sequential:
    """Build the small MLP used by concat, FiLM, and future fusion heads."""

    input_dim = _validate_positive_int(input_dim, name="input_dim")
    output_dim = _validate_positive_int(output_dim, name="output_dim")
    hidden_dims_tuple = _normalise_hidden_dims(hidden_dims)
    dropout = _validate_dropout(dropout)
    make_activation = _activation_factory(activation)

    dims = (input_dim, *hidden_dims_tuple, output_dim)
    layers: list[nn.Module] = []
    for layer_index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(in_dim, out_dim))
        is_last_layer = layer_index == len(dims) - 2
        if not is_last_layer:
            layers.append(make_activation())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))

    return nn.Sequential(*layers)


def _zero_init_last_linear(module: nn.Module) -> None:
    for submodule in reversed(list(module.modules())):
        if isinstance(submodule, nn.Linear):
            nn.init.zeros_(submodule.weight)
            if submodule.bias is not None:
                nn.init.zeros_(submodule.bias)
            return
    raise ValueError("Could not find a Linear layer to zero-initialize.")


@dataclass(frozen=True)
class ConcatExpressionHeadConfig:
    """Configuration for the trainable concat expression head."""

    sequence_embedding_dim: int
    host_embedding_dim: int
    hidden_dims: tuple[int, ...] = ()
    dropout: float = 0.0
    output_dim: int = 1
    activation: str = "gelu"

@dataclass(frozen=True)
class SequenceOnlyExpressionHeadConfig:
    """Configuration for a host-specific sequence-only baseline head."""

    sequence_embedding_dim: int
    host_embedding_dim: int
    hidden_dims: tuple[int, ...] = ()
    dropout: float = 0.0
    output_dim: int = 1
    activation: str = "gelu"

@dataclass(frozen=True)
class FiLMExpressionHeadConfig:
    """Configuration for host-conditioned FiLM fusion.

    FiLM applies feature-wise affine modulation to a projected sequence vector:

        z_seq = project(sequence_embedding)
        gamma, beta = film_generator(host_embedding)
        z_mod = z_seq * (1 + gamma_scale * gamma) + beta
        expression = predictor(z_mod)

    ``identity_init=True`` initializes the final FiLM-generator layer to zero,
    so the head starts as a sequence-only predictor and learns host modulation
    from a stable identity point.
    """

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


class ConcatExpressionHead(EmbeddingFusionHead):
    """Trainable MLP over concatenated sequence and host embeddings."""

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

        self.sequence_embedding_dim = _validate_positive_int(
            sequence_embedding_dim,
            name="sequence_embedding_dim",
        )
        self.host_embedding_dim = _validate_positive_int(host_embedding_dim, name="host_embedding_dim")
        output_dim = _validate_positive_int(output_dim, name="output_dim")
        hidden_dims_tuple = _normalise_hidden_dims(hidden_dims)
        dropout = _validate_dropout(dropout)

        self.config = ConcatExpressionHeadConfig(
            sequence_embedding_dim=self.sequence_embedding_dim,
            host_embedding_dim=self.host_embedding_dim,
            hidden_dims=hidden_dims_tuple,
            dropout=dropout,
            output_dim=output_dim,
            activation=activation,
        )

        self.network = _build_mlp(
            self.sequence_embedding_dim + self.host_embedding_dim,
            hidden_dims_tuple,
            output_dim,
            dropout=dropout,
            activation=activation,
        )

    @property
    def input_dim(self) -> int:
        return self.sequence_embedding_dim + self.host_embedding_dim

    def forward(
        self,
        sequence_embedding: torch.Tensor,
        host_embedding: torch.Tensor,
    ) -> torch.Tensor:
        sequence_embedding, host_embedding = self._validate_pair(sequence_embedding, host_embedding)
        fused = torch.cat((sequence_embedding, host_embedding), dim=-1)
        return self.network(fused)

class SequenceOnlyExpressionHead(EmbeddingFusionHead):
    """Trainable MLP over sequence embeddings only.

    This head is intended as a host-specific baseline. It preserves the shared
    forward(sequence_embedding, host_embedding) interface used by the training
    loop, but deliberately ignores host_embedding so predictions depend only on
    the sequence embedding.
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

        self.sequence_embedding_dim = _validate_positive_int(
            sequence_embedding_dim,
            name="sequence_embedding_dim",
        )

        # Stored for config/checkpoint compatibility with the other heads.
        # It is intentionally not used during forward().
        self.host_embedding_dim = _validate_positive_int(
            host_embedding_dim,
            name="host_embedding_dim",
        )

        output_dim = _validate_positive_int(output_dim, name="output_dim")
        hidden_dims_tuple = _normalise_hidden_dims(hidden_dims)
        dropout = _validate_dropout(dropout)

        self.config = SequenceOnlyExpressionHeadConfig(
            sequence_embedding_dim=self.sequence_embedding_dim,
            host_embedding_dim=self.host_embedding_dim,
            hidden_dims=hidden_dims_tuple,
            dropout=dropout,
            output_dim=output_dim,
            activation=activation,
        )

        self.network = _build_mlp(
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


class FiLMExpressionHead(EmbeddingFusionHead):
    """Host-conditioned feature-wise linear modulation head.

    This is the next step after concat.  Instead of directly concatenating host
    and sequence embeddings, the host embedding produces feature-wise ``gamma``
    and ``beta`` parameters that modulate a projected sequence representation.
    The predictor MLP is then shared in spirit with the concat head: it consumes
    one fused vector and emits one scalar expression prediction.
    """

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

        self.sequence_embedding_dim = _validate_positive_int(
            sequence_embedding_dim,
            name="sequence_embedding_dim",
        )
        self.host_embedding_dim = _validate_positive_int(host_embedding_dim, name="host_embedding_dim")
        if fusion_dim is None:
            fusion_dim = self.sequence_embedding_dim
        self.fusion_dim = _validate_positive_int(fusion_dim, name="fusion_dim")
        output_dim = _validate_positive_int(output_dim, name="output_dim")
        hidden_dims_tuple = _normalise_hidden_dims(hidden_dims)
        if film_hidden_dims is None:
            film_hidden_dims_tuple = (self.fusion_dim,)
        else:
            film_hidden_dims_tuple = _normalise_hidden_dims(film_hidden_dims)
        dropout = _validate_dropout(dropout)
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

        self.host_to_film = _build_mlp(
            self.host_embedding_dim,
            film_hidden_dims_tuple,
            2 * self.fusion_dim,
            dropout=dropout,
            activation=activation,
        )
        if identity_init:
            _zero_init_last_linear(self.host_to_film)

        predictor_input_dim = self.fusion_dim
        self.host_skip_projection: nn.Module | None = None
        if include_host_skip:
            self.host_skip_projection = nn.Sequential(
                nn.Linear(self.host_embedding_dim, self.fusion_dim),
                nn.LayerNorm(self.fusion_dim) if use_layer_norm else nn.Identity(),
            )
            predictor_input_dim += self.fusion_dim

        self.predictor = _build_mlp(
            predictor_input_dim,
            hidden_dims_tuple,
            output_dim,
            dropout=dropout,
            activation=activation,
        )

    @property
    def input_dim(self) -> int:
        return self.fusion_dim + (self.fusion_dim if self.config.include_host_skip else 0)

    def forward(
        self,
        sequence_embedding: torch.Tensor,
        host_embedding: torch.Tensor,
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | FiLMExpressionOutput:
        sequence_embedding, host_embedding = self._validate_pair(sequence_embedding, host_embedding)

        sequence_features = self.sequence_norm(self.sequence_projection(sequence_embedding))
        gamma_beta = self.host_to_film(host_embedding)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        modulated = sequence_features * (1.0 + self.config.gamma_scale * gamma) + beta

        predictor_input = modulated
        if self.host_skip_projection is not None:
            host_skip = self.host_skip_projection(host_embedding)
            predictor_input = torch.cat((predictor_input, host_skip), dim=-1)

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


HEAD_REGISTRY: Mapping[str, type[EmbeddingFusionHead]] = {
    "concat": ConcatExpressionHead,
    "film": FiLMExpressionHead,
    "sequence_only": SequenceOnlyExpressionHead,
    # Future example:
    # "query": QueryExpressionHead,
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
    **extra_head_kwargs: Any,
) -> EmbeddingFusionHead:
    """Instantiate a registered expression head by name.

    The train script calls this function once and then treats every head through
    the same ``forward(sequence_embedding, host_embedding)`` interface.
    """

    head = str(head).lower()
    if head == "concat":
        if extra_head_kwargs:
            raise ValueError(f"Unsupported extra kwargs for concat head: {sorted(extra_head_kwargs)}")
        return ConcatExpressionHead(
            sequence_embedding_dim=sequence_embedding_dim,
            host_embedding_dim=host_embedding_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            output_dim=output_dim,
            activation=activation,
        )
    
    if head == "sequence_only":
        if extra_head_kwargs:
            raise ValueError(f"Unsupported extra kwargs for sequence_only head: {sorted(extra_head_kwargs)}")
        return SequenceOnlyExpressionHead(
            sequence_embedding_dim=sequence_embedding_dim,
            host_embedding_dim=host_embedding_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            output_dim=output_dim,
            activation=activation,
        )

    if head == "film":
        if extra_head_kwargs:
            raise ValueError(f"Unsupported extra kwargs for FiLM head: {sorted(extra_head_kwargs)}")
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

    raise ValueError(f"Unknown head {head!r}. Available heads: {available_head_names()}.")


def expression_head_config_dict(model: nn.Module) -> dict[str, Any]:
    """Return a JSON-serializable config dict for a fusion head."""

    config = getattr(model, "config", None)
    if config is None:
        return {}
    if hasattr(config, "__dict__"):
        return dict(config.__dict__)
    return {"repr": repr(config)}


__all__ = [
    "ConcatExpressionHead",
    "ConcatExpressionHeadConfig",
    "EmbeddingFusionHead",
    "FiLMExpressionHead",
    "FiLMExpressionHeadConfig",
    "FiLMExpressionOutput",
    "HEAD_REGISTRY",
    "SequenceOnlyExpressionHead",
    "SequenceOnlyExpressionHeadConfig",
    "available_head_names",
    "build_expression_head",
    "expression_head_config_dict",
]
