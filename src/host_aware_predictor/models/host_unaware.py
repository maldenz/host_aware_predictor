"""Host-unaware baseline heads.

This model intentionally uses sequence embeddings only. Host/transcriptome
embeddings are not part of this baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn


def _activation(name: str) -> nn.Module:
    lowered = name.lower()
    if lowered == "relu":
        return nn.ReLU()
    if lowered == "gelu":
        return nn.GELU()
    if lowered == "silu":
        return nn.SiLU()
    if lowered == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name!r}")


@dataclass(frozen=True)
class HostUnawareConfig:
    input_dim: int
    output_dim: int
    hidden_dims: tuple[int, ...] = (512, 128)
    dropout: float = 0.1
    activation: str = "gelu"


class HostUnawarePredictor(nn.Module):
    """MLP prediction head over frozen sequence-only embeddings."""

    def __init__(
        self,
        config: HostUnawareConfig | None = None,
        *,
        input_dim: int | None = None,
        output_dim: int | None = None,
        hidden_dims: Sequence[int] = (512, 128),
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()

        if config is None:
            if input_dim is None or output_dim is None:
                raise ValueError("Pass either config or both input_dim and output_dim.")
            config = HostUnawareConfig(
                input_dim=int(input_dim),
                output_dim=int(output_dim),
                hidden_dims=tuple(int(dim) for dim in hidden_dims),
                dropout=float(dropout),
                activation=activation,
            )

        self.config = config

        layers: list[nn.Module] = []
        previous_dim = config.input_dim
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(_activation(config.activation))
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            previous_dim = hidden_dim

        layers.append(nn.Linear(previous_dim, config.output_dim))
        self.head = nn.Sequential(*layers)

    def forward(self, sequence_embedding: torch.Tensor) -> torch.Tensor:
        return self.head(sequence_embedding.float())


HostUnawareModel = HostUnawarePredictor

__all__ = ["HostUnawareConfig", "HostUnawarePredictor", "HostUnawareModel"]
