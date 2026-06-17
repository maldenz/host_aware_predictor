"""Shared helpers for expression fusion heads."""

from __future__ import annotations

from typing import Any, Callable, Sequence

from torch import nn


ActivationFactory = Callable[[], nn.Module]


def validate_positive_int(value: int, *, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}.")
    return value


def normalise_hidden_dims(hidden_dims: Sequence[int] | None) -> tuple[int, ...]:
    if hidden_dims is None:
        return ()

    dims = tuple(int(dim) for dim in hidden_dims)
    for index, dim in enumerate(dims):
        validate_positive_int(dim, name=f"hidden_dims[{index}]")
    return dims


def validate_dropout(dropout: float) -> float:
    dropout = float(dropout)
    if not 0.0 <= dropout < 1.0:
        raise ValueError(f"dropout must be in [0, 1), got {dropout!r}.")
    return dropout


def activation_factory(name: str) -> ActivationFactory:
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


def build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int] | None,
    output_dim: int,
    *,
    dropout: float = 0.0,
    activation: str = "gelu",
) -> nn.Sequential:
    """Build the small MLP shared by fusion heads."""

    input_dim = validate_positive_int(input_dim, name="input_dim")
    output_dim = validate_positive_int(output_dim, name="output_dim")
    hidden_dims_tuple = normalise_hidden_dims(hidden_dims)
    dropout = validate_dropout(dropout)
    make_activation = activation_factory(activation)

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


def zero_init_last_linear(module: nn.Module) -> None:
    for submodule in reversed(list(module.modules())):
        if isinstance(submodule, nn.Linear):
            nn.init.zeros_(submodule.weight)
            if submodule.bias is not None:
                nn.init.zeros_(submodule.bias)
            return
    raise ValueError("Could not find a Linear layer to zero-initialize.")


def expression_head_config_dict(model: nn.Module) -> dict[str, Any]:
    """Return a JSON-serializable config dict for a fusion head."""

    config = getattr(model, "config", None)
    if config is None:
        return {}
    if hasattr(config, "__dict__"):
        return dict(config.__dict__)
    return {"repr": repr(config)}


__all__ = [
    "ActivationFactory",
    "activation_factory",
    "build_mlp",
    "expression_head_config_dict",
    "normalise_hidden_dims",
    "validate_dropout",
    "validate_positive_int",
    "zero_init_last_linear",
]
