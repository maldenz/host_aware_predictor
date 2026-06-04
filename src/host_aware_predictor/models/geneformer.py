"""Local Geneformer V2 encoder wrapper.

This module intentionally loads only local files:

- model/config/checkpoint assets from: external/Geneformer/Geneformer-V2-316M

The large model.safetensors file is expected to exist locally in that directory
but is not expected to be committed to git.

This wrapper is model-only. It expects Geneformer-tokenized transcriptomes
already represented as token IDs. Raw expression matrix tokenization should live
in a separate data/tokenization pipeline.

Default behavior:
- load the base BertModel via AutoModel
- do not load the MLM head
- do not instantiate the unused BERT pooler
- freeze the encoder
- return token-level and pooled host transcriptome embeddings
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM

PoolMode = Literal["mean", "cls", "first", "none"]
TokenBatch = torch.Tensor | Sequence[int] | Sequence[Sequence[int]] | Mapping[str, Any]


@dataclass(frozen=True)
class GeneformerConfig:
    """Configuration for the frozen local Geneformer transcriptome encoder."""

    model_dir: str | Path = Path("external/Geneformer/Geneformer-V2-316M")

    max_length: int | None = None
    pooling: PoolMode = "mean"
    layer: int = -1

    freeze_encoder: bool = True
    require_weights: bool = True
    local_files_only: bool = True
    trust_remote_code: bool = False
    torch_dtype: str | torch.dtype | None = None

    # False is the embedding path: load BertModel and avoid computing huge MLM logits.
    # True loads BertForMaskedLM from the same local checkpoint.
    load_mlm_head: bool = False

    # Geneformer V2 config uses pad_token_id=0. Keep overrideable for tests/future checkpoints.
    pad_token_id: int | None = None


@dataclass
class GeneformerOutput:
    """Output bundle from GeneformerEncoder.forward()."""

    token_embeddings: torch.Tensor
    pooled_embedding: torch.Tensor | None
    attention_mask: torch.Tensor
    input_ids: torch.Tensor
    token_type_ids: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None


def _project_root() -> Path:
    # src/host_aware_predictor/models/geneformer.py -> repo root
    try:
        return Path(__file__).resolve().parents[3]
    except IndexError:
        return Path.cwd()


def _resolve_repo_path(path: str | Path) -> Path:
    path = Path(path).expanduser()

    if path.is_absolute():
        return path.resolve()

    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    return (_project_root() / path).resolve()


def _has_local_weight_file(model_dir: Path) -> bool:
    patterns = (
        "model.safetensors",
        "model-*.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model-*.bin",
        "pytorch_model.bin.index.json",
    )
    return any(any(model_dir.glob(pattern)) for pattern in patterns)


def _validate_local_model_dir(model_dir: Path, *, require_weights: bool) -> None:
    if not model_dir.exists():
        raise FileNotFoundError(f"Geneformer model directory does not exist: {model_dir}")

    if not model_dir.is_dir():
        raise NotADirectoryError(f"Geneformer model path is not a directory: {model_dir}")

    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(f"Geneformer model directory is missing config.json: {model_dir}")

    if require_weights and not _has_local_weight_file(model_dir):
        raise FileNotFoundError(
            "No local Geneformer checkpoint weights found. Expected model.safetensors "
            f"or a sharded equivalent under: {model_dir}"
        )


def _resolve_torch_dtype(torch_dtype: str | torch.dtype | None) -> str | torch.dtype | None:
    if torch_dtype is None or isinstance(torch_dtype, torch.dtype):
        return torch_dtype

    if torch_dtype == "auto":
        return "auto"

    dtype = getattr(torch, str(torch_dtype), None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unknown torch dtype: {torch_dtype!r}")

    return dtype


def _move_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _coerce_2d_long_tensor(
    value: Any,
    *,
    name: str,
    max_length: int | None,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.long)

    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)

    if tensor.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D, got shape {tuple(tensor.shape)}")

    if max_length is not None:
        tensor = tensor[:, :max_length]

    if tensor.shape[1] == 0:
        raise ValueError(f"{name} has zero sequence length after truncation.")

    return tensor


def _coerce_optional_2d_long_tensor(
    value: Any,
    *,
    name: str,
    max_length: int | None,
) -> torch.Tensor | None:
    if value is None:
        return None

    return _coerce_2d_long_tensor(
        value,
        name=name,
        max_length=max_length,
    )


def _select_hidden_state(outputs: Any, layer: int) -> torch.Tensor:
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is not None:
        return tuple(hidden_states)[layer]

    last_hidden_state = getattr(outputs, "last_hidden_state", None)
    if last_hidden_state is not None:
        if layer not in {-1, 0}:
            raise RuntimeError(
                "Model output did not include hidden_states, so only the last hidden "
                f"state is available; requested layer={layer}."
            )
        return last_hidden_state

    encoder_last_hidden_state = getattr(outputs, "encoder_last_hidden_state", None)
    if encoder_last_hidden_state is not None:
        if layer not in {-1, 0}:
            raise RuntimeError(
                "Model output did not include hidden_states, so only the encoder last "
                f"hidden state is available; requested layer={layer}."
            )
        return encoder_last_hidden_state

    raise RuntimeError("Geneformer model output does not include hidden_states or last_hidden_state.")


def _pool_hidden_states(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    pooling: PoolMode,
) -> torch.Tensor | None:
    if pooling == "none":
        return None

    if pooling in {"cls", "first"}:
        return hidden_states[:, 0, :]

    if pooling != "mean":
        raise ValueError(f"Unsupported pooling mode: {pooling!r}")

    mask = attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
    denominator = mask.sum(dim=1).clamp_min(1.0)

    return (hidden_states * mask).sum(dim=1) / denominator


def _load_pretrained_local_model(
    model_cls: type,
    model_dir: Path,
    model_kwargs: dict[str, Any],
) -> nn.Module:
    """Load local HF model, preferring modern `dtype` while supporting older HF versions.

    Recent Transformers warns that `torch_dtype` is deprecated in favor of `dtype`.
    Older Transformers may not accept `dtype`, so fall back to `torch_dtype` only
    if needed.
    """
    try:
        return model_cls.from_pretrained(str(model_dir), **model_kwargs)
    except TypeError as exc:
        if "dtype" not in model_kwargs:
            raise

        fallback_kwargs = dict(model_kwargs)
        fallback_kwargs["torch_dtype"] = fallback_kwargs.pop("dtype")

        try:
            return model_cls.from_pretrained(str(model_dir), **fallback_kwargs)
        except TypeError:
            raise exc


class GeneformerEncoder(nn.Module):
    """Frozen local Geneformer encoder over tokenized host-cell transcriptomes."""

    def __init__(
        self,
        config: GeneformerConfig | None = None,
        **overrides: Any,
    ) -> None:
        super().__init__()

        self.config = replace(config or GeneformerConfig(), **overrides)

        if not self.config.local_files_only:
            raise ValueError("GeneformerEncoder is local-only; local_files_only must be True.")

        self.model_dir = _resolve_repo_path(self.config.model_dir)

        _validate_local_model_dir(
            self.model_dir,
            require_weights=self.config.require_weights,
        )

        config_kwargs: dict[str, Any] = {
            "local_files_only": True,
            "trust_remote_code": self.config.trust_remote_code,
        }

        self.hf_config = AutoConfig.from_pretrained(
            str(self.model_dir),
            **config_kwargs,
        )

        configured_pad_id = getattr(self.hf_config, "pad_token_id", None)
        self.pad_token_id = (
            int(self.config.pad_token_id)
            if self.config.pad_token_id is not None
            else int(configured_pad_id if configured_pad_id is not None else 0)
        )

        model_cls = AutoModelForMaskedLM if self.config.load_mlm_head else AutoModel

        model_kwargs: dict[str, Any] = {
            "config": self.hf_config,
            "local_files_only": True,
            "trust_remote_code": self.config.trust_remote_code,
        }

        # For the normal embedding path, avoid the unused BertModel pooler.
        # This removes the "pooler weights newly initialized" warning and saves a tiny bit of memory.
        if not self.config.load_mlm_head:
            model_kwargs["add_pooling_layer"] = False

        resolved_dtype = _resolve_torch_dtype(self.config.torch_dtype)
        if resolved_dtype is not None:
            # Modern Transformers accepts `dtype`; fallback loader handles older releases.
            model_kwargs["dtype"] = resolved_dtype

        self.model = _load_pretrained_local_model(
            model_cls=model_cls,
            model_dir=self.model_dir,
            model_kwargs=model_kwargs,
        )

        if self.config.freeze_encoder:
            self.freeze()

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def embedding_dim(self) -> int:
        for attr in ("hidden_size", "d_model", "embed_dim"):
            value = getattr(self.hf_config, attr, None)
            if value is not None:
                return int(value)

        input_embeddings = self.model.get_input_embeddings()
        return int(getattr(input_embeddings, "embedding_dim", input_embeddings.weight.shape[-1]))

    def train(self, mode: bool = True) -> "GeneformerEncoder":
        super().train(mode)

        # The encoder is frozen in the main host-aware predictor, so keep dropout
        # and other stochastic layers disabled even when the parent fusion model trains.
        if getattr(self, "config", None) is not None and self.config.freeze_encoder:
            if hasattr(self, "model"):
                self.model.eval()

        return self

    def freeze(self) -> "GeneformerEncoder":
        self.config = replace(self.config, freeze_encoder=True)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        return self

    def unfreeze(self) -> "GeneformerEncoder":
        self.config = replace(self.config, freeze_encoder=False)
        for parameter in self.model.parameters():
            parameter.requires_grad = True
        self.model.train(self.training)
        return self

    def _effective_max_length(self) -> int | None:
        if self.config.max_length is not None:
            max_length = int(self.config.max_length)
        else:
            max_positions = getattr(self.hf_config, "max_position_embeddings", None)
            max_length = int(max_positions) if max_positions is not None else None

        if max_length is not None and max_length <= 0:
            raise ValueError(f"max_length must be positive or None, got {max_length!r}")

        return max_length

    def collate_input_ids(
        self,
        tokenized_cells: torch.Tensor | Sequence[int] | Sequence[Sequence[int]],
    ) -> dict[str, torch.Tensor]:
        """Pad a batch of Geneformer token-id sequences.

        A single cell may be passed as a 1D tensor/list. A batch may be passed as
        a 2D tensor or a sequence of variable-length token-id sequences.
        """
        max_length = self._effective_max_length()

        if torch.is_tensor(tokenized_cells):
            input_ids = _coerce_2d_long_tensor(
                tokenized_cells,
                name="input_ids",
                max_length=max_length,
            )

            attention_mask = input_ids.ne(self.pad_token_id).long()

            return _move_to_device(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                },
                self.device,
            )

        cells = list(tokenized_cells)

        if len(cells) == 0:
            raise ValueError("tokenized_cells is empty.")

        if isinstance(cells[0], int):
            cells = [cells]  # type: ignore[list-item]

        sequences: list[list[int]] = []

        for ids in cells:  # type: ignore[assignment]
            if torch.is_tensor(ids):
                ids = ids.detach().cpu().tolist()

            ids = [int(token_id) for token_id in ids]  # type: ignore[arg-type]

            if max_length is not None:
                ids = ids[:max_length]

            if len(ids) == 0:
                raise ValueError("Encountered an empty Geneformer token-id sequence.")

            sequences.append(ids)

        padded_length = max(len(ids) for ids in sequences)
        if max_length is not None:
            padded_length = min(padded_length, max_length)

        input_ids = torch.full(
            (len(sequences), padded_length),
            fill_value=self.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros_like(input_ids)

        for row_index, ids in enumerate(sequences):
            ids = ids[:padded_length]
            length = len(ids)
            input_ids[row_index, :length] = torch.tensor(ids, dtype=torch.long)
            attention_mask[row_index, :length] = 1

        # Important: if caller includes explicit pad IDs inside a list, exclude them from pooling.
        attention_mask = attention_mask * input_ids.ne(self.pad_token_id).long()

        return _move_to_device(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
            self.device,
        )

    def _prepare_tensor_batch(
        self,
        input_ids: Any,
        *,
        attention_mask: Any = None,
        token_type_ids: Any = None,
    ) -> dict[str, torch.Tensor]:
        max_length = self._effective_max_length()

        tensor_input_ids = _coerce_2d_long_tensor(
            input_ids,
            name="input_ids",
            max_length=max_length,
        )

        tensor_attention_mask = _coerce_optional_2d_long_tensor(
            attention_mask,
            name="attention_mask",
            max_length=max_length,
        )

        if tensor_attention_mask is None:
            tensor_attention_mask = tensor_input_ids.ne(self.pad_token_id).long()

        if tensor_attention_mask.shape != tensor_input_ids.shape:
            raise ValueError(
                "attention_mask shape must match input_ids shape after truncation; "
                f"got attention_mask={tuple(tensor_attention_mask.shape)}, "
                f"input_ids={tuple(tensor_input_ids.shape)}"
            )

        tensor_token_type_ids = _coerce_optional_2d_long_tensor(
            token_type_ids,
            name="token_type_ids",
            max_length=max_length,
        )

        if tensor_token_type_ids is not None and tensor_token_type_ids.shape != tensor_input_ids.shape:
            raise ValueError(
                "token_type_ids shape must match input_ids shape after truncation; "
                f"got token_type_ids={tuple(tensor_token_type_ids.shape)}, "
                f"input_ids={tuple(tensor_input_ids.shape)}"
            )

        batch: dict[str, torch.Tensor] = {
            "input_ids": tensor_input_ids,
            "attention_mask": tensor_attention_mask,
        }

        if tensor_token_type_ids is not None:
            batch["token_type_ids"] = tensor_token_type_ids

        return _move_to_device(batch, self.device)

    def forward(
        self,
        input_ids: TokenBatch,
        *,
        attention_mask: torch.Tensor | Sequence[Sequence[int]] | Sequence[int] | None = None,
        token_type_ids: torch.Tensor | Sequence[Sequence[int]] | Sequence[int] | None = None,
        layer: int | None = None,
        pooling: PoolMode | None = None,
        return_hidden_states: bool = False,
        **model_kwargs: Any,
    ) -> GeneformerOutput:
        if isinstance(input_ids, Mapping):
            input_mapping = dict(input_ids)

            if "input_ids" not in input_mapping:
                raise KeyError("Geneformer input mapping must contain 'input_ids'.")

            attention_mask = input_mapping.get("attention_mask", attention_mask)
            token_type_ids = input_mapping.get("token_type_ids", token_type_ids)
            input_ids = input_mapping["input_ids"]

        if torch.is_tensor(input_ids):
            batch = self._prepare_tensor_batch(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

        elif attention_mask is not None or token_type_ids is not None:
            # Explicit masks require rectangular input that can be safely coerced to a tensor.
            try:
                batch = self._prepare_tensor_batch(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Pass tensor or rectangular list input_ids when providing explicit "
                    "attention_mask or token_type_ids."
                ) from exc

        else:
            batch = self.collate_input_ids(input_ids)  # type: ignore[arg-type]

        call_kwargs: dict[str, Any] = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "output_hidden_states": True,
            "return_dict": True,
        }

        if "token_type_ids" in batch:
            call_kwargs["token_type_ids"] = batch["token_type_ids"]

        call_kwargs.update(_move_to_device(dict(model_kwargs), self.device))

        grad_context = torch.no_grad() if self.config.freeze_encoder else nullcontext()

        with grad_context:
            outputs = self.model(**call_kwargs)

        layer_index = self.config.layer if layer is None else int(layer)
        token_embeddings = _select_hidden_state(outputs, layer=layer_index)

        if token_embeddings.shape[:2] != batch["attention_mask"].shape:
            raise RuntimeError(
                "Geneformer token embedding shape does not match attention_mask shape: "
                f"token_embeddings={tuple(token_embeddings.shape)}, "
                f"attention_mask={tuple(batch['attention_mask'].shape)}"
            )

        pooling_mode = self.config.pooling if pooling is None else pooling
        pooled_embedding = _pool_hidden_states(
            token_embeddings,
            attention_mask=batch["attention_mask"],
            pooling=pooling_mode,
        )

        hidden_states = getattr(outputs, "hidden_states", None)

        return GeneformerOutput(
            token_embeddings=token_embeddings,
            pooled_embedding=pooled_embedding,
            attention_mask=batch["attention_mask"],
            input_ids=batch["input_ids"],
            token_type_ids=batch.get("token_type_ids"),
            logits=getattr(outputs, "logits", None),
            hidden_states=tuple(hidden_states) if return_hidden_states and hidden_states is not None else None,
        )

    @torch.no_grad()
    def embed(
        self,
        tokenized_cells: TokenBatch,
        *,
        batch_size: int = 8,
        layer: int | None = None,
        pooling: PoolMode | None = None,
    ) -> torch.Tensor:
        """Return CPU pooled embeddings for Geneformer-tokenized host cells."""
        was_training = self.training
        self.eval()

        if isinstance(tokenized_cells, Mapping):
            output = self.forward(tokenized_cells, layer=layer, pooling=pooling)
            if output.pooled_embedding is None:
                raise ValueError("embed() requires pooling='mean', 'cls', or 'first'.")
            self.train(was_training)
            return output.pooled_embedding.detach().cpu()

        if torch.is_tensor(tokenized_cells):
            if tokenized_cells.ndim == 1:
                output = self.forward(tokenized_cells, layer=layer, pooling=pooling)
                if output.pooled_embedding is None:
                    raise ValueError("embed() requires pooling='mean', 'cls', or 'first'.")
                self.train(was_training)
                return output.pooled_embedding.detach().cpu()

            n_items = int(tokenized_cells.shape[0])
            cell_accessor: torch.Tensor | list[Any] = tokenized_cells

        else:
            cells = list(tokenized_cells)

            if len(cells) == 0:
                raise ValueError("tokenized_cells is empty.")

            if isinstance(cells[0], int):
                output = self.forward(cells, layer=layer, pooling=pooling)
                if output.pooled_embedding is None:
                    raise ValueError("embed() requires pooling='mean', 'cls', or 'first'.")
                self.train(was_training)
                return output.pooled_embedding.detach().cpu()

            n_items = len(cells)
            cell_accessor = cells

        embeddings: list[torch.Tensor] = []

        for start in range(0, n_items, batch_size):
            chunk = cell_accessor[start : start + batch_size]
            output = self.forward(chunk, layer=layer, pooling=pooling)

            if output.pooled_embedding is None:
                raise ValueError("embed() requires pooling='mean', 'cls', or 'first'.")

            embeddings.append(output.pooled_embedding.detach().cpu())

        self.train(was_training)
        return torch.cat(embeddings, dim=0)


GeneformerWrapper = GeneformerEncoder

__all__ = [
    "GeneformerConfig",
    "GeneformerEncoder",
    "GeneformerWrapper",
    "GeneformerOutput",
]