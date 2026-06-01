"""Wrapper around pretrained Geneformer checkpoints.

This wrapper expects Geneformer token IDs. Raw AnnData/expression-table tokenisation
should be handled in the data pipeline so this module remains model-only.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal, Mapping, Sequence

import torch
from torch import nn
from transformers import AutoModel, AutoModelForMaskedLM

PoolMode = Literal["mean", "cls", "none"]
TokenBatch = torch.Tensor | Sequence[int] | Sequence[Sequence[int]] | Mapping[str, Any]


def _resolve_torch_dtype(torch_dtype: str | torch.dtype | None) -> str | torch.dtype | None:
    if torch_dtype is None or isinstance(torch_dtype, torch.dtype):
        return torch_dtype
    if torch_dtype == "auto":
        return "auto"
    dtype = getattr(torch, torch_dtype, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unknown torch dtype: {torch_dtype!r}")
    return dtype


def _move_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _select_hidden_state(outputs: Any, layer: int) -> torch.Tensor:
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is not None:
        return hidden_states[layer]

    last_hidden_state = getattr(outputs, "last_hidden_state", None)
    if last_hidden_state is not None:
        return last_hidden_state

    encoder_last_hidden_state = getattr(outputs, "encoder_last_hidden_state", None)
    if encoder_last_hidden_state is not None:
        return encoder_last_hidden_state

    raise RuntimeError("Model output does not include hidden states or last_hidden_state.")


def _pool_hidden_states(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    pooling: PoolMode,
) -> torch.Tensor | None:
    if pooling == "none":
        return None

    if pooling == "cls":
        return hidden_states[:, 0, :]

    if pooling != "mean":
        raise ValueError(f"Unsupported pooling mode: {pooling!r}")

    mask = attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
    denominator = mask.sum(dim=1).clamp_min(1.0)

    return (hidden_states * mask).sum(dim=1) / denominator


@dataclass(frozen=True)
class GeneformerConfig:
    model_name_or_path: str = "external/Geneformer/Geneformer-V2-316M"
    subfolder: str | None = None
    max_length: int | None = None
    truncate: bool = True
    pooling: PoolMode = "mean"
    freeze_encoder: bool = True
    load_mlm_head: bool = True
    trust_remote_code: bool = False
    torch_dtype: str | torch.dtype | None = None
    cache_dir: str | None = None
    local_files_only: bool = False
    pad_token_id: int | None = None


class GeneformerWrapper(nn.Module):
    """Thin PyTorch wrapper for pretrained Geneformer encoders."""

    def __init__(self, config: GeneformerConfig | None = None, **overrides: Any) -> None:
        super().__init__()
        self.config = replace(config or GeneformerConfig(), **overrides)

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": self.config.trust_remote_code,
            "local_files_only": self.config.local_files_only,
        }

        if self.config.subfolder is not None:
            model_kwargs["subfolder"] = self.config.subfolder

        if self.config.cache_dir is not None:
            model_kwargs["cache_dir"] = self.config.cache_dir

        resolved_dtype = _resolve_torch_dtype(self.config.torch_dtype)
        if resolved_dtype is not None:
            model_kwargs["torch_dtype"] = resolved_dtype

        model_cls = AutoModelForMaskedLM if self.config.load_mlm_head else AutoModel

        self.model = model_cls.from_pretrained(
            self.config.model_name_or_path,
            **model_kwargs,
        )

        configured_pad_id = getattr(self.model.config, "pad_token_id", None)
        self.pad_token_id = (
            self.config.pad_token_id
            if self.config.pad_token_id is not None
            else configured_pad_id
        )

        if self.pad_token_id is None:
            self.pad_token_id = 0

        if self.config.freeze_encoder:
            self.freeze()

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def embedding_dim(self) -> int:
        for attr in ("hidden_size", "d_model", "embed_dim"):
            value = getattr(self.model.config, attr, None)
            if value is not None:
                return int(value)

        embedding_layer = self.model.get_input_embeddings()
        return int(getattr(embedding_layer, "embedding_dim", embedding_layer.weight.shape[-1]))

    def freeze(self) -> "GeneformerWrapper":
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        return self

    def unfreeze(self) -> "GeneformerWrapper":
        for parameter in self.model.parameters():
            parameter.requires_grad = True
        return self

    def _effective_max_length(self) -> int | None:
        if not self.config.truncate:
            return None

        if self.config.max_length is not None:
            return self.config.max_length

        max_positions = getattr(self.model.config, "max_position_embeddings", None)
        return int(max_positions) if max_positions is not None else None

    def collate_input_ids(
        self,
        tokenized_cells: torch.Tensor | Sequence[int] | Sequence[Sequence[int]],
    ) -> dict[str, torch.Tensor]:
        """Pad a batch of Geneformer token-id sequences."""
        max_length = self._effective_max_length()

        if torch.is_tensor(tokenized_cells):
            input_ids = tokenized_cells.long()

            if input_ids.ndim == 1:
                input_ids = input_ids.unsqueeze(0)

            if max_length is not None:
                input_ids = input_ids[:, :max_length]

            attention_mask = input_ids.ne(int(self.pad_token_id)).long()

            return _move_to_device(
                {"input_ids": input_ids, "attention_mask": attention_mask},
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
                ids = ids.tolist()

            ids = list(ids)  # type: ignore[arg-type]

            if max_length is not None:
                ids = ids[:max_length]

            sequences.append([int(token_id) for token_id in ids])

        padded_length = max(len(ids) for ids in sequences)

        input_ids = torch.full(
            (len(sequences), padded_length),
            fill_value=int(self.pad_token_id),
            dtype=torch.long,
        )
        attention_mask = torch.zeros_like(input_ids)

        for row_index, ids in enumerate(sequences):
            length = len(ids)
            input_ids[row_index, :length] = torch.tensor(ids, dtype=torch.long)
            attention_mask[row_index, :length] = 1

        return _move_to_device(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            self.device,
        )

    def forward(
        self,
        input_ids: TokenBatch,
        *,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        layer: int = -1,
        pooling: PoolMode | None = None,
        return_hidden_states: bool = False,
        **model_kwargs: Any,
    ) -> dict[str, Any]:
        if isinstance(input_ids, Mapping):
            batch = dict(input_ids)
            attention_mask = batch.get("attention_mask", attention_mask)
            token_type_ids = batch.get("token_type_ids", token_type_ids)
            input_ids = batch["input_ids"]

        if torch.is_tensor(input_ids):
            tensor_input_ids = input_ids.long()

            if tensor_input_ids.ndim == 1:
                tensor_input_ids = tensor_input_ids.unsqueeze(0)

            max_length = self._effective_max_length()

            if max_length is not None:
                tensor_input_ids = tensor_input_ids[:, :max_length]

                if attention_mask is not None:
                    attention_mask = attention_mask[:, :max_length]

            if attention_mask is None:
                attention_mask = tensor_input_ids.ne(int(self.pad_token_id)).long()

            input_ids = tensor_input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

        else:
            batch = self.collate_input_ids(input_ids)
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]

        call_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_hidden_states": True,
            "return_dict": True,
        }

        if token_type_ids is not None:
            call_kwargs["token_type_ids"] = token_type_ids.to(self.device)

        call_kwargs.update(_move_to_device(model_kwargs, self.device))

        outputs = self.model(**call_kwargs)

        token_embeddings = _select_hidden_state(outputs, layer=layer)

        pooled_embedding = _pool_hidden_states(
            token_embeddings,
            attention_mask=attention_mask,
            pooling=pooling or self.config.pooling,
        )

        return {
            "token_embeddings": token_embeddings,
            "last_hidden_state": token_embeddings,
            "pooled_embedding": pooled_embedding,
            "attention_mask": attention_mask,
            "hidden_states": getattr(outputs, "hidden_states", None) if return_hidden_states else None,
            "logits": getattr(outputs, "logits", None),
        }

    @torch.inference_mode()
    def embed(
        self,
        tokenized_cells: TokenBatch,
        *,
        batch_size: int = 8,
        layer: int = -1,
        pooling: PoolMode | None = None,
    ) -> torch.Tensor:
        """Return CPU pooled embeddings for Geneformer-tokenized cells."""
        was_training = self.training
        self.eval()

        if isinstance(tokenized_cells, Mapping):
            output = self.forward(tokenized_cells, layer=layer, pooling=pooling)
            pooled = output["pooled_embedding"]

            if pooled is None:
                raise ValueError("embed() requires pooling='mean' or pooling='cls'.")

            return pooled.detach().cpu()

        if torch.is_tensor(tokenized_cells):
            cells: torch.Tensor | list[Any] = tokenized_cells
            n_items = 1 if cells.ndim == 1 else cells.shape[0]
        else:
            cells = list(tokenized_cells)

            if len(cells) > 0 and isinstance(cells[0], int):
                cells = [cells]

            n_items = len(cells)

        embeddings: list[torch.Tensor] = []

        for start in range(0, n_items, batch_size):
            if torch.is_tensor(cells):
                chunk = cells if cells.ndim == 1 else cells[start : start + batch_size]
            else:
                chunk = cells[start : start + batch_size]

            output = self.forward(chunk, layer=layer, pooling=pooling)
            pooled = output["pooled_embedding"]

            if pooled is None:
                raise ValueError("embed() requires pooling='mean' or pooling='cls'.")

            embeddings.append(pooled.detach().cpu())

            if torch.is_tensor(cells) and cells.ndim == 1:
                break

        if was_training and not self.config.freeze_encoder:
            self.train()

        return torch.cat(embeddings, dim=0)


GeneformerEncoder = GeneformerWrapper

__all__ = [
    "GeneformerConfig",
    "GeneformerWrapper",
    "GeneformerEncoder",
]