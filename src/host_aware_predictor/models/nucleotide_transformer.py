"""Wrapper around pretrained Nucleotide Transformer checkpoints.

The wrapper accepts raw DNA strings and returns token-level and pooled embeddings.
It is intentionally small: windowing/chunking long genomes should live in the
project data pipeline, not inside this model wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal, Mapping, Sequence

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, AutoTokenizer

PoolMode = Literal["mean", "cls", "none"]


def _resolve_torch_dtype(torch_dtype: str | torch.dtype | None) -> str | torch.dtype | None:
    if torch_dtype is None or isinstance(torch_dtype, torch.dtype):
        return torch_dtype
    if torch_dtype == "auto":
        return "auto"
    dtype = getattr(torch, torch_dtype, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unknown torch dtype: {torch_dtype!r}")
    return dtype



def _patch_esm_rotary_config(config: Any) -> Any:
    """Patch older NT/ESM configs for newer transformers ESM implementations.

    This is only a fallback for cases where transformers uses its built-in ESM.
    The preferred NT path is AutoModelForMaskedLM + trust_remote_code=True.
    """
    if getattr(config, "model_type", None) != "esm":
        return config

    defaults = {
        "rope_theta": 10000.0,
        "is_decoder": False,
        "add_cross_attention": False,
        "chunk_size_feed_forward": 0,
        "is_encoder_decoder": False,
        "use_cache": False,
    }

    for key, value in defaults.items():
        # Write both ways to avoid PretrainedConfig attribute-map edge cases.
        config.__dict__.setdefault(key, value)
        try:
            getattr(config, key)
        except AttributeError:
            object.__setattr__(config, key, value)
            config.__dict__[key] = value

    return config


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
    special_tokens_mask: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if pooling == "none":
        return None

    if pooling == "cls":
        return hidden_states[:, 0, :]

    if pooling != "mean":
        raise ValueError(f"Unsupported pooling mode: {pooling!r}")

    mask = attention_mask
    if special_tokens_mask is not None:
        mask = mask * (1 - special_tokens_mask.to(dtype=mask.dtype))

    mask = mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
    denominator = mask.sum(dim=1).clamp_min(1.0)

    return (hidden_states * mask).sum(dim=1) / denominator


@dataclass(frozen=True)
class NucleotideTransformerConfig:
    model_name_or_path: str = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
    max_length: int | None = None
    truncate: bool = True
    pooling: PoolMode = "mean"
    freeze_encoder: bool = True
    load_mlm_head: bool = True
    trust_remote_code: bool = True
    torch_dtype: str | torch.dtype | None = None
    cache_dir: str | None = None
    local_files_only: bool = False
    replace_ambiguous_bases_with_n: bool = True


class NucleotideTransformerWrapper(nn.Module):
    """Thin PyTorch wrapper for pretrained Nucleotide Transformer encoders."""

    def __init__(self, config: NucleotideTransformerConfig | None = None, **overrides: Any) -> None:
        super().__init__()
        self.config = replace(config or NucleotideTransformerConfig(), **overrides)

        tokenizer_kwargs: dict[str, Any] = {
            "trust_remote_code": self.config.trust_remote_code,
            "local_files_only": self.config.local_files_only,
        }
        model_kwargs: dict[str, Any] = dict(tokenizer_kwargs)

        if self.config.cache_dir is not None:
            tokenizer_kwargs["cache_dir"] = self.config.cache_dir
            model_kwargs["cache_dir"] = self.config.cache_dir

        resolved_dtype = _resolve_torch_dtype(self.config.torch_dtype)
        if resolved_dtype is not None:
            model_kwargs["torch_dtype"] = resolved_dtype

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path,
            **tokenizer_kwargs,
        )

        # Preferred NT path:
        # InstaDeep NT v2 checkpoints provide remote code for AutoModelForMaskedLM.
        # The MLM wrapper still returns hidden_states, which we use for embeddings.
        if self.config.load_mlm_head:
            self.model = AutoModelForMaskedLM.from_pretrained(
                self.config.model_name_or_path,
                **model_kwargs,
            )
        else:
            # Fallback path for checkpoints that support a bare encoder AutoModel.
            config_kwargs = {
                "trust_remote_code": self.config.trust_remote_code,
                "local_files_only": self.config.local_files_only,
            }
            if self.config.cache_dir is not None:
                config_kwargs["cache_dir"] = self.config.cache_dir

            hf_config = AutoConfig.from_pretrained(
                self.config.model_name_or_path,
                **config_kwargs,
            )
            hf_config = _patch_esm_rotary_config(hf_config)
            auto_map = getattr(hf_config, "auto_map", {}) or {}

            if "AutoModel" in auto_map:
                self.model = AutoModel.from_pretrained(
                    self.config.model_name_or_path,
                    config=hf_config,
                    **model_kwargs,
                )
            else:
                # If bare AutoModel would fall back to incompatible built-in ESM,
                # use MLM remote code and extract hidden states from it.
                self.model = AutoModelForMaskedLM.from_pretrained(
                    self.config.model_name_or_path,
                    **model_kwargs,
                )

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

    def freeze(self) -> "NucleotideTransformerWrapper":
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        return self

    def unfreeze(self) -> "NucleotideTransformerWrapper":
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

    def _normalise_sequence(self, sequence: str) -> str:
        sequence = "".join(sequence.upper().replace("U", "T").split())

        valid_bases = {"A", "C", "G", "T", "N"}
        invalid_bases = set(sequence) - valid_bases

        if invalid_bases and not self.config.replace_ambiguous_bases_with_n:
            raise ValueError(f"Invalid nucleotide characters: {sorted(invalid_bases)}")

        if invalid_bases:
            sequence = "".join(base if base in valid_bases else "N" for base in sequence)

        return sequence

    def tokenize(self, sequences: Sequence[str] | str) -> dict[str, torch.Tensor]:
        if isinstance(sequences, str):
            sequences = [sequences]

        cleaned_sequences = [self._normalise_sequence(sequence) for sequence in sequences]

        tokenizer_kwargs: dict[str, Any] = {
            "padding": True,
            "return_tensors": "pt",
        }

        max_length = self._effective_max_length()
        if max_length is not None:
            tokenizer_kwargs.update({"truncation": True, "max_length": max_length})

        try:
            batch = self.tokenizer(
                cleaned_sequences,
                return_special_tokens_mask=True,
                **tokenizer_kwargs,
            )
        except TypeError:
            batch = self.tokenizer(cleaned_sequences, **tokenizer_kwargs)

        batch = dict(batch)

        if "attention_mask" not in batch:
            batch["attention_mask"] = torch.ones_like(batch["input_ids"])

        if "special_tokens_mask" not in batch:
            batch["special_tokens_mask"] = torch.zeros_like(batch["input_ids"])

        return _move_to_device(batch, self.device)

    def forward(
        self,
        sequences: Sequence[str] | str | None = None,
        *,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        special_tokens_mask: torch.Tensor | None = None,
        layer: int = -1,
        pooling: PoolMode | None = None,
        return_hidden_states: bool = False,
        **model_kwargs: Any,
    ) -> dict[str, Any]:
        if sequences is not None:
            if input_ids is not None:
                raise ValueError("Pass either sequences or input_ids, not both.")

            batch = self.tokenize(sequences)
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            special_tokens_mask = batch.get("special_tokens_mask")

        elif input_ids is None:
            raise ValueError("Pass raw sequences or tokenized input_ids.")

        input_ids = input_ids.to(self.device)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        attention_mask = attention_mask.to(self.device)

        if special_tokens_mask is not None:
            special_tokens_mask = special_tokens_mask.to(self.device)

        model_kwargs = _move_to_device(model_kwargs, self.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            **model_kwargs,
        )

        token_embeddings = _select_hidden_state(outputs, layer=layer)

        pooled_embedding = _pool_hidden_states(
            token_embeddings,
            attention_mask=attention_mask,
            pooling=pooling or self.config.pooling,
            special_tokens_mask=special_tokens_mask,
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
        sequences: Sequence[str] | str,
        *,
        batch_size: int = 4,
        layer: int = -1,
        pooling: PoolMode | None = None,
    ) -> torch.Tensor:
        """Return CPU pooled embeddings for raw DNA sequences."""
        sequence_list = [sequences] if isinstance(sequences, str) else list(sequences)

        was_training = self.training
        self.eval()

        embeddings: list[torch.Tensor] = []

        for start in range(0, len(sequence_list), batch_size):
            output = self.forward(
                sequence_list[start : start + batch_size],
                layer=layer,
                pooling=pooling,
            )

            pooled = output["pooled_embedding"]
            if pooled is None:
                raise ValueError("embed() requires pooling='mean' or pooling='cls'.")

            embeddings.append(pooled.detach().cpu())

        if was_training and not self.config.freeze_encoder:
            self.train()

        return torch.cat(embeddings, dim=0)


NucleotideTransformerEncoder = NucleotideTransformerWrapper

__all__ = [
    "NucleotideTransformerConfig",
    "NucleotideTransformerWrapper",
    "NucleotideTransformerEncoder",
]