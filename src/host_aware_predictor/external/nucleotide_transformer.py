"""Local NTv3/Nucleotide Transformer encoder wrapper.

This module intentionally loads only local files:

- model/config/tokenizer assets from: external/NTv3/NTv3_100M_pre
- Python model/tokenizer implementation from: external/NTv3/ntv3_base_model

No Hugging Face Hub download path is used. The large model.safetensors file is
expected to exist locally but is not expected to be committed to git.
"""

from __future__ import annotations

import importlib
import json
import math
import sys
from contextlib import nullcontext
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Sequence

import torch
import torch.nn.functional as F
from torch import nn

PoolMode = Literal["mean", "cls", "first", "none"]


@dataclass(frozen=True)
class NucleotideTransformerConfig:
    """Configuration for the frozen local NTv3 sequence encoder."""

    model_dir: str | Path = Path("external/NTv3/NTv3_100M_pre")
    base_model_dir: str | Path = Path("external/NTv3/ntv3_base_model")

    max_length: int | None = None
    pad_to_multiple_of: int | None = 128
    pooling: PoolMode = "mean"
    layer: int = -1

    freeze_encoder: bool = True
    require_weights: bool = True
    local_files_only: bool = True
    torch_dtype: str | torch.dtype | None = None

    replace_ambiguous_bases_with_n: bool = True


@dataclass
class NucleotideTransformerOutput:
    """Output bundle from NucleotideTransformerEncoder.forward()."""

    token_embeddings: torch.Tensor
    pooled_embedding: torch.Tensor | None
    attention_mask: torch.Tensor
    input_ids: torch.Tensor
    logits: torch.Tensor | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None


def _project_root() -> Path:
    # src/host_aware_predictor/models/nucleotide_transformer.py -> repo root
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


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _has_local_weight_file(model_dir: Path) -> bool:
    patterns = (
        "model.safetensors",
        "model-*.safetensors",
        "pytorch_model.bin",
        "pytorch_model-*.bin",
    )
    return any(any(model_dir.glob(pattern)) for pattern in patterns)


def _validate_local_model_dir(model_dir: Path, *, require_weights: bool) -> None:
    if not model_dir.exists():
        raise FileNotFoundError(f"NTv3 model directory does not exist: {model_dir}")

    if not model_dir.is_dir():
        raise NotADirectoryError(f"NTv3 model path is not a directory: {model_dir}")

    missing = [
        name
        for name in ("config.json", "tokenizer_config.json")
        if not (model_dir / name).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"NTv3 model directory {model_dir} is missing required file(s): {missing}"
        )

    if require_weights and not _has_local_weight_file(model_dir):
        raise FileNotFoundError(
            "No local NTv3 checkpoint weights found. Expected model.safetensors "
            f"or a sharded equivalent under: {model_dir}"
        )


def _validate_local_base_model_dir(base_model_dir: Path) -> None:
    if not base_model_dir.exists():
        raise FileNotFoundError(f"NTv3 base model directory does not exist: {base_model_dir}")

    required = (
        "configuration_ntv3_pretrained.py",
        "modeling_ntv3_pretrained.py",
        "tokenization_ntv3.py",
    )
    missing = [name for name in required if not (base_model_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"NTv3 base model directory {base_model_dir} is missing required file(s): {missing}"
        )


@lru_cache(maxsize=8)
def _load_local_ntv3_classes(base_model_dir: str) -> tuple[type, type, type]:
    """Load local NTv3 implementation classes from external/NTv3/ntv3_base_model.

    Kept separate and cacheable so tests can monkeypatch this function without
    loading the real external model code.
    """
    base_model_path = Path(base_model_dir).expanduser().resolve()
    _validate_local_base_model_dir(base_model_path)

    base_model_str = str(base_model_path)
    if base_model_str not in sys.path:
        sys.path.insert(0, base_model_str)

    importlib.invalidate_caches()

    config_module = importlib.import_module("configuration_ntv3_pretrained")
    model_module = importlib.import_module("modeling_ntv3_pretrained")
    tokenizer_module = importlib.import_module("tokenization_ntv3")

    return (
        config_module.Ntv3PreTrainedConfig,
        model_module.NTv3PreTrained,
        tokenizer_module.NTv3Tokenizer,
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


def _load_tokenizer_from_local_class(tokenizer_cls: type, model_dir: Path) -> Any:
    """Load the local NTv3 tokenizer without using AutoTokenizer remote-code resolution."""
    try:
        return tokenizer_cls.from_pretrained(str(model_dir), local_files_only=True)
    except Exception as from_pretrained_error:
        tokenizer_config = _read_json(model_dir / "tokenizer_config.json")
        vocab_file = model_dir / "vocab.json"

        allowed_keys = {
            "unk_token",
            "pad_token",
            "mask_token",
            "cls_token",
            "eos_token",
            "bos_token",
            "clean_up_tokenization_spaces",
            "model_max_length",
        }
        kwargs = {
            key: value
            for key, value in tokenizer_config.items()
            if key in allowed_keys
        }

        try:
            return tokenizer_cls(
                vocab_file=str(vocab_file) if vocab_file.exists() else None,
                **kwargs,
            )
        except Exception as fallback_error:
            raise RuntimeError(
                f"Failed to load local NTv3 tokenizer from {model_dir}"
            ) from fallback_error


def _move_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _align_attention_mask_to_hidden(
    attention_mask: torch.Tensor,
    target_length: int,
) -> torch.Tensor:
    """Resize attention mask if a non-final hidden layer has a different sequence length."""
    if attention_mask.shape[1] == target_length:
        return attention_mask

    mask = attention_mask.float().unsqueeze(1)

    if target_length < attention_mask.shape[1]:
        resized = F.adaptive_max_pool1d(mask, output_size=target_length)
    else:
        resized = F.interpolate(mask, size=target_length, mode="nearest")

    return resized.squeeze(1).to(dtype=attention_mask.dtype)


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


class NucleotideTransformerEncoder(nn.Module):
    """Frozen local NTv3 encoder that returns pooled sequence embeddings."""

    def __init__(
        self,
        config: NucleotideTransformerConfig | None = None,
        **overrides: Any,
    ) -> None:
        super().__init__()

        self.config = replace(config or NucleotideTransformerConfig(), **overrides)

        self.model_dir = _resolve_repo_path(self.config.model_dir)
        self.base_model_dir = _resolve_repo_path(self.config.base_model_dir)

        _validate_local_model_dir(
            self.model_dir,
            require_weights=self.config.require_weights,
        )
        _validate_local_base_model_dir(self.base_model_dir)

        config_cls, model_cls, tokenizer_cls = _load_local_ntv3_classes(str(self.base_model_dir))

        self.nt_config = config_cls.from_pretrained(
            str(self.model_dir),
            local_files_only=self.config.local_files_only,
        )
        self.tokenizer = _load_tokenizer_from_local_class(tokenizer_cls, self.model_dir)

        model_kwargs: dict[str, Any] = {
            "config": self.nt_config,
            "local_files_only": self.config.local_files_only,
        }

        resolved_dtype = _resolve_torch_dtype(self.config.torch_dtype)
        if resolved_dtype is not None:
            model_kwargs["torch_dtype"] = resolved_dtype

        self.model = model_cls.from_pretrained(str(self.model_dir), **model_kwargs)

        if self.config.freeze_encoder:
            self.freeze()

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def embedding_dim(self) -> int:
        # The final deconvolution hidden state has conv_init_embed_dim channels.
        for attr in ("conv_init_embed_dim", "embed_dim", "hidden_size", "d_model"):
            value = getattr(self.nt_config, attr, None)
            if value is not None:
                return int(value)

        input_embeddings = self.model.get_input_embeddings()
        return int(getattr(input_embeddings, "embedding_dim", input_embeddings.weight.shape[-1]))

    def train(self, mode: bool = True) -> "NucleotideTransformerEncoder":
        super().train(mode)

        # The encoder is frozen in the main host-aware predictor, so keep dropout
        # and any stochastic layers disabled even when the parent fusion model trains.
        if getattr(self, "config", None) is not None and self.config.freeze_encoder:
            if hasattr(self, "model"):
                self.model.eval()

        return self

    def freeze(self) -> "NucleotideTransformerEncoder":
        self.config = replace(self.config, freeze_encoder=True)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        return self

    def unfreeze(self) -> "NucleotideTransformerEncoder":
        self.config = replace(self.config, freeze_encoder=False)
        for parameter in self.model.parameters():
            parameter.requires_grad = True
        self.model.train(self.training)
        return self

    def _effective_pad_to_multiple_of(self) -> int | None:
        if self.config.pad_to_multiple_of is not None:
            return int(self.config.pad_to_multiple_of)

        num_downsamples = getattr(self.nt_config, "num_downsamples", None)
        if num_downsamples is None:
            return None

        return 2 ** int(num_downsamples)

    def _effective_max_length(self) -> int | None:
        if self.config.max_length is None:
            return None

        max_length = int(self.config.max_length)
        pad_to_multiple_of = self._effective_pad_to_multiple_of()

        if pad_to_multiple_of is not None and max_length % pad_to_multiple_of != 0:
            max_length = int(math.ceil(max_length / pad_to_multiple_of) * pad_to_multiple_of)

        return max_length

    def _normalise_sequence(self, sequence: str) -> str:
        sequence = "".join(str(sequence).upper().replace("U", "T").split())

        if not sequence:
            raise ValueError("Encountered an empty nucleotide sequence after normalization.")

        valid_bases = {"A", "C", "G", "T", "N"}
        invalid_bases = set(sequence) - valid_bases

        if invalid_bases and not self.config.replace_ambiguous_bases_with_n:
            raise ValueError(f"Invalid nucleotide characters: {sorted(invalid_bases)}")

        if invalid_bases:
            sequence = "".join(base if base in valid_bases else "N" for base in sequence)

        return sequence

    def tokenize(self, sequences: str | Sequence[str]) -> dict[str, torch.Tensor]:
        if isinstance(sequences, str):
            sequence_list = [sequences]
        else:
            sequence_list = list(sequences)

        if not sequence_list:
            raise ValueError("No sequences were provided.")

        cleaned_sequences = [self._normalise_sequence(sequence) for sequence in sequence_list]

        tokenizer_kwargs: dict[str, Any] = {
            "add_special_tokens": False,
            "padding": True,
            "return_tensors": "pt",
            "return_attention_mask": True,
        }

        max_length = self._effective_max_length()
        if max_length is not None:
            tokenizer_kwargs["truncation"] = True
            tokenizer_kwargs["max_length"] = max_length

        pad_to_multiple_of = self._effective_pad_to_multiple_of()
        if pad_to_multiple_of is not None:
            tokenizer_kwargs["pad_to_multiple_of"] = pad_to_multiple_of

        batch = dict(self.tokenizer(cleaned_sequences, **tokenizer_kwargs))

        input_ids = batch["input_ids"].long()

        if "attention_mask" not in batch:
            pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
            if pad_token_id is None:
                pad_token_id = getattr(self.nt_config, "pad_token_id", 1)
            batch["attention_mask"] = input_ids.ne(int(pad_token_id)).long()
        else:
            batch["attention_mask"] = batch["attention_mask"].long()

        batch["input_ids"] = input_ids

        return _move_to_device(batch, self.device)

    def forward(
        self,
        sequences: str | Sequence[str] | None = None,
        *,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        layer: int | None = None,
        pooling: PoolMode | None = None,
        return_hidden_states: bool = False,
        **model_kwargs: Any,
    ) -> NucleotideTransformerOutput:
        if sequences is not None:
            if input_ids is not None:
                raise ValueError("Pass either raw sequences or input_ids, not both.")

            batch = self.tokenize(sequences)
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]

        elif input_ids is None:
            raise ValueError("Pass raw sequences or tokenized input_ids.")

        input_ids = input_ids.to(self.device).long()

        if attention_mask is None:
            pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
            if pad_token_id is None:
                pad_token_id = getattr(self.nt_config, "pad_token_id", 1)
            attention_mask = input_ids.ne(int(pad_token_id)).long()

        attention_mask = attention_mask.to(self.device).long()
        model_kwargs = _move_to_device(dict(model_kwargs), self.device)

        grad_context = torch.no_grad() if self.config.freeze_encoder else nullcontext()

        with grad_context:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                **model_kwargs,
            )

        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            raise RuntimeError(
                "Local NTv3 model did not return hidden_states. "
                "The encoder requires output_hidden_states=True."
            )

        hidden_states_tuple = tuple(hidden_states)
        layer_index = self.config.layer if layer is None else int(layer)
        token_embeddings = hidden_states_tuple[layer_index]

        pooling_mode = self.config.pooling if pooling is None else pooling
        pooling_mask = _align_attention_mask_to_hidden(
            attention_mask,
            target_length=token_embeddings.shape[1],
        )
        pooled_embedding = _pool_hidden_states(
            token_embeddings,
            attention_mask=pooling_mask,
            pooling=pooling_mode,
        )

        return NucleotideTransformerOutput(
            token_embeddings=token_embeddings,
            pooled_embedding=pooled_embedding,
            attention_mask=pooling_mask,
            input_ids=input_ids,
            logits=getattr(outputs, "logits", None),
            hidden_states=hidden_states_tuple if return_hidden_states else None,
        )

    @torch.no_grad()
    def embed(
        self,
        sequences: str | Sequence[str],
        *,
        batch_size: int = 4,
        layer: int | None = None,
        pooling: PoolMode | None = None,
    ) -> torch.Tensor:
        """Return CPU pooled embeddings for raw DNA sequences."""
        if isinstance(sequences, str):
            sequence_list = [sequences]
        else:
            sequence_list = list(sequences)

        if not sequence_list:
            raise ValueError("No sequences were provided.")

        was_training = self.training
        self.eval()

        embeddings: list[torch.Tensor] = []

        for start in range(0, len(sequence_list), batch_size):
            output = self.forward(
                sequence_list[start : start + batch_size],
                layer=layer,
                pooling=pooling,
            )

            if output.pooled_embedding is None:
                raise ValueError("embed() requires pooling='mean', 'cls', or 'first'.")

            embeddings.append(output.pooled_embedding.detach().cpu())

        self.train(was_training)
        return torch.cat(embeddings, dim=0)


NucleotideTransformerWrapper = NucleotideTransformerEncoder

__all__ = [
    "NucleotideTransformerConfig",
    "NucleotideTransformerEncoder",
    "NucleotideTransformerWrapper",
    "NucleotideTransformerOutput",
]