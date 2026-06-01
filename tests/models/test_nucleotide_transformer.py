from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from host_aware_predictor.models import nucleotide_transformer as nt


class DummyNTv3Config:
    conv_init_embed_dim = 4
    embed_dim = 4
    num_downsamples = 3
    pad_token_id = 1

    last_from_pretrained_call = None

    @classmethod
    def from_pretrained(cls, model_dir: str, **kwargs):
        cls.last_from_pretrained_call = (Path(model_dir), kwargs)
        return cls()


class DummyNTv3Tokenizer:
    pad_token_id = 1
    last_from_pretrained_call = None
    last_sequences = None
    last_tokenizer_kwargs = None

    vocab = {
        "<unk>": 0,
        "<pad>": 1,
        "<mask>": 2,
        "<cls>": 3,
        "<eos>": 4,
        "<bos>": 5,
        "A": 6,
        "T": 7,
        "C": 8,
        "G": 9,
        "N": 10,
    }

    @classmethod
    def from_pretrained(cls, model_dir: str, **kwargs):
        cls.last_from_pretrained_call = (Path(model_dir), kwargs)
        return cls()

    def __call__(self, sequences, **kwargs):
        self.__class__.last_sequences = list(sequences)
        self.__class__.last_tokenizer_kwargs = dict(kwargs)

        input_rows = []
        for sequence in sequences:
            row = [self.vocab.get(base, self.vocab["<unk>"]) for base in sequence]

            if kwargs.get("truncation") and kwargs.get("max_length") is not None:
                row = row[: int(kwargs["max_length"])]

            input_rows.append(row)

        max_length = max(len(row) for row in input_rows)
        pad_to_multiple_of = kwargs.get("pad_to_multiple_of")
        if pad_to_multiple_of is not None and max_length % int(pad_to_multiple_of) != 0:
            max_length = ((max_length + int(pad_to_multiple_of) - 1) // int(pad_to_multiple_of)) * int(
                pad_to_multiple_of
            )

        padded = []
        masks = []
        for row in input_rows:
            pad_count = max_length - len(row)
            padded.append(row + [self.pad_token_id] * pad_count)
            masks.append([1] * len(row) + [0] * pad_count)

        return {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }


class DummyNTv3Model(nn.Module):
    last_from_pretrained_call = None

    @classmethod
    def from_pretrained(cls, model_dir: str, **kwargs):
        cls.last_from_pretrained_call = (Path(model_dir), kwargs)
        return cls(config=kwargs["config"])

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.weight = nn.Parameter(torch.ones(1))

    def forward(
        self,
        *,
        input_ids,
        attention_mask=None,
        output_hidden_states=False,
        return_dict=True,
        **kwargs,
    ):
        del attention_mask, output_hidden_states, return_dict, kwargs

        # Shape: [batch, length, 4]
        token_embeddings = torch.stack(
            [
                input_ids.float(),
                input_ids.float() + 1.0,
                input_ids.float() + 2.0,
                input_ids.float() + 3.0,
            ],
            dim=-1,
        )
        logits = torch.zeros(
            input_ids.shape[0],
            input_ids.shape[1],
            11,
            device=input_ids.device,
            dtype=token_embeddings.dtype,
        )

        return SimpleNamespace(
            logits=logits,
            hidden_states=(token_embeddings,),
        )


def _write_minimal_local_model_dir(tmp_path: Path, *, with_weights: bool = True) -> Path:
    model_dir = tmp_path / "external" / "NTv3" / "NTv3_100M_pre"
    model_dir.mkdir(parents=True)

    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "tokenizer_config.json").write_text(
        """
        {
          "unk_token": "<unk>",
          "pad_token": "<pad>",
          "mask_token": "<mask>",
          "cls_token": "<cls>",
          "eos_token": "<eos>",
          "bos_token": "<bos>"
        }
        """,
        encoding="utf-8",
    )

    if with_weights:
        # The wrapper only validates that a local weight file exists. The mocked
        # model class does not read this file.
        (model_dir / "model.safetensors").write_bytes(b"dummy")

    return model_dir


def _write_minimal_base_model_dir(tmp_path: Path) -> Path:
    base_model_dir = tmp_path / "external" / "NTv3" / "ntv3_base_model"
    base_model_dir.mkdir(parents=True)

    for filename in (
        "configuration_ntv3_pretrained.py",
        "modeling_ntv3_pretrained.py",
        "tokenization_ntv3.py",
    ):
        (base_model_dir / filename).write_text("# mocked in tests\n", encoding="utf-8")

    return base_model_dir


def _patch_ntv3_loader(monkeypatch):
    monkeypatch.setattr(
        nt,
        "_load_local_ntv3_classes",
        lambda base_model_dir: (
            DummyNTv3Config,
            DummyNTv3Model,
            DummyNTv3Tokenizer,
        ),
    )


def test_encoder_loads_only_local_model_classes(monkeypatch, tmp_path):
    _patch_ntv3_loader(monkeypatch)

    model_dir = _write_minimal_local_model_dir(tmp_path)
    base_model_dir = _write_minimal_base_model_dir(tmp_path)

    encoder = nt.NucleotideTransformerEncoder(
        model_dir=model_dir,
        base_model_dir=base_model_dir,
        pad_to_multiple_of=8,
    )

    assert DummyNTv3Config.last_from_pretrained_call[0] == model_dir # type: ignore
    assert DummyNTv3Config.last_from_pretrained_call[1]["local_files_only"] is True # type: ignore

    assert DummyNTv3Tokenizer.last_from_pretrained_call[0] == model_dir # type: ignore
    assert DummyNTv3Tokenizer.last_from_pretrained_call[1]["local_files_only"] is True # type: ignore

    assert DummyNTv3Model.last_from_pretrained_call[0] == model_dir # type: ignore
    assert DummyNTv3Model.last_from_pretrained_call[1]["local_files_only"] is True # type: ignore

    assert encoder.model_dir == model_dir.resolve()
    assert encoder.base_model_dir == base_model_dir.resolve()
    assert encoder.embedding_dim == 4


def test_forward_normalizes_sequences_pads_and_mean_pools(monkeypatch, tmp_path):
    _patch_ntv3_loader(monkeypatch)

    model_dir = _write_minimal_local_model_dir(tmp_path)
    base_model_dir = _write_minimal_base_model_dir(tmp_path)

    encoder = nt.NucleotideTransformerEncoder(
        model_dir=model_dir,
        base_model_dir=base_model_dir,
        pad_to_multiple_of=8,
    )

    output = encoder(["acgu z", "TT"])

    # U -> T, whitespace removed, invalid Z -> N.
    assert DummyNTv3Tokenizer.last_sequences == ["ACGTN", "TT"]

    assert output.input_ids.shape == (2, 8)
    assert output.attention_mask.shape == (2, 8)
    assert output.token_embeddings.shape == (2, 8, 4)
    assert output.pooled_embedding.shape == (2, 4)
    assert output.logits.shape == (2, 8, 11)

    # First sequence ids: A=6, C=8, G=9, T=7, N=10. Mean id = 8.
    torch.testing.assert_close(
        output.pooled_embedding[0],
        torch.tensor([8.0, 9.0, 10.0, 11.0]),
    )

    # Second sequence ids: T=7, T=7. Mean id = 7.
    torch.testing.assert_close(
        output.pooled_embedding[1],
        torch.tensor([7.0, 8.0, 9.0, 10.0]),
    )


def test_frozen_encoder_keeps_model_eval_even_when_parent_trains(monkeypatch, tmp_path):
    _patch_ntv3_loader(monkeypatch)

    model_dir = _write_minimal_local_model_dir(tmp_path)
    base_model_dir = _write_minimal_base_model_dir(tmp_path)

    encoder = nt.NucleotideTransformerEncoder(
        model_dir=model_dir,
        base_model_dir=base_model_dir,
        freeze_encoder=True,
    )

    assert all(not parameter.requires_grad for parameter in encoder.model.parameters())
    assert encoder.model.training is False

    encoder.train()

    assert encoder.training is True
    assert encoder.model.training is False


def test_missing_local_safetensors_raises_before_loading_model(monkeypatch, tmp_path):
    _patch_ntv3_loader(monkeypatch)

    model_dir = _write_minimal_local_model_dir(tmp_path, with_weights=False)
    base_model_dir = _write_minimal_base_model_dir(tmp_path)

    with pytest.raises(FileNotFoundError, match="model.safetensors"):
        nt.NucleotideTransformerEncoder(
            model_dir=model_dir,
            base_model_dir=base_model_dir,
            require_weights=True,
        )


def test_empty_sequence_after_normalization_raises(monkeypatch, tmp_path):
    _patch_ntv3_loader(monkeypatch)

    model_dir = _write_minimal_local_model_dir(tmp_path)
    base_model_dir = _write_minimal_base_model_dir(tmp_path)

    encoder = nt.NucleotideTransformerEncoder(
        model_dir=model_dir,
        base_model_dir=base_model_dir,
    )

    with pytest.raises(ValueError, match="empty nucleotide sequence"):
        encoder(["   \n\t   "])


def test_optional_real_local_ntv3_checkpoint_smoke():
    """Opt-in integration smoke test for the real uncommitted local checkpoint.

    Run manually with:

        RUN_LOCAL_NTV3_TESTS=1 pytest tests/models/test_nucleotide_transformer.py -k real_local
    """
    if os.environ.get("RUN_LOCAL_NTV3_TESTS") != "1":
        pytest.skip("Set RUN_LOCAL_NTV3_TESTS=1 to load the real local NTv3 checkpoint.")

    model_dir = Path("external/NTv3/NTv3_100M_pre")
    base_model_dir = Path("external/NTv3/ntv3_base_model")

    if not (model_dir / "model.safetensors").exists():
        pytest.skip(f"Missing local checkpoint: {model_dir / 'model.safetensors'}")

    encoder = nt.NucleotideTransformerEncoder(
        model_dir=model_dir,
        base_model_dir=base_model_dir,
        max_length=128,
        pad_to_multiple_of=128,
        torch_dtype="float32",
    )

    output = encoder(["ACGT" * 32])

    assert output.pooled_embedding is not None
    assert output.pooled_embedding.shape == (1, encoder.embedding_dim)
    assert output.token_embeddings.shape[0] == 1