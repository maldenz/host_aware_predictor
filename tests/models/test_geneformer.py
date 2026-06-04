from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from host_aware_predictor.models import geneformer as gf


class DummyGeneformerHFConfig:
    hidden_size = 4
    max_position_embeddings = 6
    pad_token_id = 0
    vocab_size = 32

    last_from_pretrained_call = None

    @classmethod
    def from_pretrained(cls, model_dir: str, **kwargs):
        cls.last_from_pretrained_call = (Path(model_dir), kwargs)
        return cls()


class DummyAutoModel:
    last_from_pretrained_call = None

    @classmethod
    def from_pretrained(cls, model_dir: str, **kwargs):
        cls.last_from_pretrained_call = (Path(model_dir), kwargs)
        return DummyGeneformerModel(config=kwargs["config"], with_logits=False)


class DummyAutoModelForMaskedLM:
    last_from_pretrained_call = None

    @classmethod
    def from_pretrained(cls, model_dir: str, **kwargs):
        cls.last_from_pretrained_call = (Path(model_dir), kwargs)
        return DummyGeneformerModel(config=kwargs["config"], with_logits=True)


class DummyGeneformerModel(nn.Module):
    def __init__(self, config, *, with_logits: bool) -> None:
        super().__init__()
        self.config = config
        self.with_logits = with_logits
        self.weight = nn.Parameter(torch.ones(1))

    def get_input_embeddings(self):
        return nn.Embedding(32, 4)

    def forward(
        self,
        *,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        output_hidden_states=False,
        return_dict=True,
        **kwargs,
    ):
        del attention_mask, token_type_ids, output_hidden_states, return_dict, kwargs

        base_hidden = torch.stack(
            [
                input_ids.float(),
                input_ids.float() + 1.0,
                input_ids.float() + 2.0,
                input_ids.float() + 3.0,
            ],
            dim=-1,
        )
        final_hidden = base_hidden + 10.0

        logits = None
        if self.with_logits:
            logits = torch.zeros(
                input_ids.shape[0],
                input_ids.shape[1],
                32,
                device=input_ids.device,
                dtype=final_hidden.dtype,
            )

        return SimpleNamespace(
            last_hidden_state=final_hidden,
            hidden_states=(base_hidden, final_hidden),
            logits=logits,
        )


def _write_minimal_local_model_dir(tmp_path: Path, *, with_weights: bool = True) -> Path:
    model_dir = tmp_path / "external" / "Geneformer" / "Geneformer-V2-316M"
    model_dir.mkdir(parents=True)

    (model_dir / "config.json").write_text(
        """
        {
          "architectures": ["BertForMaskedLM"],
          "hidden_size": 4,
          "max_position_embeddings": 6,
          "model_type": "bert",
          "pad_token_id": 0,
          "vocab_size": 32
        }
        """,
        encoding="utf-8",
    )

    if with_weights:
        # The wrapper validates that a local checkpoint exists. The mocked model
        # class does not read this file.
        (model_dir / "model.safetensors").write_bytes(b"dummy")

    return model_dir


def _patch_geneformer_hf_loaders(monkeypatch):
    monkeypatch.setattr(gf, "AutoConfig", DummyGeneformerHFConfig)
    monkeypatch.setattr(gf, "AutoModel", DummyAutoModel)
    monkeypatch.setattr(gf, "AutoModelForMaskedLM", DummyAutoModelForMaskedLM)


def test_encoder_loads_only_local_base_model_by_default(monkeypatch, tmp_path):
    _patch_geneformer_hf_loaders(monkeypatch)

    model_dir = _write_minimal_local_model_dir(tmp_path)

    encoder = gf.GeneformerEncoder(model_dir=model_dir)

    assert DummyGeneformerHFConfig.last_from_pretrained_call[0] == model_dir
    assert DummyGeneformerHFConfig.last_from_pretrained_call[1]["local_files_only"] is True
    assert DummyGeneformerHFConfig.last_from_pretrained_call[1]["trust_remote_code"] is False

    assert DummyAutoModel.last_from_pretrained_call[0] == model_dir
    assert DummyAutoModel.last_from_pretrained_call[1]["local_files_only"] is True
    assert DummyAutoModel.last_from_pretrained_call[1]["trust_remote_code"] is False

    assert DummyAutoModelForMaskedLM.last_from_pretrained_call is None

    assert encoder.model_dir == model_dir.resolve()
    assert encoder.pad_token_id == 0
    assert encoder.embedding_dim == 4


def test_encoder_can_load_mlm_head_when_requested(monkeypatch, tmp_path):
    _patch_geneformer_hf_loaders(monkeypatch)

    model_dir = _write_minimal_local_model_dir(tmp_path)

    encoder = gf.GeneformerEncoder(
        model_dir=model_dir,
        load_mlm_head=True,
    )

    output = encoder([[2, 4, 0]])

    assert DummyAutoModelForMaskedLM.last_from_pretrained_call[0] == model_dir
    assert output.logits is not None
    assert output.logits.shape == (1, 3, 32)


def test_collates_variable_length_tokenized_cells_and_mean_pools(monkeypatch, tmp_path):
    _patch_geneformer_hf_loaders(monkeypatch)

    model_dir = _write_minimal_local_model_dir(tmp_path)
    encoder = gf.GeneformerEncoder(model_dir=model_dir)

    output = encoder([[2, 4], [6]])

    assert output.input_ids.tolist() == [[2, 4], [6, 0]]
    assert output.attention_mask.tolist() == [[1, 1], [1, 0]]
    assert output.token_embeddings.shape == (2, 2, 4)
    assert output.pooled_embedding.shape == (2, 4)
    assert output.logits is None

    # Final hidden state is [id + 10, id + 11, id + 12, id + 13].
    # First row mean id = 3.
    torch.testing.assert_close(
        output.pooled_embedding[0],
        torch.tensor([13.0, 14.0, 15.0, 16.0]),
    )

    # Second row mean excludes pad token id 0, so mean id = 6.
    torch.testing.assert_close(
        output.pooled_embedding[1],
        torch.tensor([16.0, 17.0, 18.0, 19.0]),
    )


def test_tensor_input_truncates_to_max_length(monkeypatch, tmp_path):
    _patch_geneformer_hf_loaders(monkeypatch)

    model_dir = _write_minimal_local_model_dir(tmp_path)
    encoder = gf.GeneformerEncoder(
        model_dir=model_dir,
        max_length=3,
    )

    output = encoder(torch.tensor([[1, 2, 3, 4, 5]]))

    assert output.input_ids.tolist() == [[1, 2, 3]]
    assert output.attention_mask.tolist() == [[1, 1, 1]]
    assert output.token_embeddings.shape == (1, 3, 4)


def test_mapping_input_respects_attention_mask_and_token_type_ids(monkeypatch, tmp_path):
    _patch_geneformer_hf_loaders(monkeypatch)

    model_dir = _write_minimal_local_model_dir(tmp_path)
    encoder = gf.GeneformerEncoder(model_dir=model_dir)

    output = encoder(
        {
            "input_ids": torch.tensor([[2, 4, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 0, 0]]),
            "token_type_ids": torch.tensor([[0, 0, 0, 0]]),
        }
    )

    assert output.input_ids.tolist() == [[2, 4, 0, 0]]
    assert output.attention_mask.tolist() == [[1, 1, 0, 0]]
    assert output.token_type_ids is not None
    assert output.token_type_ids.tolist() == [[0, 0, 0, 0]]

    torch.testing.assert_close(
        output.pooled_embedding[0],
        torch.tensor([13.0, 14.0, 15.0, 16.0]),
    )


def test_frozen_encoder_keeps_model_eval_even_when_parent_trains(monkeypatch, tmp_path):
    _patch_geneformer_hf_loaders(monkeypatch)

    model_dir = _write_minimal_local_model_dir(tmp_path)
    encoder = gf.GeneformerEncoder(
        model_dir=model_dir,
        freeze_encoder=True,
    )

    assert all(not parameter.requires_grad for parameter in encoder.model.parameters())
    assert encoder.model.training is False

    encoder.train()

    assert encoder.training is True
    assert encoder.model.training is False


def test_missing_local_safetensors_raises_before_loading_model(monkeypatch, tmp_path):
    _patch_geneformer_hf_loaders(monkeypatch)

    model_dir = _write_minimal_local_model_dir(tmp_path, with_weights=False)

    with pytest.raises(FileNotFoundError, match="model.safetensors"):
        gf.GeneformerEncoder(
            model_dir=model_dir,
            require_weights=True,
        )


def test_empty_tokenized_cells_raise(monkeypatch, tmp_path):
    _patch_geneformer_hf_loaders(monkeypatch)

    model_dir = _write_minimal_local_model_dir(tmp_path)
    encoder = gf.GeneformerEncoder(model_dir=model_dir)

    with pytest.raises(ValueError, match="tokenized_cells is empty"):
        encoder([])


def test_empty_cell_token_sequence_raises(monkeypatch, tmp_path):
    _patch_geneformer_hf_loaders(monkeypatch)

    model_dir = _write_minimal_local_model_dir(tmp_path)
    encoder = gf.GeneformerEncoder(model_dir=model_dir)

    with pytest.raises(ValueError, match="empty Geneformer token-id sequence"):
        encoder([[]])


def test_embed_batches_and_returns_cpu_tensor(monkeypatch, tmp_path):
    _patch_geneformer_hf_loaders(monkeypatch)

    model_dir = _write_minimal_local_model_dir(tmp_path)
    encoder = gf.GeneformerEncoder(model_dir=model_dir)

    embeddings = encoder.embed(
        [[2, 4], [6], [8, 10]],
        batch_size=2,
    )

    assert embeddings.device.type == "cpu"
    assert embeddings.shape == (3, 4)


def test_optional_real_local_geneformer_checkpoint_smoke():
    """Opt-in integration smoke test for the real uncommitted local checkpoint.

    CPU default:

        RUN_LOCAL_GENEFORMER_TESTS=1 pytest tests/models/test_geneformer.py -k real_local

    Explicit GPU:

        GENEFORMER_TEST_DEVICE=cuda RUN_LOCAL_GENEFORMER_TESTS=1 pytest tests/models/test_geneformer.py -k real_local
    """
    if os.environ.get("RUN_LOCAL_GENEFORMER_TESTS") != "1":
        pytest.skip("Set RUN_LOCAL_GENEFORMER_TESTS=1 to load the real local Geneformer checkpoint.")

    model_dir = Path("external/Geneformer/Geneformer-V2-316M")
    checkpoint_path = model_dir / "model.safetensors"

    if not checkpoint_path.exists():
        pytest.skip(f"Missing local checkpoint: {checkpoint_path}")

    # Default to CPU. Shared A100 boxes often report torch.cuda.is_available()
    # even when the current user/process cannot allocate the device.
    requested_device = os.environ.get("GENEFORMER_TEST_DEVICE", "cpu")
    device = torch.device(requested_device)

    if device.type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA was requested but torch.cuda.is_available() is False.")

    try:
        encoder = gf.GeneformerEncoder(
            model_dir=model_dir,
            max_length=8,
            torch_dtype="float32",
            load_mlm_head=False,
        ).to(device)
    except Exception as exc:
        message = str(exc).lower()
        if device.type == "cuda" and ("cuda" in message or "accelerator" in message):
            pytest.skip(f"CUDA device requested but unavailable: {exc}")
        raise

    input_ids = torch.tensor(
        [
            [1, 2, 3, 4, 5, 0, 0, 0],
            [6, 7, 8, 9, 0, 0, 0, 0],
        ],
        dtype=torch.long,
        device=device,
    )

    output = encoder(input_ids)

    assert output.pooled_embedding is not None
    assert output.pooled_embedding.shape == (2, encoder.embedding_dim)
    assert output.token_embeddings.shape == (2, 8, encoder.embedding_dim)
    assert output.logits is None