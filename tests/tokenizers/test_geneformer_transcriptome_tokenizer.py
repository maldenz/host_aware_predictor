from pathlib import Path

import pytest

from host_aware_predictor.tokenizers import geneformer_transcriptome as module


def test_infer_geneformer_file_format_from_suffix():
    assert module.infer_geneformer_file_format(Path("sample.h5ad")) == "h5ad"
    assert module.infer_geneformer_file_format(Path("sample.loom")) == "loom"


class DummyTranscriptomeTokenizer:
    init_kwargs = None
    calls = []

    def __init__(self, **kwargs):
        DummyTranscriptomeTokenizer.init_kwargs = kwargs

    def tokenize_data(self, **kwargs):
        DummyTranscriptomeTokenizer.calls.append(kwargs)

        output_path = Path(kwargs["output_directory"]) / f"{kwargs['output_prefix']}.dataset"
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "dummy.txt").write_text("ok")


def test_tokenize_calls_geneformer_transcriptome_tokenizer(tmp_path, monkeypatch):
    DummyTranscriptomeTokenizer.init_kwargs = None
    DummyTranscriptomeTokenizer.calls = []

    monkeypatch.setattr(
        module,
        "_load_geneformer_transcriptome_tokenizer",
        lambda geneformer_repo: DummyTranscriptomeTokenizer,
    )

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "sample.loom").write_text("not a real loom; Geneformer is mocked")

    tokenizer = module.GeneformerTranscriptomeTokenizer(
        module.GeneformerTranscriptomeTokenizerConfig(
            geneformer_repo=None,
            nproc=2,
            custom_attr_name_dict={"cell_type": "cell_type"},
        )
    )

    dataset_path = tokenizer.tokenize(
        raw_dir,
        output_dir=tmp_path / "out",
        output_prefix="toy",
        file_format="loom",
    )

    assert dataset_path == tmp_path / "out" / "toy.dataset"
    assert (dataset_path / "dummy.txt").read_text() == "ok"

    assert DummyTranscriptomeTokenizer.init_kwargs["nproc"] == 2
    assert DummyTranscriptomeTokenizer.init_kwargs["custom_attr_name_dict"] == {
        "cell_type": "cell_type"
    }

    assert DummyTranscriptomeTokenizer.calls[0]["data_directory"] == str(raw_dir)
    assert DummyTranscriptomeTokenizer.calls[0]["file_format"] == "loom"


def test_prepare_h5ad_for_geneformer_creates_required_columns(tmp_path):
    ad = pytest.importorskip("anndata")
    pytest.importorskip("scipy")

    import numpy as np
    from scipy import sparse

    adata = ad.AnnData(
        X=sparse.csr_matrix(
            np.array(
                [
                    [1, 0, 3],
                    [0, 2, 0],
                ],
                dtype=np.float32,
            )
        )
    )
    adata.obs_names = ["cell_1", "cell_2"]
    adata.var_names = ["ENSG000001.1", "ENSG000002.5", "ENSG000003.7"]

    input_path = tmp_path / "raw.h5ad"
    output_path = tmp_path / "prepared.h5ad"

    adata.write_h5ad(input_path)

    module.prepare_h5ad_for_geneformer(input_path, output_path)
    prepared = ad.read_h5ad(output_path)

    assert prepared.var["ensembl_id"].tolist() == [
        "ENSG000001",
        "ENSG000002",
        "ENSG000003",
    ]
    assert prepared.obs["n_counts"].tolist() == [4.0, 2.0]