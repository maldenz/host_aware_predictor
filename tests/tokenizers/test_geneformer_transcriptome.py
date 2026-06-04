from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from host_aware_predictor.tokenizers.geneformer_transcriptome import (
    GeneformerTranscriptomeTokenizer,
    TokenizedTranscriptomes,
)


CLS = 1001
EOS = 1002

TOKEN_A = 101
TOKEN_B = 102
TOKEN_C = 103
TOKEN_COLLAPSED = 104
TOKEN_D = 105


@pytest.fixture()
def fake_geneformer_repo(tmp_path: Path) -> Path:
    """Create a tiny local external/Geneformer layout.

    This avoids depending on the real external Geneformer pickle files while
    preserving the same file names expected by the tokenizer.
    """
    package_dir = tmp_path / "external" / "Geneformer" / "geneformer"
    package_dir.mkdir(parents=True)

    gene_median_dict = {
        "ENSG_A": 10.0,
        "ENSG_B": 1.0,
        "ENSG_C": 5.0,
        "ENSG_COLLAPSED": 2.0,
        "ENSG_D": 1.0,
    }

    gene_token_dict = {
        "<cls>": CLS,
        "<eos>": EOS,
        "ENSG_A": TOKEN_A,
        "ENSG_B": TOKEN_B,
        "ENSG_C": TOKEN_C,
        "ENSG_COLLAPSED": TOKEN_COLLAPSED,
        "ENSG_D": TOKEN_D,
    }

    gene_mapping_dict = {
        "ENSG000001": "ENSG_A",
        "ENSG000002": "ENSG_B",
        "ENSG000003": "ENSG_C",
        "ENSG_DUP1": "ENSG_COLLAPSED",
        "ENSG_DUP2": "ENSG_COLLAPSED",
        "ENSG000004": "ENSG_D",
    }

    with (package_dir / "gene_median_dictionary_gc104M.pkl").open("wb") as handle:
        pickle.dump(gene_median_dict, handle)

    with (package_dir / "token_dictionary_gc104M.pkl").open("wb") as handle:
        pickle.dump(gene_token_dict, handle)

    with (package_dir / "ensembl_mapping_dict_gc104M.pkl").open("wb") as handle:
        pickle.dump(gene_mapping_dict, handle)

    return tmp_path / "external" / "Geneformer"


def test_tokenize_counts_ranks_by_median_scaled_expression(fake_geneformer_repo: Path):
    tokenizer = GeneformerTranscriptomeTokenizer(
        geneformer_repo=fake_geneformer_repo,
        model_input_size=5,
    )

    # Scores are proportional to raw_count / gene_median because target_sum
    # and n_counts are constant within a cell.
    #
    # ENSG_A: 10 / 10 = 1
    # ENSG_B:  5 /  1 = 5
    # ENSG_C: 20 /  5 = 4
    #
    # Expected rank: B, C, A.
    counts = np.array([[10.0, 5.0, 20.0]])
    gene_ids = ["ENSG000001.7", "ENSG000002", "ENSG000003"]

    tokenized = tokenizer.tokenize_counts(
        counts,
        gene_ids=gene_ids,
        cell_ids=["host_0"],
    )

    assert tokenized.cell_ids == ["host_0"]
    assert tokenized.input_ids == [[CLS, TOKEN_B, TOKEN_C, TOKEN_A, EOS]]
    assert tokenized.lengths == [5]


def test_model_input_size_crops_ranked_genes_leaving_cls_and_eos(fake_geneformer_repo: Path):
    tokenizer = GeneformerTranscriptomeTokenizer(
        geneformer_repo=fake_geneformer_repo,
        model_input_size=4,
    )

    counts = np.array([[10.0, 5.0, 20.0]])
    gene_ids = ["ENSG000001", "ENSG000002", "ENSG000003"]

    tokenized = tokenizer.tokenize_counts(
        counts,
        gene_ids=gene_ids,
        cell_ids=["host_0"],
    )

    # V2 special tokens leave model_input_size - 2 ranked gene slots.
    assert tokenized.input_ids == [[CLS, TOKEN_B, TOKEN_C, EOS]]
    assert tokenized.lengths == [4]


def test_duplicate_ensembl_ids_are_collapsed_before_ranking(fake_geneformer_repo: Path):
    tokenizer = GeneformerTranscriptomeTokenizer(
        geneformer_repo=fake_geneformer_repo,
        model_input_size=6,
    )

    # ENSG_DUP1 and ENSG_DUP2 both map to ENSG_COLLAPSED.
    #
    # collapsed score: (1 + 3) / 2 = 2
    # ENSG_D score: 5 / 1 = 5
    #
    # Expected rank: D, collapsed.
    counts = np.array([[1.0, 3.0, 5.0]])
    gene_ids = ["ENSG_DUP1", "ENSG_DUP2", "ENSG000004"]

    tokenized = tokenizer.tokenize_counts(
        counts,
        gene_ids=gene_ids,
        cell_ids=["host_0"],
    )

    assert tokenized.input_ids == [[CLS, TOKEN_D, TOKEN_COLLAPSED, EOS]]
    assert tokenized.input_ids[0].count(TOKEN_COLLAPSED) == 1


def test_to_model_inputs_pads_and_masks_pad_token():
    tokenized = TokenizedTranscriptomes(
        input_ids=[
            [CLS, TOKEN_A, EOS],
            [CLS, TOKEN_B, TOKEN_C, EOS],
        ],
        cell_ids=["host_a", "host_b"],
        lengths=[3, 4],
        metadata={},
    )

    model_inputs = tokenized.to_model_inputs(pad_token_id=0)

    assert model_inputs["input_ids"].dtype == torch.long
    assert model_inputs["attention_mask"].dtype == torch.long

    assert model_inputs["input_ids"].tolist() == [
        [CLS, TOKEN_A, EOS, 0],
        [CLS, TOKEN_B, TOKEN_C, EOS],
    ]
    assert model_inputs["attention_mask"].tolist() == [
        [1, 1, 1, 0],
        [1, 1, 1, 1],
    ]


def test_to_model_inputs_respects_explicit_pad_to_length():
    tokenized = TokenizedTranscriptomes(
        input_ids=[
            [CLS, TOKEN_A, EOS],
            [CLS, TOKEN_B, EOS],
        ],
        cell_ids=["host_a", "host_b"],
        lengths=[3, 3],
        metadata={},
    )

    model_inputs = tokenized.to_model_inputs(
        pad_token_id=0,
        pad_to_length=6,
    )

    assert model_inputs["input_ids"].shape == (2, 6)
    assert model_inputs["attention_mask"].tolist() == [
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
    ]


def test_filter_pass_and_metadata_are_preserved(fake_geneformer_repo: Path):
    tokenizer = GeneformerTranscriptomeTokenizer(
        geneformer_repo=fake_geneformer_repo,
        model_input_size=6,
    )

    counts = np.array(
        [
            [1.0, 0.0],
            [0.0, 3.0],
            [2.0, 4.0],
        ]
    )
    gene_ids = ["ENSG000001", "ENSG000002"]

    tokenized = tokenizer.tokenize_counts(
        counts,
        gene_ids=gene_ids,
        cell_ids=["cell_a", "cell_b", "cell_c"],
        filter_pass=[1, 0, 1],
        metadata={
            "cell_type": ["type_a", "type_b", "type_c"],
            "donor": ["donor_1", "donor_2", "donor_3"],
        },
    )

    assert tokenized.cell_ids == ["cell_a", "cell_c"]
    assert tokenized.metadata == {
        "cell_type": ["type_a", "type_c"],
        "donor": ["donor_1", "donor_3"],
    }


def test_genes_by_cells_orientation(fake_geneformer_repo: Path):
    tokenizer = GeneformerTranscriptomeTokenizer(
        geneformer_repo=fake_geneformer_repo,
        model_input_size=6,
    )

    # Shape is [genes, cells].
    counts = np.array(
        [
            [10.0, 0.0],
            [5.0, 3.0],
            [20.0, 4.0],
        ]
    )
    gene_ids = ["ENSG000001", "ENSG000002", "ENSG000003"]

    tokenized = tokenizer.tokenize_counts(
        counts,
        gene_ids=gene_ids,
        orientation="genes_by_cells",
        cell_ids=["host_a", "host_b"],
    )

    assert tokenized.cell_ids == ["host_a", "host_b"]

    # host_a ranking: B, C, A as in the first test.
    assert tokenized.input_ids[0] == [CLS, TOKEN_B, TOKEN_C, TOKEN_A, EOS]

    # host_b only has B and C detected.
    # B score: 3 / 1 = 3
    # C score: 4 / 5 = 0.8
    assert tokenized.input_ids[1] == [CLS, TOKEN_B, TOKEN_C, EOS]


def test_unknown_genes_are_ignored(fake_geneformer_repo: Path):
    tokenizer = GeneformerTranscriptomeTokenizer(
        geneformer_repo=fake_geneformer_repo,
        model_input_size=6,
    )

    counts = np.array([[10.0, 99.0, 5.0]])
    gene_ids = ["ENSG000001", "ENSG_UNKNOWN", "ENSG000002"]

    tokenized = tokenizer.tokenize_counts(
        counts,
        gene_ids=gene_ids,
        cell_ids=["host_0"],
    )

    assert TOKEN_A in tokenized.input_ids[0]
    assert TOKEN_B in tokenized.input_ids[0]

    # Unknown gene has no token ID and should not appear.
    assert 99 not in tokenized.input_ids[0]


def test_no_tokenizable_cells_raises(fake_geneformer_repo: Path):
    tokenizer = GeneformerTranscriptomeTokenizer(
        geneformer_repo=fake_geneformer_repo,
        model_input_size=6,
    )

    counts = np.array([[10.0, 99.0]])
    gene_ids = ["ENSG_UNKNOWN_1", "ENSG_UNKNOWN_2"]

    with pytest.raises(ValueError, match="No cells were tokenized"):
        tokenizer.tokenize_counts(
            counts,
            gene_ids=gene_ids,
            cell_ids=["host_0"],
        )


def test_tokenize_dataframe_genes_by_cells(fake_geneformer_repo: Path):
    pd = pytest.importorskip("pandas")

    tokenizer = GeneformerTranscriptomeTokenizer(
        geneformer_repo=fake_geneformer_repo,
        model_input_size=6,
    )

    df = pd.DataFrame(
        {
            "ensembl_id": ["ENSG000001", "ENSG000002", "ENSG000003"],
            "host_a": [10.0, 5.0, 20.0],
            "host_b": [0.0, 3.0, 4.0],
        }
    )

    tokenized = tokenizer.tokenize_dataframe(
        df,
        gene_id_column="ensembl_id",
        count_columns=["host_a", "host_b"],
    )

    assert tokenized.cell_ids == ["host_a", "host_b"]
    assert tokenized.input_ids[0] == [CLS, TOKEN_B, TOKEN_C, TOKEN_A, EOS]
    assert tokenized.input_ids[1] == [CLS, TOKEN_B, TOKEN_C, EOS]


def test_tokenize_h5ad_uses_var_and_obs_columns(fake_geneformer_repo: Path, tmp_path: Path):
    ad = pytest.importorskip("anndata")
    sp = pytest.importorskip("scipy.sparse")

    tokenizer = GeneformerTranscriptomeTokenizer(
        geneformer_repo=fake_geneformer_repo,
        model_input_size=6,
    )

    adata = ad.AnnData(
        X=sp.csr_matrix(
            np.array(
                [
                    [10.0, 5.0, 20.0],
                    [0.0, 3.0, 4.0],
                ]
            )
        )
    )
    adata.obs_names = ["cell_a", "cell_b"]
    adata.var_names = ["gene_a", "gene_b", "gene_c"]
    adata.var["ensembl_id"] = ["ENSG000001", "ENSG000002", "ENSG000003"]
    adata.obs["n_counts"] = [35.0, 7.0]
    adata.obs["cell_type"] = ["type_a", "type_b"]

    path = tmp_path / "host_cells.h5ad"
    adata.write_h5ad(path)

    tokenized = tokenizer.tokenize_h5ad(
        path,
        ensembl_id_column="ensembl_id",
        n_counts_column="n_counts",
        custom_attr_name_dict={"cell_type": "cell_type"},
    )

    assert tokenized.cell_ids == ["cell_a", "cell_b"]
    assert tokenized.metadata["cell_type"] == ["type_a", "type_b"]
    assert tokenized.input_ids[0] == [CLS, TOKEN_B, TOKEN_C, TOKEN_A, EOS]
    assert tokenized.input_ids[1] == [CLS, TOKEN_B, TOKEN_C, EOS]