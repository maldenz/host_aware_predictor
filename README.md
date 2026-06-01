# Host-Aware Predictor

Fresh first-commit scaffold for a minimal host-aware predictor.

The first version intentionally keeps only one fusion strategy: **concat fusion**.

It includes:

- Nucleotide Transformer sequence encoder using Hugging Face `AutoTokenizer` and `AutoModelForMaskedLM`
- host categorical embedding
- concat fusion head
- train and predict scripts
- no checked-in `data/`, `logs/`, or previous-project files
- a reserved `external/geneformer/` placeholder for adding Geneformer later

## Project layout

```text
host-aware-predictor/
├── .gitignore
├── README.md
├── pyproject.toml
├── requirements.txt
├── external/
│   └── geneformer/
│       └── README.md
├── scripts/
│   ├── predict.py
│   ├── smoke_test.py
│   └── train.py
└── src/
    └── host_aware_predictor/
        ├── __init__.py
        ├── checkpoint.py
        ├── data.py
        ├── inference.py
        ├── metrics.py
        ├── training.py
        └── models/
            ├── __init__.py
            ├── concat_fusion.py
            ├── geneformer_placeholder.py
            ├── nucleotide_transformer.py
            └── predictor.py
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For CUDA-enabled PyTorch, install the PyTorch build that matches your CUDA version before `pip install -e .`.

## Smoke test without downloading the NT checkpoint

The smoke test uses a deterministic tiny hash encoder instead of Hugging Face, so it can verify the training and inference scripts quickly.

```bash
python scripts/smoke_test.py --output-dir ./smoke_run
```

Expected outputs:

```text
smoke_run/
├── checkpoint/
│   ├── config.json
│   ├── host_vocab.json
│   ├── metrics.json
│   └── model_head.pt
└── predictions.csv
```

`smoke_run/` is ignored by git.

## Train with Nucleotide Transformer

Default NT model:

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species")
model = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species")
```

This project wraps the same Hugging Face loading pattern in `NucleotideTransformerEncoder`.

```bash
python scripts/train.py \
  --train-csv /path/to/train.csv \
  --val-csv /path/to/val.csv \
  --output-dir /path/to/checkpoint \
  --task-type binary \
  --nt-model-name InstaDeepAI/nucleotide-transformer-2.5b-multi-species \
  --nt-max-length 1024 \
  --batch-size 2 \
  --epochs 3
```

The first commit freezes the NT encoder and trains only the host embedding plus concat fusion head. Checkpoints save only the lightweight head state, not the 2.5B NT weights.

## Predict

```bash
python scripts/predict.py \
  --checkpoint-dir /path/to/checkpoint \
  --input-csv /path/to/input.csv \
  --output-csv /path/to/predictions.csv
```

## Input CSV schema

Default column names:

| column | required | description |
|---|---:|---|
| `sequence` | yes | nucleotide sequence string |
| `host` | yes | host category, such as species or cell type |
| `label` | train only | target label |

Example:

```csv
sequence,host,label
ACGTACGTACGT,human,1
TTTTGGGGCCCC,mouse,0
AACCGGTTAACC,human,1
```

Use CLI flags to change the column names:

```bash
python scripts/train.py \
  --train-csv train.csv \
  --output-dir checkpoint \
  --sequence-column nt_sequence \
  --host-column host_name \
  --label-column target
```

## Tasks

Supported first-commit task types:

- `binary`: one output logit, trained with BCE-with-logits
- `multiclass`: one output per class, trained with cross entropy
- `regression`: one output value, trained with MSE

## Geneformer space

Geneformer is intentionally not vendored in this commit.

When ready, upload or clone Geneformer into:

```text
external/geneformer/
```

Then implement the adapter in:

```text
src/host_aware_predictor/models/geneformer_placeholder.py
```

The expected future adapter contract is simple:

```python
forward(...) -> torch.Tensor  # shape: [batch_size, geneformer_embedding_dim]
```

The predictor already supports concat-style extension, so later Geneformer embeddings can be concatenated with:

```text
[NT sequence embedding, host embedding, Geneformer embedding]
```

without introducing attention, gating, cross-fusion, or other fusion layers.
