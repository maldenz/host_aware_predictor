---
library_name: transformers
pipeline_tag: fill-mask
tags:
  - genomics
  - dna
  - masked-lm
  - ntv3
  - long-range
  - base-model
license: other
language:
  - code
model_parameter_count: 7692475
---

# InstaDeepAI/ntv3_base_model

**Unified base model repository for NTv3 models.**

This repository contains shared modeling code used by both:
- **Pre-trained models** (masked language models)
- **Post-trained models** (conditioned multi-species models with functional genomics heads)

**Note:** This repo should not be used standalone. It provides modeling code that is referenced by individual model checkpoints via `trust_remote_code=True`.

## Contents

| File | Purpose |
|------|---------|
| `configuration_ntv3_pretrained.py` | Config class: `Ntv3PreTrainedConfig` |
| `configuration_ntv3_posttrained.py` | Config classes: `DiscreteConditionedNTv3Config`, `NTv3PostTrainedConfig` |
| `modeling_ntv3_pretrained.py` | Pre-trained model: `NTv3PreTrained` |
| `modeling_ntv3_posttrained.py` | Post-trained model: `NTv3PostTrained` with conditioned towers and heads |
| `tokenization_ntv3.py` | Tokenizer: `NTv3Tokenizer` (DNA) |

## Architecture

- U-Net style conv tower → Transformer stack → deconv tower → LM head
- Post-trained models add adaptive layer norms and multi-species prediction heads
- Tokenizer: character-level over A T C G N + specials (`<unk>` `<pad>` `<mask>` `<cls>` `<eos>` `<bos>`)
